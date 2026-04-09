"""
Freshdesk Service - Integración con la API de Freshdesk
Extrae tickets, respuestas y artículos de knowledge base.
"""
import httpx
import logging
import os
import base64
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

FRESHDESK_DOMAIN = os.getenv("FRESHDESK_DOMAIN", "")       # ej: tuempresa.freshdesk.com
FRESHDESK_API_KEY = os.getenv("FRESHDESK_API_KEY", "")


def _get_auth_headers() -> Dict[str, str]:
    token = base64.b64encode(f"{FRESHDESK_API_KEY}:X".encode()).decode()
    return {
        "Authorization": f"Basic {token}",
        "Content-Type": "application/json"
    }


def _base_url() -> str:
    domain = FRESHDESK_DOMAIN.replace("https://", "").replace("http://", "").rstrip("/")
    return f"https://{domain}/api/v2"


class FreshdeskService:
    def __init__(self):
        self.timeout = 30.0

    def _check_config(self):
        if not FRESHDESK_DOMAIN or not FRESHDESK_API_KEY:
            raise ValueError(
                "Freshdesk no configurado. "
                "Definí FRESHDESK_DOMAIN y FRESHDESK_API_KEY en .env"
            )

    async def get_tickets(
        self,
        page: int = 1,
        per_page: int = 100,
        status: Optional[int] = None,
        include_conversations: bool = True
    ) -> List[Dict]:
        """
        Extrae tickets de Freshdesk.
        status: 2=abierto, 3=pendiente, 4=resuelto, 5=cerrado
        """
        self._check_config()
        params = {"page": page, "per_page": per_page}
        if status:
            params["status"] = status

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(
                f"{_base_url()}/tickets",
                headers=_get_auth_headers(),
                params=params
            )
            resp.raise_for_status()
            tickets = resp.json()

        if include_conversations:
            enriched = []
            for ticket in tickets:
                try:
                    convs = await self.get_ticket_conversations(ticket["id"])
                    ticket["conversations"] = convs
                except Exception as e:
                    logger.warning(f"No se pudo obtener conversaciones del ticket {ticket['id']}: {e}")
                    ticket["conversations"] = []
                enriched.append(ticket)
            return enriched

        return tickets

    async def get_all_resolved_tickets(self, max_pages: int = 10) -> List[Dict]:
        """Extrae todos los tickets resueltos/cerrados para la base de conocimiento."""
        all_tickets = []
        for page in range(1, max_pages + 1):
            # Resueltos (4) y Cerrados (5)
            for status in [4, 5]:
                try:
                    tickets = await self.get_tickets(
                        page=page, per_page=100, status=status,
                        include_conversations=False
                    )
                    all_tickets.extend(tickets)
                    if len(tickets) < 100:
                        break
                except Exception as e:
                    logger.error(f"Error obteniendo tickets página {page} status {status}: {e}")
                    break
        logger.info(f"Total tickets extraídos: {len(all_tickets)}")
        return all_tickets

    async def get_ticket_conversations(self, ticket_id: int) -> List[Dict]:
        """Obtiene todas las conversaciones de un ticket."""
        self._check_config()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(
                f"{_base_url()}/tickets/{ticket_id}/conversations",
                headers=_get_auth_headers()
            )
            resp.raise_for_status()
            return resp.json()

    async def get_solution_articles(
        self,
        folder_id: Optional[int] = None,
        page: int = 1,
        per_page: int = 100
    ) -> List[Dict]:
        """Extrae artículos de la knowledge base de Freshdesk."""
        self._check_config()
        endpoint = (
            f"{_base_url()}/solutions/folders/{folder_id}/articles"
            if folder_id
            else f"{_base_url()}/solutions/articles"
        )
        params = {"page": page, "per_page": per_page}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(
                endpoint,
                headers=_get_auth_headers(),
                params=params
            )
            resp.raise_for_status()
            return resp.json()

    async def get_all_solution_folders(self) -> List[Dict]:
        """Obtiene todas las carpetas de soluciones."""
        self._check_config()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(
                f"{_base_url()}/solutions/categories",
                headers=_get_auth_headers()
            )
            resp.raise_for_status()
            categories = resp.json()

        folders = []
        for cat in categories:
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.get(
                        f"{_base_url()}/solutions/categories/{cat['id']}/folders",
                        headers=_get_auth_headers()
                    )
                    resp.raise_for_status()
                    folders.extend(resp.json())
            except Exception as e:
                logger.warning(f"Error obteniendo folders de categoría {cat['id']}: {e}")

        return folders

    async def get_all_articles(self) -> List[Dict]:
        """Extrae todos los artículos de solución de todas las carpetas."""
        all_articles = []
        try:
            folders = await self.get_all_solution_folders()
        except Exception as e:
            logger.error(f"Error obteniendo folders: {e}")
            return []

        for folder in folders:
            page = 1
            while True:
                try:
                    articles = await self.get_solution_articles(
                        folder_id=folder["id"],
                        page=page
                    )
                    all_articles.extend(articles)
                    if len(articles) < 100:
                        break
                    page += 1
                except Exception as e:
                    logger.warning(f"Error obteniendo artículos folder {folder['id']}: {e}")
                    break

        logger.info(f"Total artículos extraídos: {len(all_articles)}")
        return all_articles

    def normalize_ticket(self, ticket: Dict) -> Dict:
        """Normaliza un ticket para indexarlo en el vector store."""
        subject = ticket.get("subject", "")
        description = ticket.get("description_text", ticket.get("description", ""))
        status_map = {2: "abierto", 3: "pendiente", 4: "resuelto", 5: "cerrado"}
        status = status_map.get(ticket.get("status", 0), "desconocido")
        priority_map = {1: "baja", 2: "media", 3: "alta", 4: "urgente"}
        priority = priority_map.get(ticket.get("priority", 0), "media")

        conversations_text = ""
        for conv in ticket.get("conversations", []):
            body = conv.get("body_text", conv.get("body", ""))
            if body:
                conversations_text += f"\n---\n{body}"

        full_text = f"TICKET #{ticket.get('id')}: {subject}\n\n{description}{conversations_text}"

        return {
            "id": f"ticket_{ticket.get('id')}",
            "text": full_text,
            "metadata": {
                "source": "freshdesk_ticket",
                "ticket_id": str(ticket.get("id", "")),
                "subject": subject[:200],
                "status": status,
                "priority": priority,
                "created_at": ticket.get("created_at", ""),
                "updated_at": ticket.get("updated_at", ""),
                "tags": ", ".join(ticket.get("tags", [])),
                "type": ticket.get("type", ""),
            }
        }

    def normalize_article(self, article: Dict) -> Dict:
        """Normaliza un artículo de solución para indexarlo."""
        title = article.get("title", "")
        description = article.get("description_text", article.get("description", ""))
        full_text = f"ARTÍCULO: {title}\n\n{description}"

        return {
            "id": f"article_{article.get('id')}",
            "text": full_text,
            "metadata": {
                "source": "freshdesk_article",
                "article_id": str(article.get("id", "")),
                "title": title[:200],
                "status": str(article.get("status", "")),
                "created_at": article.get("created_at", ""),
                "updated_at": article.get("updated_at", ""),
                "folder_id": str(article.get("folder_id", "")),
            }
        }


freshdesk_service = FreshdeskService()
