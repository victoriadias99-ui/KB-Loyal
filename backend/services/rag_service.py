"""
RAG Service - Retrieval-Augmented Generation con Claude
Busca contexto relevante y genera respuestas fundamentadas.
"""
import anthropic
import logging
import os
import base64
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MODEL = "claude-sonnet-4-20250514"

SYSTEM_PROMPT = """Eres un asistente experto en soporte técnico. Tu función es ayudar al equipo de soporte a resolver problemas utilizando la base de conocimiento interna.

INSTRUCCIONES:
1. Responde SIEMPRE basándote en el contexto proporcionado (tickets anteriores, artículos de solución).
2. Si encontrás un ticket o artículo que resuelve el problema, mencioná el ID y la solución específica.
3. Priorizá soluciones que ya fueron aplicadas exitosamente.
4. Si el contexto no contiene información relevante, indicalo claramente y sugierí crear un nuevo ticket o artículo.
5. Usa formato claro: pasos numerados cuando corresponda, resaltá datos importantes.
6. Responde en español.
7. Sé conciso pero completo. No inventes información que no esté en el contexto.

FORMATO DE RESPUESTA:
- Si encontrás soluciones previas: "Encontré casos similares: [detalle]"
- Si es información nueva: "No encontré casos similares en la base de conocimiento, pero puedo sugerir:"
- Siempre indicá la fuente: (Ticket #X, Artículo: Título)"""


class RAGService:
    def __init__(self):
        self.client: Optional[anthropic.Anthropic] = None

    def _get_client(self) -> anthropic.Anthropic:
        if not self.client:
            if not ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY no configurada")
            self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        return self.client

    def _build_context(self, retrieved_docs: List[Dict]) -> str:
        """Construye el contexto para el prompt a partir de documentos recuperados."""
        if not retrieved_docs:
            return "No se encontraron documentos relevantes en la base de conocimiento."

        context_parts = ["=== BASE DE CONOCIMIENTO RELEVANTE ===\n"]
        for i, doc in enumerate(retrieved_docs, 1):
            meta = doc.get("metadata", {})
            source = meta.get("source", "")
            score = doc.get("score", 0)

            if source == "freshdesk_ticket":
                header = f"[{i}] TICKET #{meta.get('ticket_id', '?')} | Estado: {meta.get('status', '?')} | Relevancia: {score:.0%}"
            elif source == "freshdesk_article":
                header = f"[{i}] ARTÍCULO: {meta.get('title', '?')} | Relevancia: {score:.0%}"
            elif source == "manual":
                header = f"[{i}] ARTÍCULO INTERNO: {meta.get('title', '?')} | Relevancia: {score:.0%}"
            else:
                header = f"[{i}] DOCUMENTO | Relevancia: {score:.0%}"

            context_parts.append(f"{header}\n{doc['text'][:1500]}\n")

        return "\n".join(context_parts)

    async def chat(
        self,
        query: str,
        retrieved_docs: List[Dict],
        conversation_history: Optional[List[Dict]] = None
    ) -> str:
        """Genera respuesta basada en RAG."""
        client = self._get_client()
        context = self._build_context(retrieved_docs)

        system = SYSTEM_PROMPT + f"\n\n{context}"

        messages = []
        if conversation_history:
            messages.extend(conversation_history[-6:])  # Últimos 3 intercambios
        messages.append({"role": "user", "content": query})

        response = client.messages.create(
            model=MODEL,
            max_tokens=1500,
            system=system,
            messages=messages
        )
        return response.content[0].text

    async def analyze_image(
        self,
        image_data: bytes,
        media_type: str,
        retrieved_docs: List[Dict]
    ) -> str:
        """Analiza una imagen (captura de error) y busca soluciones."""
        client = self._get_client()
        context = self._build_context(retrieved_docs)

        image_b64 = base64.standard_b64encode(image_data).decode("utf-8")

        system = """Eres un experto en soporte técnico con capacidad de analizar capturas de pantalla y errores.

TAREA:
1. Analiza la imagen enviada e identifica:
   - El error o problema visible
   - Mensajes de error específicos
   - Código de error si existe
   - Aplicación o sistema afectado

2. Busca en la base de conocimiento proporcionada casos similares.

3. Proporciona una respuesta estructurada con:
   - Descripción del problema detectado
   - Soluciones encontradas en la base de conocimiento (si existen)
   - Pasos de resolución recomendados
   - Si no hay casos similares, sugiere pasos de diagnóstico

Responde en español. Sé específico y práctico."""

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64
                        }
                    },
                    {
                        "type": "text",
                        "text": f"Analizá esta captura de pantalla y buscá soluciones en la base de conocimiento.\n\n{context}"
                    }
                ]
            }
        ]

        response = client.messages.create(
            model=MODEL,
            max_tokens=1500,
            system=system,
            messages=messages
        )
        return response.content[0].text

    async def extract_text_from_image(self, image_data: bytes, media_type: str) -> str:
        """Extrae texto visible en una imagen (OCR con vision)."""
        client = self._get_client()
        image_b64 = base64.standard_b64encode(image_data).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64
                        }
                    },
                    {
                        "type": "text",
                        "text": "Extrae todo el texto visible en esta imagen. Solo devuelve el texto extraído, sin comentarios adicionales."
                    }
                ]
            }
        ]

        response = client.messages.create(
            model=MODEL,
            max_tokens=500,
            messages=messages
        )
        return response.content[0].text


rag_service = RAGService()
