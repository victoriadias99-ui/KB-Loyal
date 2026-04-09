"""
Freshdesk Route - Sincronización con Freshdesk
"""
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
import logging

from services.freshdesk_service import freshdesk_service

logger = logging.getLogger(__name__)
router = APIRouter()

sync_status = {
    "running": False,
    "last_sync": None,
    "tickets_indexed": 0,
    "articles_indexed": 0,
    "errors": []
}


async def _run_sync(vector_store, include_tickets: bool, include_articles: bool):
    global sync_status
    sync_status["running"] = True
    sync_status["errors"] = []

    try:
        # Sincronizar tickets
        if include_tickets:
            logger.info("Sincronizando tickets de Freshdesk...")
            tickets = await freshdesk_service.get_all_resolved_tickets(max_pages=10)
            docs = [freshdesk_service.normalize_ticket(t) for t in tickets]
            added = vector_store.upsert_documents(docs)
            sync_status["tickets_indexed"] = added
            logger.info(f"Tickets indexados: {added}")

        # Sincronizar artículos
        if include_articles:
            logger.info("Sincronizando artículos de Freshdesk...")
            articles = await freshdesk_service.get_all_articles()
            docs = [freshdesk_service.normalize_article(a) for a in articles]
            added = vector_store.upsert_documents(docs)
            sync_status["articles_indexed"] = added
            logger.info(f"Artículos indexados: {added}")

    except Exception as e:
        logger.error(f"Error durante sync: {e}", exc_info=True)
        sync_status["errors"].append(str(e))
    finally:
        from datetime import datetime
        sync_status["running"] = False
        sync_status["last_sync"] = datetime.now().isoformat()


class SyncRequest(BaseModel):
    include_tickets: bool = True
    include_articles: bool = True


@router.post("/sync")
async def sync_freshdesk(
    request: Request,
    body: SyncRequest,
    background_tasks: BackgroundTasks
):
    """Inicia la sincronización con Freshdesk en background."""
    if sync_status["running"]:
        raise HTTPException(status_code=409, detail="Ya hay una sincronización en curso")

    vector_store = request.app.state.vector_store
    background_tasks.add_task(
        _run_sync,
        vector_store,
        body.include_tickets,
        body.include_articles
    )
    return {"message": "Sincronización iniciada", "status": "running"}


@router.get("/sync/status")
async def get_sync_status():
    """Estado de la última sincronización."""
    return sync_status


@router.get("/test")
async def test_connection():
    """Prueba la conexión con Freshdesk."""
    try:
        # Intenta obtener 1 ticket para verificar conexión
        tickets = await freshdesk_service.get_tickets(page=1, per_page=1, include_conversations=False)
        return {
            "connected": True,
            "sample_count": len(tickets)
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error conectando a Freshdesk: {str(e)}")
