"""
Chat Route - Endpoint principal de chat con RAG
"""
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import logging

from services.rag_service import rag_service

logger = logging.getLogger(__name__)
router = APIRouter()


class Message(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    query: str
    history: Optional[List[Message]] = []
    n_results: int = 5


class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict]
    query: str


@router.post("/", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest):
    """Chat principal con RAG - busca contexto y genera respuesta."""
    try:
        vector_store = request.app.state.vector_store

        # 1. Recuperar documentos relevantes
        retrieved = vector_store.search(
            query=body.query,
            n_results=body.n_results
        )

        # 2. Preparar historial
        history = [
            {"role": m.role, "content": m.content}
            for m in (body.history or [])
        ]

        # 3. Generar respuesta con RAG
        answer = await rag_service.chat(
            query=body.query,
            retrieved_docs=retrieved,
            conversation_history=history
        )

        # 4. Formatear fuentes para el frontend
        sources = []
        for doc in retrieved:
            meta = doc.get("metadata", {})
            source_type = meta.get("source", "")
            if source_type == "freshdesk_ticket":
                sources.append({
                    "type": "ticket",
                    "id": meta.get("ticket_id", ""),
                    "title": meta.get("subject", ""),
                    "status": meta.get("status", ""),
                    "score": round(doc.get("score", 0), 3)
                })
            elif source_type in ("freshdesk_article", "manual"):
                sources.append({
                    "type": "article",
                    "id": meta.get("article_id", meta.get("id", "")),
                    "title": meta.get("title", ""),
                    "score": round(doc.get("score", 0), 3)
                })

        return ChatResponse(answer=answer, sources=sources, query=body.query)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error en chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@router.get("/stats")
async def get_stats(request: Request):
    """Estadísticas de la base de conocimiento."""
    vector_store = request.app.state.vector_store
    return vector_store.get_stats()
