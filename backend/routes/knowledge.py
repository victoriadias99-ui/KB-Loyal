"""
Knowledge Base Route - Gestión de artículos internos (manuales)
"""
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter()


class ArticleCreate(BaseModel):
    title: str
    content: str
    category: str = "tech"
    author: str = "Anónimo"
    tags: Optional[List[str]] = []


class ArticleUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    category: Optional[str] = None
    author: Optional[str] = None


@router.post("/articles")
async def create_article(request: Request, body: ArticleCreate):
    """Crea un artículo manual e indexa en vector store."""
    vector_store = request.app.state.vector_store

    article_id = f"manual_{int(datetime.now().timestamp() * 1000)}"
    full_text = f"ARTÍCULO: {body.title}\n\n{body.content}"

    doc = {
        "id": article_id,
        "text": full_text,
        "metadata": {
            "source": "manual",
            "title": body.title[:200],
            "category": body.category,
            "author": body.author,
            "tags": ", ".join(body.tags or []),
            "created_at": datetime.now().isoformat(),
        }
    }

    added = vector_store.upsert_documents([doc])
    return {
        "id": article_id,
        "indexed": added > 0,
        "message": "Artículo creado e indexado correctamente"
    }


@router.post("/bulk")
async def bulk_import(request: Request, articles: List[ArticleCreate]):
    """Importa múltiples artículos de una vez."""
    vector_store = request.app.state.vector_store

    docs = []
    for i, article in enumerate(articles):
        article_id = f"manual_{int(datetime.now().timestamp() * 1000)}_{i}"
        docs.append({
            "id": article_id,
            "text": f"ARTÍCULO: {article.title}\n\n{article.content}",
            "metadata": {
                "source": "manual",
                "title": article.title[:200],
                "category": article.category,
                "author": article.author,
                "tags": ", ".join(article.tags or []),
                "created_at": datetime.now().isoformat(),
            }
        })

    added = vector_store.upsert_documents(docs)
    return {"indexed": added, "total": len(docs)}


@router.get("/search")
async def search_knowledge(request: Request, q: str, n: int = 5):
    """Búsqueda semántica directa en la base de conocimiento."""
    vector_store = request.app.state.vector_store
    results = vector_store.search(query=q, n_results=n)
    return {"query": q, "results": results, "count": len(results)}


@router.get("/stats")
async def knowledge_stats(request: Request):
    """Estadísticas de la base de conocimiento."""
    vector_store = request.app.state.vector_store
    return vector_store.get_stats()
