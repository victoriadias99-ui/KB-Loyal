"""
Image Route - Análisis de imágenes con visión artificial
Detecta errores en capturas y busca soluciones en la KB.
"""
from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from typing import Optional
import logging

from services.rag_service import rag_service

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_TYPES = {
    "image/jpeg": "image/jpeg",
    "image/jpg": "image/jpeg",
    "image/png": "image/png",
    "image/gif": "image/gif",
    "image/webp": "image/webp"
}

MAX_SIZE_MB = 10


@router.post("/analyze")
async def analyze_image(
    request: Request,
    file: UploadFile = File(...),
    extract_text_only: bool = False
):
    """
    Analiza una imagen de error/captura y busca soluciones.
    - Detecta el error visible en la imagen
    - Busca tickets y artículos similares
    - Devuelve una respuesta contextual
    """
    # Validar tipo
    content_type = file.content_type or ""
    media_type = ALLOWED_TYPES.get(content_type.lower())
    if not media_type:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de archivo no soportado: {content_type}. Use JPEG, PNG, GIF o WebP."
        )

    # Leer y validar tamaño
    image_data = await file.read()
    size_mb = len(image_data) / (1024 * 1024)
    if size_mb > MAX_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"Imagen demasiado grande ({size_mb:.1f}MB). Máximo {MAX_SIZE_MB}MB."
        )

    try:
        vector_store = request.app.state.vector_store

        if extract_text_only:
            # Solo extrae texto (OCR)
            text = await rag_service.extract_text_from_image(image_data, media_type)
            return {"extracted_text": text}

        # 1. Extraer texto de la imagen para la búsqueda semántica
        extracted_text = await rag_service.extract_text_from_image(image_data, media_type)

        # 2. Buscar documentos similares usando el texto extraído
        retrieved = []
        if extracted_text.strip():
            retrieved = vector_store.search(
                query=extracted_text,
                n_results=5
            )

        # 3. Analizar imagen con contexto de KB
        analysis = await rag_service.analyze_image(
            image_data=image_data,
            media_type=media_type,
            retrieved_docs=retrieved
        )

        # 4. Formatear fuentes
        sources = []
        for doc in retrieved:
            meta = doc.get("metadata", {})
            source_type = meta.get("source", "")
            if source_type == "freshdesk_ticket":
                sources.append({
                    "type": "ticket",
                    "id": meta.get("ticket_id", ""),
                    "title": meta.get("subject", ""),
                    "score": round(doc.get("score", 0), 3)
                })
            elif source_type in ("freshdesk_article", "manual"):
                sources.append({
                    "type": "article",
                    "title": meta.get("title", ""),
                    "score": round(doc.get("score", 0), 3)
                })

        return {
            "analysis": analysis,
            "extracted_text": extracted_text,
            "sources": sources,
            "filename": file.filename
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error analizando imagen: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error analizando imagen: {str(e)}")
