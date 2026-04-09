"""
KB-Loyal Backend - Soporte técnico con IA + RAG + Freshdesk
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import os, uvicorn, logging

from routes.chat import router as chat_router
from routes.freshdesk import router as freshdesk_router
from routes.knowledge import router as knowledge_router
from routes.image import router as image_router
from services.vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

vector_store = VectorStore()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Iniciando KB-Loyal Backend...")
    await vector_store.initialize()
    app.state.vector_store = vector_store
    yield
    logger.info("Apagando KB-Loyal Backend...")

app = FastAPI(
    title="KB-Loyal API",
    description="Base de Conocimiento con IA para Soporte Técnico",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api/chat", tags=["Chat"])
app.include_router(freshdesk_router, prefix="/api/freshdesk", tags=["Freshdesk"])
app.include_router(knowledge_router, prefix="/api/knowledge", tags=["Knowledge Base"])
app.include_router(image_router, prefix="/api/image", tags=["Image Analysis"])

# Servir el frontend
frontend_path = os.path.join(os.path.dirname(__file__), "..")
if os.path.exists(os.path.join(frontend_path, "base-de-conocimiento-ia.html")):
    @app.get("/")
    async def serve_frontend():
        return FileResponse(os.path.join(frontend_path, "base-de-conocimiento-ia.html"))

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
