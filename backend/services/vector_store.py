"""
Vector Store - ChromaDB para RAG (Retrieval-Augmented Generation)
Almacena y busca documentos por similitud semántica.
"""
import chromadb
from chromadb.config import Settings
import logging, os, json, re
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma")


def _clean_text(text: str) -> str:
    """Elimina HTML y normaliza el texto."""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


class VectorStore:
    def __init__(self):
        self.client: Optional[chromadb.Client] = None
        self.collection = None
        self._initialized = False

    async def initialize(self):
        os.makedirs(CHROMA_DIR, exist_ok=True)
        os.makedirs(DATA_DIR, exist_ok=True)
        try:
            self.client = chromadb.PersistentClient(
                path=CHROMA_DIR,
                settings=Settings(anonymized_telemetry=False)
            )
            self.collection = self.client.get_or_create_collection(
                name="kb_loyal",
                metadata={"hnsw:space": "cosine"}
            )
            self._initialized = True
            count = self.collection.count()
            logger.info(f"VectorStore listo. Documentos indexados: {count}")
        except Exception as e:
            logger.error(f"Error inicializando VectorStore: {e}")
            raise

    def _embed_text(self, text: str) -> List[float]:
        """
        Embedding simple basado en TF-IDF aproximado.
        En producción reemplazar con sentence-transformers o API de embeddings.
        ChromaDB usa su propio modelo de embedding por defecto (all-MiniLM-L6-v2).
        """
        return None  # ChromaDB genera embeddings automáticamente

    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        Agrega documentos al vector store.
        Cada documento debe tener: id, text, metadata
        """
        if not self._initialized:
            raise RuntimeError("VectorStore no inicializado")

        ids, texts, metadatas = [], [], []
        for doc in documents:
            doc_id = str(doc["id"])
            text = _clean_text(doc.get("text", ""))
            if not text:
                continue
            # Truncar a 8000 chars para evitar límites
            text = text[:8000]
            ids.append(doc_id)
            texts.append(text)
            metadatas.append({
                k: str(v) if v is not None else ""
                for k, v in doc.get("metadata", {}).items()
            })

        if not ids:
            return 0

        # Upsert para evitar duplicados
        existing = set(self.collection.get(ids=ids)["ids"])
        new_ids = [i for i in ids if i not in existing]
        new_texts = [texts[ids.index(i)] for i in new_ids]
        new_metas = [metadatas[ids.index(i)] for i in new_ids]

        if new_ids:
            self.collection.add(
                ids=new_ids,
                documents=new_texts,
                metadatas=new_metas
            )
        logger.info(f"Agregados {len(new_ids)} nuevos documentos (de {len(ids)} enviados)")
        return len(new_ids)

    def upsert_documents(self, documents: List[Dict[str, Any]]) -> int:
        """Inserta o actualiza documentos."""
        if not self._initialized:
            raise RuntimeError("VectorStore no inicializado")

        ids, texts, metadatas = [], [], []
        for doc in documents:
            text = _clean_text(doc.get("text", ""))
            if not text:
                continue
            text = text[:8000]
            ids.append(str(doc["id"]))
            texts.append(text)
            metadatas.append({
                k: str(v) if v is not None else ""
                for k, v in doc.get("metadata", {}).items()
            })

        if not ids:
            return 0

        self.collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
        return len(ids)

    def search(self, query: str, n_results: int = 5, filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """Busca documentos similares a la consulta."""
        if not self._initialized:
            return []

        count = self.collection.count()
        if count == 0:
            return []

        n_results = min(n_results, count)
        query = _clean_text(query)

        kwargs = {
            "query_texts": [query],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"]
        }
        if filter_metadata:
            kwargs["where"] = filter_metadata

        results = self.collection.query(**kwargs)

        docs = []
        for i, doc_id in enumerate(results["ids"][0]):
            docs.append({
                "id": doc_id,
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i]  # cosine similarity
            })
        return docs

    def get_stats(self) -> Dict:
        if not self._initialized:
            return {"total": 0, "initialized": False}
        return {
            "total": self.collection.count(),
            "initialized": True
        }

    def delete_by_source(self, source: str):
        """Elimina todos los documentos de una fuente específica."""
        try:
            self.collection.delete(where={"source": source})
        except Exception as e:
            logger.warning(f"Error eliminando por source={source}: {e}")
