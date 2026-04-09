# KB-Loyal — Base de Conocimiento con IA

Sistema de soporte técnico con RAG (Retrieval-Augmented Generation) integrado con Freshdesk.

## Inicio rápido

```bash
# 1. Instalar dependencias
cd backend
pip install -r requirements.txt

# 2. Configurar variables de entorno
cp .env.example .env
# Editá .env con tus claves reales

# 3. Iniciar el backend
python main.py
# Servidor en http://localhost:8000

# 4. Abrir el frontend
# Abrí base-de-conocimiento-ia.html en el navegador
# O accedé a http://localhost:8000 directamente
```

## Variables de entorno (.env)

| Variable | Descripción | Requerida |
|---|---|---|
| `ANTHROPIC_API_KEY` | Clave de API de Anthropic (Claude) | ✅ Sí |
| `FRESHDESK_DOMAIN` | Dominio de Freshdesk (ej: empresa.freshdesk.com) | Para sync |
| `FRESHDESK_API_KEY` | API Key de Freshdesk | Para sync |

## Endpoints principales

| Método | URL | Descripción |
|---|---|---|
| POST | `/api/chat/` | Chat con RAG |
| POST | `/api/freshdesk/sync` | Sincronizar Freshdesk |
| GET | `/api/freshdesk/sync/status` | Estado del sync |
| POST | `/api/knowledge/articles` | Indexar artículo manual |
| POST | `/api/image/analyze` | Analizar imagen de error |
| GET | `/health` | Health check |

## Arquitectura

```
Frontend HTML
    ↓ fetch()
FastAPI Backend
    ├── /api/chat      → VectorStore.search() → Claude RAG
    ├── /api/freshdesk → Freshdesk API → VectorStore.upsert()
    ├── /api/knowledge → VectorStore.upsert()
    └── /api/image     → Claude Vision → VectorStore.search() → RAG
         ↓
    ChromaDB (./data/chroma)  ← persiste en disco
```

## Cómo funciona el RAG

1. El usuario escribe una consulta o sube una imagen
2. El backend busca los 5 documentos más similares en ChromaDB (búsqueda semántica)
3. Esos documentos se pasan como contexto a Claude junto con la consulta
4. Claude responde basándose exclusivamente en ese contexto
5. El frontend muestra la respuesta con las fuentes citadas

## Obtener API Keys

- **Anthropic**: https://console.anthropic.com/settings/keys
- **Freshdesk**: Ir a Admin → API Settings en tu cuenta de Freshdesk
