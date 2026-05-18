"""
Backend RAG - Guías Médicas
POST /preguntar  → pregunta → ChromaDB → Claude Haiku → respuesta + fuente
GET  /health     → Railway healthcheck
GET  /           → sirve el frontend
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import chromadb
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

# ── Configuración ──────────────────────────────────────────────
CHROMA_DIR   = Path(os.getenv("CHROMA_DIR", "data/chroma_db"))
COLLECTION   = "guias_medicas"
N_RESULTS    = 5
HAIKU_MODEL  = "claude-haiku-4-5"

# ── Inicialización ─────────────────────────────────────────────
print("Conectando a ChromaDB...")

# Detectar y limpiar sqlite corrupto
sqlite_path = CHROMA_DIR / "chroma.sqlite3"
if sqlite_path.exists():
    import sqlite3 as _sqlite3
    try:
        _conn = _sqlite3.connect(str(sqlite_path))
        _conn.execute("PRAGMA integrity_check")
        _conn.close()
    except Exception as _e:
        print(f"⚠️  chroma.sqlite3 corrupto ({_e}), eliminando...")
        sqlite_path.unlink()

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

collection = None
try:
    collection = chroma_client.get_collection(COLLECTION)
    print(f"✅ Backend listo — {collection.count()} chunks indexados")
except Exception:
    print("⚠️  Colección no encontrada — sube el chroma_db al Volume y redeploya.")

print("Inicializando cliente Anthropic...")
ai_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ── App ────────────────────────────────────────────────────────
app = FastAPI(title="RAG Guías Médicas")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Modelos ────────────────────────────────────────────────────
class Pregunta(BaseModel):
    texto: str

class Fuente(BaseModel):
    archivo: str
    pagina: int
    fragmento: str

class Respuesta(BaseModel):
    respuesta: str
    fuentes: list[Fuente]

# ── Prompt ─────────────────────────────────────────────────────
SYSTEM_PROMPT = """Eres un asistente médico especializado en Guías de Práctica Clínica (GPC) mexicanas.

Tu tarea es responder preguntas usando ÚNICAMENTE la información de los fragmentos proporcionados.

REGLAS:
1. Si la pregunta tiene opciones (A), B), C)...), indica cuál es la correcta según la GPC.
2. Siempre cita el fragmento exacto de la GPC que respalda tu respuesta.
3. Sé claro y directo. No inventes información que no esté en los fragmentos.
4. Si los fragmentos no contienen información suficiente, dilo explícitamente.
5. Responde siempre en español."""

def construir_prompt(pregunta: str, chunks: list[dict]) -> str:
    contexto = ""
    for i, chunk in enumerate(chunks, 1):
        contexto += (
            f"\n[Fragmento {i}]\n"
            f"Archivo: {chunk['archivo']} | Página: {chunk['pagina']}\n"
            f"Texto: {chunk['texto']}\n"
        )

    return f"""Usa los siguientes fragmentos de Guías de Práctica Clínica para responder la pregunta.

FRAGMENTOS:
{contexto}

PREGUNTA:
{pregunta}

INSTRUCCIONES DE RESPUESTA:
- Si hay opciones (A, B, C...), indica cuál es correcta y por qué según la GPC.
- Cita el fragmento relevante entre comillas.
- Indica el nombre del archivo y página al final como: [Fuente: NOMBRE_PDF, p. X]
- Si ningún fragmento responde la pregunta, dilo claramente."""

# ── Endpoints ──────────────────────────────────────────────────
@app.get("/health")
def health():
    chunks = collection.count() if collection else 0
    return {"status": "ok", "chunks": chunks}

@app.post("/preguntar", response_model=Respuesta)
def preguntar(body: Pregunta):
    import traceback
    try:
        if collection is None:
            raise HTTPException(status_code=503, detail="El índice no está disponible.")

        if not body.texto.strip():
            raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")

        resultados = collection.query(
            query_texts=[body.texto],
            n_results=N_RESULTS,
            include=["documents", "metadatas", "distances"]
        )

        chunks = [
            {
                "texto":   doc,
                "archivo": meta["archivo"],
                "pagina":  meta["pagina"],
            }
            for doc, meta in zip(
                resultados["documents"][0],
                resultados["metadatas"][0]
            )
        ]

        mensaje = ai_client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": construir_prompt(body.texto, chunks)}]
        )

        respuesta_texto = mensaje.content[0].text

        fuentes = [
            Fuente(
                archivo=c["archivo"],
                pagina=c["pagina"],
                fragmento=c["texto"][:300]
            )
            for c in chunks
        ]

        return Respuesta(respuesta=respuesta_texto, fuentes=fuentes)

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ── Frontend (debe ir al final) ────────────────────────────────
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

@app.get("/")
def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")

app.mount("/", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
