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
from sentence_transformers import SentenceTransformer
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

# ── Configuración ──────────────────────────────────────────────
CHROMA_DIR   = Path(os.getenv("CHROMA_DIR", "data/chroma_db"))
COLLECTION   = "guias_medicas"
MODEL_NAME   = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
N_RESULTS    = 10        # ↑ de 5 a 10 para tener más candidatos
HAIKU_MODEL  = "claude-haiku-4-5"

# Expansión de consulta: sinónimos clínicos frecuentes
EXPANSIONES = {
    "tratamiento":    ["tratamiento", "manejo", "terapéutica", "esquema farmacológico", "medicamento"],
    "diagnóstico":    ["diagnóstico", "criterios diagnósticos", "identificación", "detección"],
    "síntomas":       ["síntomas", "manifestaciones clínicas", "signos", "cuadro clínico"],
    "dosis":          ["dosis", "dosificación", "posología", "cantidad"],
    "prevención":     ["prevención", "profilaxis", "medidas preventivas"],
    "complicaciones": ["complicaciones", "efectos adversos", "riesgos", "pronóstico"],
    "seguimiento":    ["seguimiento", "monitoreo", "control", "vigilancia"],
}

# ── Inicialización ─────────────────────────────────────────────
print("Cargando modelo de embeddings...")
embed_model = SentenceTransformer(MODEL_NAME)

print("Conectando a ChromaDB...")

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

# ── Query expansion ────────────────────────────────────────────
def expandir_consulta(texto: str) -> list[str]:
    """
    Genera variantes de búsqueda para mejorar recall semántico.
    Siempre incluye la pregunta original + variantes por sinónimos clínicos.
    """
    variantes = [texto]
    texto_lower = texto.lower()
    for termino, sinonimos in EXPANSIONES.items():
        if termino in texto_lower:
            for sin in sinonimos:
                if sin != termino:
                    variante = texto_lower.replace(termino, sin)
                    if variante not in variantes:
                        variantes.append(variante)
            break  # solo expandir el primer término encontrado
    return variantes[:4]  # máximo 4 variantes para no disparar latencia


def buscar_chunks(texto: str) -> list[dict]:
    """
    Búsqueda con query expansion: combina resultados de múltiples variantes,
    deduplica por id y ordena por distancia (menor = más relevante).
    """
    variantes = expandir_consulta(texto)
    embeddings = embed_model.encode(variantes).tolist()

    vistos = {}  # id → chunk con menor distancia

    for emb in embeddings:
        resultados = collection.query(
            query_embeddings=[emb],
            n_results=N_RESULTS,
            include=["documents", "metadatas", "distances"]
        )
        for doc, meta, dist in zip(
            resultados["documents"][0],
            resultados["metadatas"][0],
            resultados["distances"][0]
        ):
            chunk_id = f"{meta['archivo']}__p{meta['pagina']}"
            if chunk_id not in vistos or dist < vistos[chunk_id]["distancia"]:
                vistos[chunk_id] = {
                    "texto":     doc,
                    "archivo":   meta["archivo"],
                    "pagina":    meta["pagina"],
                    "distancia": dist,
                }

    # Ordenar por relevancia y devolver los 8 mejores
    ordenados = sorted(vistos.values(), key=lambda x: x["distancia"])
    return ordenados[:8]


# ── Prompt ─────────────────────────────────────────────────────
SYSTEM_PROMPT = """Eres un asistente médico especializado en Guías de Práctica Clínica (GPC) mexicanas del IMSS.

Tu tarea es responder preguntas usando ÚNICAMENTE la información de los fragmentos proporcionados.

REGLAS:
1. Si la pregunta tiene opciones (A), B), C)...), identifica cuál es la correcta según la GPC y explica por qué.
2. Cita el fragmento exacto de la GPC que respalda tu respuesta (entre comillas).
3. Si el fragmento contiene una tabla o algoritmo, descríbelo claramente en tu respuesta.
4. Sé directo y preciso. No inventes información que no esté en los fragmentos.
5. Si los fragmentos no contienen información suficiente para responder, dilo explícitamente indicando qué sí encontraste.
6. Responde siempre en español médico claro."""

def construir_prompt(pregunta: str, chunks: list[dict]) -> str:
    contexto = ""
    for i, chunk in enumerate(chunks, 1):
        contexto += (
            f"\n[Fragmento {i} | {chunk['archivo']} | p.{chunk['pagina']}]\n"
            f"{chunk['texto']}\n"
            f"{'─'*60}\n"
        )

    return f"""Usa los siguientes fragmentos de Guías de Práctica Clínica para responder la pregunta.

FRAGMENTOS:
{contexto}

PREGUNTA:
{pregunta}

INSTRUCCIONES:
- Si hay opciones (A, B, C...), indica cuál es correcta y cita el fragmento que lo justifica.
- Si hay tablas o algoritmos en los fragmentos, incorpóralos en tu respuesta.
- Indica siempre la fuente al final: [Fuente: NOMBRE_PDF, p. X]
- Si ningún fragmento responde directamente, indica qué información sí encontraste y sugiere reformular la pregunta."""


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

        chunks = buscar_chunks(body.texto)

        mensaje = ai_client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=1500,   # ↑ de 1024 a 1500 para tablas/algoritmos
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
