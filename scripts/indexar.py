"""
Script de indexación: lee los 1,285 PDFs y los almacena en ChromaDB.
Ejecutar UNA sola vez. Tarda ~2-3 horas.

Uso:
    python scripts/indexar.py
"""

import os
import fitz  # PyMuPDF
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pathlib import Path

# ── Configuración ────────────────────────────────────────────────────────────
PDF_DIR    = Path("data/pdfs/Guías de Practica Clínica")
CHROMA_DIR = Path("data/chroma_db")
COLLECTION = "guias_medicas"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MIN_CHARS  = 100   # páginas con menos chars se descartan (portadas/índices)
CHUNK_SIZE = 800   # caracteres por chunk
OVERLAP    = 100   # solapamiento entre chunks

# ── Inicializar ChromaDB ──────────────────────────────────────────────────────
client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = client.get_or_create_collection(
    name=COLLECTION,
    metadata={"hnsw:space": "cosine"}
)

# ── Inicializar modelo de embeddings ─────────────────────────────────────────
print(f"Cargando modelo: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

# ── Helpers ───────────────────────────────────────────────────────────────────
def extraer_chunks(texto: str, archivo: str, pagina: int) -> list[dict]:
    """Divide el texto en chunks con overlap."""
    chunks = []
    start = 0
    idx = 0
    while start < len(texto):
        end = start + CHUNK_SIZE
        chunk = texto[start:end].strip()
        if len(chunk) >= MIN_CHARS:
            chunks.append({
                "text": chunk,
                "id": f"{archivo}__p{pagina}__c{idx}",
                "metadata": {
                    "archivo": archivo,
                    "pagina": pagina,
                    "chunk": idx,
                }
            })
        start += CHUNK_SIZE - OVERLAP
        idx += 1
    return chunks


def indexar_pdf(pdf_path: Path) -> int:
    """Procesa un PDF y agrega sus chunks a ChromaDB. Retorna número de chunks."""
    nombre = pdf_path.stem  # e.g. "IMSS-007-08-ER"
    doc = fitz.open(str(pdf_path))
    all_chunks = []

    for num_pagina, pagina in enumerate(doc, start=1):
        texto = pagina.get_text().strip()
        if len(texto) < MIN_CHARS:
            continue  # descarta portadas e índices
        chunks = extraer_chunks(texto, nombre, num_pagina)
        all_chunks.extend(chunks)

    doc.close()

    if not all_chunks:
        return 0

    # Generar embeddings en batch
    textos    = [c["text"] for c in all_chunks]
    ids       = [c["id"]   for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]
    embeddings = model.encode(textos, show_progress_bar=False).tolist()

    collection.upsert(
        ids=ids,
        documents=textos,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(all_chunks)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pdfs = sorted(PDF_DIR.glob("*.pdf")) + sorted(PDF_DIR.glob("*.PDF"))
    total = len(pdfs)
    print(f"PDFs encontrados: {total}")

    total_chunks = 0
    errores = []

    for i, pdf_path in enumerate(pdfs, start=1):
        try:
            n = indexar_pdf(pdf_path)
            total_chunks += n
            print(f"[{i}/{total}] {pdf_path.name} → {n} chunks")
        except Exception as e:
            errores.append((pdf_path.name, str(e)))
            print(f"[{i}/{total}] ERROR {pdf_path.name}: {e}")

    print(f"\n✅ Indexación completa: {total_chunks} chunks de {total - len(errores)} PDFs")
    if errores:
        print(f"⚠️  Errores ({len(errores)}):")
        for nombre, err in errores:
            print(f"   {nombre}: {err}")
