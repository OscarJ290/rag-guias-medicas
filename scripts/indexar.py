"""
Script de indexación: lee los PDFs y los almacena en ChromaDB.
Reanudable: salta PDFs ya indexados.
v2: chunks más grandes, mejor extracción de tablas y algoritmos.

Uso:
    python scripts/indexar.py
"""

import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
import re

# ── Configuración ────────────────────────────────────────────────────────────
PDF_DIR    = Path("data/pdfs/Guías de Practica Clínica")
CHROMA_DIR = Path("data/chroma_db")
COLLECTION = "guias_medicas"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MIN_CHARS  = 80     # ↓ de 100 a 80 para no perder páginas con tablas cortas
CHUNK_SIZE = 1200   # ↑ de 800 a 1200 para no partir tablas/algoritmos
OVERLAP    = 200    # ↑ de 100 a 200 para no perder contexto entre chunks

# ── Inicializar ChromaDB ──────────────────────────────────────────────────────
client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = client.get_or_create_collection(
    name=COLLECTION,
    metadata={"hnsw:space": "cosine"}
)

# ── Cargar archivos ya indexados ──────────────────────────────────────────────
print("Cargando archivos ya indexados...")
ya_indexados = set()
offset = 0
BATCH = 50000
while True:
    r = collection.get(limit=BATCH, offset=offset, include=["metadatas"])
    if not r["metadatas"]:
        break
    for m in r["metadatas"]:
        ya_indexados.add(m["archivo"])
    offset += len(r["metadatas"])
    if len(r["metadatas"]) < BATCH:
        break
print(f"Archivos ya indexados: {len(ya_indexados)}")

# ── Inicializar modelo de embeddings ─────────────────────────────────────────
print(f"Cargando modelo: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

# ── Helpers ───────────────────────────────────────────────────────────────────

def limpiar_texto(texto: str) -> str:
    """Limpia artefactos comunes de extracción PDF sin destruir tablas."""
    # Colapsar líneas en blanco múltiples (>2) pero preservar estructura
    texto = re.sub(r'\n{3,}', '\n\n', texto)
    # Eliminar guiones de separación de página
    texto = re.sub(r'-\n(?=[a-záéíóúüñ])', '', texto)
    return texto.strip()


def extraer_texto_pagina(pagina: fitz.Page) -> str:
    """
    Extrae texto preservando estructura de tablas y algoritmos.
    Usa 'blocks' para mantener el orden espacial correcto.
    """
    bloques = pagina.get_text("blocks", sort=True)  # sort=True: orden de lectura
    lineas = []
    for bloque in bloques:
        # bloque = (x0, y0, x1, y1, texto, block_no, block_type)
        # block_type 0 = texto, 1 = imagen
        if bloque[6] == 0 and bloque[4].strip():
            lineas.append(bloque[4].strip())
    return "\n".join(lineas)


def extraer_chunks(texto: str, archivo: str, pagina: int) -> list[dict]:
    chunks = []
    start = 0
    idx = 0
    while start < len(texto):
        end = start + CHUNK_SIZE
        # Intentar cortar en un salto de párrafo para no partir oraciones
        if end < len(texto):
            corte = texto.rfind('\n\n', start, end)
            if corte > start + CHUNK_SIZE // 2:
                end = corte
        chunk = texto[start:end].strip()
        if len(chunk) >= MIN_CHARS:
            chunks.append({
                "text": chunk,
                "id": f"{archivo}__p{pagina}__c{idx}",
                "metadata": {"archivo": archivo, "pagina": pagina, "chunk": idx}
            })
        start = end - OVERLAP if end < len(texto) else len(texto)
        idx += 1
    return chunks


def indexar_pdf(pdf_path: Path) -> int:
    nombre = pdf_path.stem
    doc = fitz.open(str(pdf_path))
    all_chunks = []
    for num_pagina, pagina in enumerate(doc, start=1):
        texto_raw = extraer_texto_pagina(pagina)
        texto = limpiar_texto(texto_raw)
        if len(texto) < MIN_CHARS:
            continue
        all_chunks.extend(extraer_chunks(texto, nombre, num_pagina))
    doc.close()
    if not all_chunks:
        return 0
    textos    = [c["text"]     for c in all_chunks]
    ids       = [c["id"]       for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]
    embeddings = model.encode(textos, show_progress_bar=False).tolist()
    collection.upsert(ids=ids, documents=textos, embeddings=embeddings, metadatas=metadatas)
    return len(all_chunks)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pdfs = sorted(PDF_DIR.glob("*.pdf")) + sorted(PDF_DIR.glob("*.PDF"))
    total = len(pdfs)
    pendientes = [p for p in pdfs if p.stem not in ya_indexados]
    print(f"PDFs encontrados: {total} | Pendientes: {len(pendientes)} | Ya indexados: {total - len(pendientes)}")

    total_chunks = 0
    errores = []

    for i, pdf_path in enumerate(pendientes, start=1):
        try:
            n = indexar_pdf(pdf_path)
            total_chunks += n
            print(f"[{i}/{len(pendientes)}] {pdf_path.name} → {n} chunks")
        except Exception as e:
            errores.append((pdf_path.name, str(e)))
            print(f"[{i}/{len(pendientes)}] ERROR {pdf_path.name}: {e}")

    print(f"\n✅ Indexación completa: {total_chunks} chunks nuevos")
    if errores:
        print(f"⚠️  Errores ({len(errores)}):")
        for nombre, err in errores:
            print(f"   {nombre}: {err}")
