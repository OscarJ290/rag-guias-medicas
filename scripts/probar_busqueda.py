"""
Script de prueba: verifica que ChromaDB esté bien y hace una consulta de ejemplo.

Uso:
    python scripts/probar_busqueda.py
"""

import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path

CHROMA_DIR = Path("data/chroma_db")
COLLECTION = "guias_medicas"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
N_RESULTS  = 5

client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = client.get_collection(COLLECTION)
model = SentenceTransformer(MODEL_NAME)

print(f"Total de chunks indexados: {collection.count()}\n")

pregunta = "¿Cuál es el tratamiento para la diabetes tipo 2?"
print(f"Pregunta de prueba: {pregunta}\n")

embedding = model.encode([pregunta]).tolist()
resultados = collection.query(
    query_embeddings=embedding,
    n_results=N_RESULTS,
    include=["documents", "metadatas", "distances"]
)

for i, (doc, meta, dist) in enumerate(zip(
    resultados["documents"][0],
    resultados["metadatas"][0],
    resultados["distances"][0]
), start=1):
    print(f"--- Resultado {i} ---")
    print(f"Archivo : {meta['archivo']}  |  Página: {meta['pagina']}  |  Distancia: {dist:.4f}")
    print(f"Texto   : {doc[:200]}...")
    print()
