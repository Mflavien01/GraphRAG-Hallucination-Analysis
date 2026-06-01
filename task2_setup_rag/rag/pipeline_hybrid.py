"""
RAG Pipeline — version hybride BM25 + FAISS + cross-encoder reranking

Objectif : réduire les cas de parametric memory leakage
où l'embedding cosine rate des chunks avec des noms propres précis.

Différence vs pipeline.py :
- build_bm25_index() en plus au démarrage
- search_hybrid() à la place de search() (FAISS∪BM25 pool → cross-encoder rerank)
- Tout le reste est identique
"""

import sys
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.dirname(__file__))

from chunker import chunk_texts
from embedder import embed_chunks
from loader import load_all_texts
from vector_store import build_index, build_bm25_index, search_hybrid

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(Path(__file__).parent.parent / "llm"))
from llm_interface import *

# ── Construction des deux index au démarrage ──────────────────────────────
_texts = load_all_texts()
_chunks = chunk_texts(_texts)
_model = SentenceTransformer("all-MiniLM-L6-v2")
_embedded_chunks = embed_chunks(_chunks, model_name="all-MiniLM-L6-v2")

_faiss_index = build_index(_embedded_chunks)      # index sémantique (existant)
_bm25_index  = build_bm25_index(_embedded_chunks) # index lexical (nouveau)


def run_rag_hybrid(question, llm: BaseLLM, k: int = 5):
    # Retrieval hybride BM25 + FAISS
    results = search_hybrid(
        question, _faiss_index, _bm25_index, _embedded_chunks, _model, k=k
    )
    context_texts = "\n\n".join([result["text"] for result in results])

    prompt = f"""Given the following retrieved context:
{context_texts}
Answer the following question based only on the context above:
{question}"""

    llm_answer = llm.generate(prompt)
    return {
        "question":     question,
        "context":      results,
        "context_texts": context_texts,
        "llm_answer":   llm_answer
    }