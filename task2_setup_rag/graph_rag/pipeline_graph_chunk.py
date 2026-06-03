"""
Graph-Chunk Hybrid Pipeline

Combines two complementary retrieval sources:
  - KG statements  : FAISS (cosine) + BM25 pool → cross-encoder rerank on KG triples
  - Text chunks    : FAISS (cosine) + BM25 pool → cross-encoder rerank on raw corpus text

Each source contributes k_graph=5 and k_chunk=5 items. The context is
presented with explicit [KG] / [CHUNK] labels so the LLM can distinguish
structured facts from narrative evidence. Total context window: 10 items.

Design rationale: the two sources are complementary (structured facts vs
prose), so fixed-budget allocation is preferred over cross-source fusion — we
want both types of evidence, not one dominating the other.
"""

import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(__file__))
sys.path.append(str(Path(__file__).parent.parent))

from kg_loader import load_all_triples
from statement_retriever_hybrid import build_statement_index_hybrid, retrieve_statements_hybrid

from rag.chunker import chunk_texts
from rag.embedder import embed_chunks
from rag.loader import load_all_texts
from rag.vector_store import build_index, build_bm25_index, search_hybrid

from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(Path(__file__).parent.parent / "llm"))
from llm_interface import *

# ── Build all four indices at startup ─────────────────────────────────────────
_triples = load_all_triples()
_faiss_stmt, _bm25_stmt, _statements, _stmt_model = build_statement_index_hybrid(_triples)

_chunk_model = SentenceTransformer("all-MiniLM-L6-v2")
_texts = load_all_texts()
_chunks = chunk_texts(_texts)
_embedded_chunks = embed_chunks(_chunks, model_name="all-MiniLM-L6-v2")
_faiss_chunks = build_index(_embedded_chunks)
_bm25_chunks  = build_bm25_index(_embedded_chunks)


def run_graph_chunk(question: str, llm: BaseLLM, k_graph: int = 20, k_chunk: int = 20):
    # KG statements via GraphRAG hybrid retriever
    kg_items = retrieve_statements_hybrid(
        question, _faiss_stmt, _bm25_stmt, _statements, _stmt_model, k=k_graph
    )

    # Text chunks via RAG hybrid retriever
    chunk_items = search_hybrid(
        question, _faiss_chunks, _bm25_chunks, _embedded_chunks, _chunk_model, k=k_chunk
    )
    chunk_texts_list = [r["text"] for r in chunk_items]

    # Build labeled context
    kg_block    = "\n".join(f"[KG]    {s}" for s in kg_items)
    chunk_block = "\n\n".join(f"[CHUNK] {t}" for t in chunk_texts_list)
    context_text = f"{kg_block}\n\n{chunk_block}"

    prompt = f"""Given the following retrieved context (knowledge graph facts and text passages):
{context_text}

Answer the following question based only on the context above:
{question}"""

    return {
        "question":     question,
        "context":      {"kg": kg_items, "chunks": chunk_items},
        "context_text": context_text,
        "llm_answer":   llm.generate(prompt),
    }
