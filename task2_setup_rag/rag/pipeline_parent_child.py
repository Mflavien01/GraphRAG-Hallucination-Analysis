"""
RAG Pipeline — Parent/Child chunking

Retrieval: child chunks (small, ~150 tokens) are indexed in FAISS for precise
           semantic matching.
Context:   when a child chunk is retrieved, its parent chunk (~1500 tokens) is
           returned to the LLM for richer, coherent context.
Dedup:     multiple children from the same parent collapse into a single parent
           context block (k unique parents are returned).
"""

import sys
import os
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.dirname(__file__))

from chunker_parent_child import chunk_texts_parent_child
from embedder import embed_chunks
from loader import load_all_texts
from vector_store import build_index

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(Path(__file__).parent.parent / "llm"))
from llm_interface import *

# ── Index build at module load time ──────────────────────────────────────────
_texts = load_all_texts()
_parent_chunks, _child_chunks = chunk_texts_parent_child(_texts)
_model = SentenceTransformer("all-MiniLM-L6-v2")
_embedded_children = embed_chunks(_child_chunks, model_name="all-MiniLM-L6-v2")
_child_index = build_index(_embedded_children)

# Fast parent lookup: parent_id → parent chunk dict
_parent_lookup = {p["id"]: p for p in _parent_chunks}


def _search_parent_child(query, k=4):
    """
    Retrieve child chunks via FAISS, then map each to its parent.
    Returns k unique parent chunks (deduplication: multiple children
    matching the same parent count as one result).
    We over-fetch children (k*4) to ensure k distinct parents.
    """
    query_vector = _model.encode(query).reshape(1, -1)
    distances, indices = _child_index.search(query_vector, k * 4)

    seen_parent_ids = set()
    results = []

    for idx, dist in zip(indices[0], distances[0]):
        if idx < 0:
            continue
        child = _embedded_children[idx]
        pid = child["parent_id"]
        if pid in seen_parent_ids:
            continue
        seen_parent_ids.add(pid)
        parent = _parent_lookup[pid]
        results.append(
            {
                "id": pid,
                "category": parent["category"],
                "dataset": parent["dataset"],
                "text": parent["text"],
                "distance": float(dist),
            }
        )
        if len(results) >= k:
            break

    return results


def run_rag_parent_child(question, llm: BaseLLM, k: int = 4):
    results = _search_parent_child(question, k=k)
    context_texts = "\n\n".join([r["text"] for r in results])
    prompt = f"""Given the following retrieved context:
{context_texts}
Answer the following question based only on the context above:
{question}"""

    llm_answer = llm.generate(prompt)
    return {
        "question": question,
        "context": results,
        "context_texts": context_texts,
        "llm_answer": llm_answer,
    }
