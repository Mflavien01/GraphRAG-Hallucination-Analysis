"""
GraphRAG Pipeline — version hybride FAISS (cosine) + BM25 + cross-encoder reranking

Reprend la méthode de rag/pipeline_hybrid.py appliquée aux statements du KG
plutôt qu'aux chunks texte.

Différence vs pipeline.py :
- build_statement_index_hybrid() en plus au démarrage (construit deux index)
- retrieve_statements_hybrid() à la place de retrieve_statements() (FAISS∪BM25 pool → cross-encoder rerank)
- Tout le reste est identique (même prompt, même format de sortie)
"""

import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(__file__))

from kg_loader import load_all_triples
from statement_retriever_hybrid import build_statement_index_hybrid, retrieve_statements_hybrid

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(Path(__file__).parent.parent / "llm"))
from llm_interface import *

# ── Construction des deux index au démarrage ──────────────────────────────
_triples = load_all_triples()
_faiss_index, _bm25_index, _statements, _model = build_statement_index_hybrid(_triples)


def run_graphrag_hybrid(question, llm: BaseLLM, k: int = 10):
    context = retrieve_statements_hybrid(
        question, _faiss_index, _bm25_index, _statements, _model, k=k
    )
    context_text = "\n".join(context)

    prompt = f"""Given the following knowledge graph context:
{context_text}

Answer the following question based only on the context above:
{question}"""

    return {
        "question":     question,
        "context":      context,
        "context_text": context_text,
        "llm_answer":   llm.generate(prompt)
    }
