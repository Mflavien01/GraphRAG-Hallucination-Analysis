from graph_builder import build_graph
from kg_loader import load_all_triples
from statement_retriever import build_statement_index, retrieve_statements
from pathlib import Path
import json
import os
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(Path(__file__).parent.parent / "llm")) # add llm/ to path for llm_interface
from llm_interface import *

_triples = load_all_triples()
_index, _statements, _model = build_statement_index(_triples)



def run_graphrag(question, llm: BaseLLM):
    context = retrieve_statements(question, _index, _statements, _model, k=10)
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
