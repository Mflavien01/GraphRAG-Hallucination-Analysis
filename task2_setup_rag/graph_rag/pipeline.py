import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(__file__)) # so siblings can be imported when loaded as a package

from graph_builder import build_graph
from kg_loader import load_all_triples
from statement_retriever import build_statement_index, retrieve_statements

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(Path(__file__).parent.parent / "llm")) # add llm/ to path for llm_interface
from llm_interface import *

_triples = load_all_triples()
_index, _statements, _model = build_statement_index(_triples)



def run_graphrag(question, llm: BaseLLM, k: int = 12):
    context = retrieve_statements(question, _index, _statements, _model, k=k)
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
