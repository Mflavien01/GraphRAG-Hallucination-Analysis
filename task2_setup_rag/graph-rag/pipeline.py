from entity_linker import link_entities
from graph_builder import build_graph
from traversal import traverse
from kg_loader import load_all_triples
from pathlib import Path
import json
import os
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(Path(__file__).parent.parent / "llm")) # add llm/ to path for llm_interface
from llm_interface import *

_triples = load_all_triples()
_graph   = build_graph(_triples)


def run_graphrag(question, llm: BaseLLM):
    anchor_nodes = link_entities(question, _graph, threshold=95)
    context      = traverse(_graph, anchor_nodes, k=2)
    context_text = "\n".join(context)

    prompt = f"""Given the following knowledge graph context:
{context_text}

Answer the following question based only on the context above:
{question}"""

    return {
        "question":     question,
        "anchor_nodes": anchor_nodes,
        "context":      context,
        "context_text": context_text,
        "llm_answer":   llm.generate(prompt)
    }
