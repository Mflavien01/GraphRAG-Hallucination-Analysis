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


def load_questions(file_path):
    questions = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)

            single_hop = entry["graph_single_hop"]
            if single_hop["question"] and single_hop["answer"]: # skip if no question generated
                questions.append({
                    "id":        entry["id"],
                    "source_id": entry["source_id"],
                    "dataset":   entry["dataset"],
                    "category":  entry["category"],
                    "hop_type":  "single_hop",
                    "question":  single_hop["question"],
                    "answer":    single_hop["answer"],
                })

            multi_hop = entry["graph_multi_hop"]
            if multi_hop and multi_hop["question"] and multi_hop["answer"]: # skip if multi-hop was not possible
                questions.append({
                    "id":        entry["id"],
                    "source_id": entry["source_id"],
                    "dataset":   entry["dataset"],
                    "category":  entry["category"],
                    "hop_type":  "multi_hop",
                    "question":  multi_hop["question"],
                    "answer":    multi_hop["answer"],
                    "chain":     multi_hop["chain"],
                })

    return questions


if __name__ == "__main__":
    llm = PlaceholderLLM()

    questions_dir = PROJECT_ROOT / "task1_questions_generation" / "output_questions"
    questions = load_questions(questions_dir / "questions_graph_lettria.jsonl")
    questions += load_questions(questions_dir / "questions_graph_oskgc.jsonl")
    print(f"Loaded {len(questions)} questions")

    results = []
    for i, q in enumerate(questions):
        print(f"  [{i+1}/{len(questions)}] {q['id']} ({q['hop_type']})")

        output = run_graphrag(q["question"], llm)

        results.append({
            "id":           q["id"],
            "source_id":    q["source_id"],
            "dataset":      q["dataset"],
            "category":     q["category"],
            "hop_type":     q["hop_type"],
            "question":     q["question"],
            "answer":       q["answer"],
            "anchor_nodes": output["anchor_nodes"],
            "context":      output["context"],
            "llm_answer":   output["llm_answer"],
        })

    output_dir = Path(__file__).parent / "output"
    os.makedirs(output_dir, exist_ok=True)

    with open(output_dir / "graphrag_results.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(results)} results")
