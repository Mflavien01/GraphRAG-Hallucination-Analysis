import json
import csv
import re
from pathlib import Path


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_rag_source(context):
    """
    Context RAG is a list of dicts with a 'text' field.
    We extract only the text, without metadata (id, distance...).
    """
    if isinstance(context, list):
        texts = []
        for chunk in context:
            if isinstance(chunk, dict):
                texts.append(chunk.get("text", ""))
            else:
                texts.append(str(chunk))
        return " ".join(t for t in texts if t)
    if isinstance(context, dict):
        return context.get("text", "")
    return str(context)


def extract_graphrag_source(context) -> str:
    """
    The GraphRAG context is a list of triples (subject, relation, object)
    (e.g., 'Arem arem -[origin]-> Indonesia').
    We deduplicate to remove repeated triples from FAISS.
    """
    if isinstance(context, list):
        seen = []
        for triple in context:
            t = str(triple).strip()
            if t and t not in seen:
                seen.append(t)
        return " ".join(seen)
    return str(context)


def convert_to_mirage(results_path, output_path, pipeline_name):
    '''Convert result JSONL from task2_setup_rag into CSV for MIRAGE (FactCC)'''
    entries = load_jsonl(results_path)
    rows = []

    for entry in entries:
        ctx = entry.get("context", "")

        if pipeline_name == "rag":
            source = extract_rag_source(ctx)
        else:
            source = extract_graphrag_source(ctx)

        rows.append({
            "source":       source,
            "gen":          entry["answer"],
            "id":           entry["id"],
            "question":     entry["question"],
            "ground_truth": entry.get("ground_truth", ""),
            "hop_type":     entry.get("hop_type", "unknown"),
            "pipeline":     pipeline_name,
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"[{pipeline_name}] {len(rows)} entries → {output_path}")


if __name__ == "__main__":
    convert_to_mirage(
        results_path="task2_setup_rag/output/rag_results.jsonl",
        output_path="task3_evaluation/inputs/rag_input.csv",
        pipeline_name="rag"
    )
    convert_to_mirage(
        results_path="task2_setup_rag/output/graphrag_results.jsonl",
        output_path="task3_evaluation/inputs/graphrag_input.csv",
        pipeline_name="graphrag"
    )