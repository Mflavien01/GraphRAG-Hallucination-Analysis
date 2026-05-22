"""
  python task3_evaluation/run_mirage.py --pipeline rag
  python task3_evaluation/run_mirage.py --pipeline graphrag
  python task3_evaluation/run_mirage.py --pipeline both
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, "task3_evaluation/mirage_lib")
from mirage import factcc


def run(input_path: str, output_dir: str):
    print(f"\n=== FactCC on {input_path} ===")
    metric = factcc()
    result = metric.evaluate_dataset(
        dataset=input_path,
        source_col="source",
        gen_col="gen",
        truncation=True,
        padding=True,
        save_result_dataset_folder_path=output_dir
    )
    print(f"Done — {result.num_rows} entries scored")
    print(f"Columns: {result.column_names}")
    print(f"Mean score: {sum(result['score']) / len(result['score']):.4f}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=["rag", "graphrag", "hybrid", "both"], default="both")
    args = parser.parse_args()

    if args.pipeline in ("rag", "both"):
        run(
            input_path="task3_evaluation/inputs/rag_input.csv",
            output_dir="task3_evaluation/outputs/factcc_rag"
        )
    if args.pipeline in ("graphrag", "both"):
        run(
            input_path="task3_evaluation/inputs/graphrag_input.csv",
            output_dir="task3_evaluation/outputs/factcc_graphrag"
        )
    if args.pipeline in ("hybrid", "both"):
        run(
            input_path="task3_evaluation/inputs/rag_hybrid_input.csv",
            output_dir="task3_evaluation/outputs/factcc_rag_hybrid"
        )