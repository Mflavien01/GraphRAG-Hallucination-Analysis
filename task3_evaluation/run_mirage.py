"""
  python task3_evaluation/run_mirage.py --pipeline rag
  python task3_evaluation/run_mirage.py --pipeline graphrag
  python task3_evaluation/run_mirage.py --pipeline graph-chunk
  python task3_evaluation/run_mirage.py --pipeline both
"""

import sys
import shutil
import tempfile
import argparse
from pathlib import Path

from datasets import load_from_disk

sys.path.insert(0, "task3_evaluation/mirage_lib")
from mirage import factcc


def add_faithfulness_and_save(output_dir: str):
    # MIRAGE writes `score = P(INCORRECT) = P(hallucination)`. Expose the inverse
    # `faithfulness = P(CORRECT)` so downstream code uses the natural "higher = better"
    # convention. Save via tmpdir + swap because save_to_disk cannot overwrite the
    # arrow files it's currently reading from.
    ds = load_from_disk(output_dir)
    ds = ds.add_column("faithfulness", [1.0 - s for s in ds["score"]])
    tmp = tempfile.mkdtemp(prefix="factcc_swap_", dir=str(Path(output_dir).parent))
    ds.save_to_disk(tmp)
    del ds  # release file handles before swap
    shutil.rmtree(output_dir)
    shutil.move(tmp, output_dir)
    return load_from_disk(output_dir)


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
    del result  # release handles on the freshly-saved arrow files
    result = add_faithfulness_and_save(output_dir)
    print(f"Done — {result.num_rows} entries scored")
    print(f"Columns: {result.column_names}")
    print(f"Mean faithfulness: {sum(result['faithfulness']) / len(result['faithfulness']):.4f}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline",
                        choices=["rag", "graphrag", "hybrid", "graphrag-hybrid", "graph-chunk", "both"],
                        default="both")
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
    if args.pipeline in ("graphrag-hybrid", "both"):
        run(
            input_path="task3_evaluation/inputs/graphrag_hybrid_input.csv",
            output_dir="task3_evaluation/outputs/factcc_graphrag_hybrid"
        )
    if args.pipeline in ("graph-chunk", "both"):
        run(
            input_path="task3_evaluation/inputs/graph_chunk_input.csv",
            output_dir="task3_evaluation/outputs/factcc_graph_chunk"
        )