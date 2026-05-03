import json
from pathlib import Path

# Flatten T5 nested qa_pairs into a simple [{sentence, question, answer}] format.
# Outputs land in output/_archive/ since this is a one-off helper, not a primary artefact.
OUTPUT_DIR = Path(__file__).parent / "output"
ARCHIVE_DIR = OUTPUT_DIR / "_archive"
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "lettria": OUTPUT_DIR / "questions_t5_lettria.jsonl",
    "oskgc":   OUTPUT_DIR / "questions_t5_oskgc.jsonl",
}

for name, filepath in DATASETS.items():
    results = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            sentence = entry["sentence"]
            for pair in entry["qa_pairs"]:
                results.append({
                    "sentence": sentence,
                    "question": pair["question"],
                    "answer":   pair["answer"],
                })

    output_file = ARCHIVE_DIR / f"questions_t5_simple_{name}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"{name} — {len(results)} pairs saved to {output_file}")
