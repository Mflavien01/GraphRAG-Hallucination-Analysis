import json

DATASETS = {
    "lettria": "questions_t5_lettria.jsonl",
    "oskgc":   "questions_t5_oskgc.jsonl",
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

    output_file = f"questions_t5_simple_{name}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"{name} — {len(results)} pairs saved to {output_file}")