"""Question generation using the T5 model (valhalla/t5-small-qa-qg-hl).

Extracts answer spans from each sentence, then generates a question per answer.
Runs locally without any API key.
"""
import json
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer

from data_loader import load_lettria, load_oskgc, sample_proportional

tokenizer = T5Tokenizer.from_pretrained("valhalla/t5-small-qa-qg-hl") # load the T5 tokenizer for the specified model to process input and output text
model     = T5ForConditionalGeneration.from_pretrained("valhalla/t5-small-qa-qg-hl") # load the T5 model for answer extraction + question generation based on t5-small-qa-qg-hl model


def run_model(input_text, max_length=128):
    """Run the T5 model with the given input text and return the generated output"""
    ids    = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True) # tokenize the input text to be processed by the model
    output = model.generate(ids, max_length=max_length, num_beams=4, early_stopping=True) # apply the model
    return tokenizer.decode(output[0], skip_special_tokens=True) # detokenize the output to get the generated text


def extract_answers(text):
    answers = []
    for sent in text.split("."): # split sentence in every dot to obtain individual sentences
        sent = sent.strip()
        if not sent:
            continue
        raw = run_model(f"extract_answers: <hl> {sent} <hl>") # model return answers spans splitted by <sep>
        for a in raw.split("<sep>"): # model splits string with <sep> to obtain answers individually
            a = a.strip() # remove whitespace of answers
            if a and a in text:
                answers.append(a)
    return list(set(answers))


def generate_question(text, answer):
    highlighted = text.replace(answer, f"<hl> {answer} <hl>", 1) # highlight the answer in the sentence to be processed by the model for question generation
    return run_model(f"generate question: {highlighted}", max_length=64) # model return the generated question based on the highlighted answer in the sentence


def generate_questions(entries, dataset_name):
    results = []
    total = len(entries)

    for i, entry in enumerate(entries):
        print(f"  [{i+1}/{total}] {entry['id']} ({entry['category']})")

        sentence = entry["sent"]
        answers  = extract_answers(sentence)
        qa_pairs = [
            {"question": generate_question(sentence, ans), "answer": ans}
            for ans in answers
        ]

        results.append({
            "id":       f"{dataset_name}_{i+1:03d}",
            "source_id": entry["id"],
            "dataset":   dataset_name,
            "category":  entry["category"],
            "sentence":  sentence,
            "qa_pairs":  qa_pairs,
        })

    return results


def save_results(results, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  → Saved: {output_path} ({len(results)} entries)")


if __name__ == "__main__":
    from dotenv import load_dotenv
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).parent.parent
    load_dotenv(PROJECT_ROOT / ".env")

    OUTPUT_DIR  = "output_questions"
    lettria_dir = PROJECT_ROOT / os.getenv("LETTRIA_DIR")
    oskgc_dir   = PROJECT_ROOT / os.getenv("OSKGC_DIR")

    print("\n=== DATASET: LettrIA ===")
    lettria_sample = sample_proportional(load_lettria(lettria_dir), 50)
    print("Generating questions...")
    lettria_results = generate_questions(lettria_sample, "lettria")
    save_results(lettria_results, f"{OUTPUT_DIR}/questions_t5_lettria.jsonl")

    print("\n=== DATASET: OSKGC ===")
    oskgc_sample = sample_proportional(load_oskgc(oskgc_dir), 50)
    print("Generating questions...")
    oskgc_results = generate_questions(oskgc_sample, "oskgc")
    save_results(oskgc_results, f"{OUTPUT_DIR}/questions_t5_oskgc.jsonl")

    print(f"\n=== DONE ===")
    print(f"Total: {len(lettria_results) + len(oskgc_results)} entries processed")
