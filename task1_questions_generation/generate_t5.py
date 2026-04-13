import json
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer
from data_loader import load_lettria, load_oskgc, sample_proportional
from dotenv import load_dotenv
from pathlib import Path

# Core T5 logic

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
            if a and a in text: # keep only non-empty answers that actually appear in the original text
                answers.append(a)
    return list(set(answers)) # remove duplicates


def generate_question(text, answer):
    highlighted = text.replace(answer, f"<hl> {answer} <hl>", 1) # highlight the answer in the sentence to be processed by the model for question generation
    return run_model(f"generate question: {highlighted}", max_length=64) # model return the generated question based on the highlighted answer in the sentence



# Dataset integration layer


def generate_questions(entries, dataset_name):
    """Run the T5 pipeline on dataset entries and return text QA results."""
    results = []
    total = len(entries)

    for i, entry in enumerate(entries):
        print(f"  [{i+1}/{total}] {entry['id']} ({entry['category']})")

        sentence = entry["sent"] # raw sentence from the dataset entry
        answers  = extract_answers(sentence) # extract all answer spans from the sentence
        qa_pairs = [
            {"question": generate_question(sentence, ans), "answer": ans}
            for ans in answers # generate one question per extracted answer
        ]

        results.append({
            "id":        f"{dataset_name}_{i+1:03d}", # unique id for the generated entry
            "source_id": entry["id"],                 # original id from the dataset
            "dataset":   dataset_name,
            "category":  entry["category"],
            "sentence":  sentence,
            "qa_pairs":  qa_pairs,
        })

    return results


def save_results(results, output_path):
    """Save results as a .jsonl file (one JSON object per line)"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True) # create output directory if it doesn't exist
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n") # write each result as a single JSON line
    print(f"  → Saved: {output_path} ({len(results)} entries)")




PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env") # load env variables (dataset paths, etc.)

OUTPUT_DIR  = "output_questions"
SAMPLES_PER_DATASET = 100
lettria_dir = PROJECT_ROOT / os.getenv("LETTRIA_DIR") # path to LettrIA dataset
oskgc_dir   = PROJECT_ROOT / os.getenv("OSKGC_DIR")   # path to OSKGC dataset

print("DATASET: LettrIA")
lettria_sample = sample_proportional(load_lettria(lettria_dir), SAMPLES_PER_DATASET)
print("Generating questions...")
lettria_results = generate_questions(lettria_sample, "lettria")
save_results(lettria_results, f"{OUTPUT_DIR}/questions_t5_lettria.jsonl")

print("DATASET: OSKGC ")
oskgc_sample = sample_proportional(load_oskgc(oskgc_dir), SAMPLES_PER_DATASET)
print("Generating questions...")
oskgc_results = generate_questions(oskgc_sample, "oskgc")
save_results(oskgc_results, f"{OUTPUT_DIR}/questions_t5_oskgc.jsonl")

print("DONE")
print(f"Total: {len(lettria_results) + len(oskgc_results)} entries processed")
