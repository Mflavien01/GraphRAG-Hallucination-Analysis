import json
from transformers import pipeline

model = pipeline("text2text-generation", model="valhalla/t5-small-qa-qg-hl")

def extract_answers(text):
    answers = []
    for sent in text.split("."):
        sent = sent.strip()
        if not sent:
            continue
        output = model(f"extract_answers: <hl> {sent} <hl>", max_length=128)
        for a in output[0]["generated_text"].split("<sep>"):
            a = a.strip()
            if a and a in text:
                answers.append(a)
    return list(set(answers))

def generate_question(text, answer):
    highlighted = text.replace(answer, f"<hl> {answer} <hl>", 1)
    output = model(f"generate question: {highlighted}", max_length=64)
    return output[0]["generated_text"]


input_file = "train.jsonl"
output_file = "questions_output.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

results = []
for item in data:
    sentence = item["sent"]
    answers = extract_answers(sentence)
    qa_pairs = []
    for ans in answers:
        q = generate_question(sentence, ans)
        qa_pairs.append({"question": q, "answer": ans})

    results.append({
        "id": item["id"],
        "sentence": sentence,
        "qa_pairs": qa_pairs
    })

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"{len(results)} sentences -> {output_file}")