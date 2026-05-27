"""
Empirical diagnostic of FactCC scoring direction.

Loads manueldeprada/FactCC and runs it on two known pairs:
  (a) faithful   — source and claim agree word-for-word
  (b) contradiction — claim contradicts the source

Prints raw logits, full softmax (per id and per label name) and the predicted
label, so we can verify which softmax index corresponds to CORRECT vs INCORRECT,
and which value mirage.py currently exports as `score`.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = "manueldeprada/FactCC"

PAIRS = [
    ("FAITHFUL",
     "The Aarhus University is located in Aarhus, Denmark.",
     "Aarhus University is located in Aarhus, Denmark."),
    ("CONTRADICTION",
     "AMC Matador is assembled in Thames (New Zealand).",
     "The AMC Matador is assembled in Mexico."),
]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.eval()

    print(f"Model: {MODEL}")
    print(f"id2label: {model.config.id2label}")
    print(f"label2id: {model.config.label2id}")
    print("=" * 70)

    for tag, source, claim in PAIRS:
        enc = tok(source, claim, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**enc).logits[0]
        probs = F.softmax(logits, dim=-1).tolist()
        pred_id = int(torch.argmax(logits).item())
        pred_label = model.config.id2label[pred_id]
        by_name = {model.config.id2label[i]: probs[i] for i in range(len(probs))}

        print(f"\n[{tag}]")
        print(f"  source: {source!r}")
        print(f"  claim : {claim!r}")
        print(f"  logits        = {logits.tolist()}")
        print(f"  softmax (idx) = {probs}")
        print(f"  softmax (name)= {by_name}")
        print(f"  argmax        = {pred_id}  -> {pred_label}")
        print(f"  P(CORRECT)    = {by_name['CORRECT']:.4f}")
        print(f"  P(INCORRECT)  = {by_name['INCORRECT']:.4f}")
        print(f"  >> mirage.py would export score = P(INCORRECT) = {by_name['INCORRECT']:.4f}")


if __name__ == "__main__":
    main()
