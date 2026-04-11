import json
import random

random.seed(42)

# Number of sentences to extract per dataset
dataset_info = {
    "airport":              4,
    "artist":               5,
    "astronaut":            4,
    "athlete":              6,
    "building":             6,
    "celestialbody":        4,
    "city":                11,
    "comicscharacter":      2,
    "company":              3,
    "film":                 7,
    "food":                 8,
    "meanoftransportation": 5,
    "monument":             1,
    "musicalwork":         11,
    "politician":           7,
    "scientist":            8,
    "sportsteam":           6,
    "university":           4,
    "writtenwork":          7,
}

with open("extracted_text_Text2KG.txt", "w", encoding="utf-8") as out:

    for dataset, n in dataset_info.items():

        # Read all sentences from the ground truth file
        sentences = []
        with open(f"./data/Text2KGBench_LettrIA/{dataset}/ground_truth.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                sentences.append(entry["sent"])

        # Randomly pick n sentences and write them to the output file
        sampled = random.sample(sentences, n)
        for sentence in sampled:
            out.write(sentence + "\n")

