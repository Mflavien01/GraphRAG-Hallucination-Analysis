import json
import os
import pandas as pd
import numpy as np

data_path = "data/Text2KGBench_LettrIA/"
stats = []

for d in sorted(os.listdir(data_path)):
    filepath = os.path.join(data_path, d, "ground_truth.jsonl")
    if not os.path.isfile(filepath):
        continue
    sentences = 0
    triplets = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            sentences += 1
            triplets += len(data["triples"])
    stats.append({"dataset": d, "sentences": sentences, "triplets": triplets})

df = pd.DataFrame(stats)
df["percentage"] = np.ceil((df["sentences"] / df["sentences"].sum() * 100).round(1))
df.loc[len(df)] = ["TOTAL", df["sentences"].sum(), df["triplets"].sum(), 100.0]
print(df.to_string(index=False))


