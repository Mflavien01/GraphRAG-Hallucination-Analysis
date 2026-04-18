from pathlib import Path
from dotenv import load_dotenv
import json
import os
import time
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / 'task1_questions_generation'))
from data_loader import load_lettria, load_oskgc

PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env") # load env variables (API key, dataset paths)

lettria_dir = PROJECT_ROOT / os.getenv("LETTRIA_DIR") # path to LettrIA dataset
oskgc_dir   = PROJECT_ROOT / os.getenv("OSKGC_DIR")   # path to OSKGC dataset

def load_all_texts():
    """Load all texts from both datasets and return as a list of dicts. [{"id": ..., "category": ..., "dataset": ..., "text": ...}, ...]"""
    all_texts = []
    lettria_data= load_lettria(lettria_dir)
    oskgc_data = load_oskgc(oskgc_dir)
    datasets = [
        (lettria_data, "LettrIA"),
        (oskgc_data, "OSKGC")
    ]

    for data, source in datasets:
        for entry in data:
            all_texts.append({
                "id": entry["id"],
                "category": entry["category"],
                "dataset": source,
                "text": entry["sent"]
            })
    print(f"Total texts loaded: {len(all_texts)}")
    return all_texts
