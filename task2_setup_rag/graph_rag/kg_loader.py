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

def load_all_triples():
    """Load all triples from both datasets and return as a list of dicts. [{"subject": ..., "predicate": ..., "object": ..., "source": ...}, ...] and remove underscores from OSKGC triples"""
    all_triples = []
    lettria_data= load_lettria(lettria_dir)
    oskgc_data = load_oskgc(oskgc_dir)
    datasets = [
        (lettria_data, "LettrIA"),
        (oskgc_data, "OSKGC")
    ]

    for data, source in datasets:
        for entry in data:
            for t in entry["triples"]:
                subject = t["sub"]
                obj = t["obj"]
                if source == "OSKGC":
                    subject = subject.replace("_", " ")
                    obj = obj.replace("_", " ")
                
                all_triples.append({
                    "subject": subject,
                    "predicate": t["rel"],
                    "object": obj,
                    "source": source
                })
    print(f"Total triples loaded: {len(all_triples)}")
    return all_triples
    
