# GraphRAG Hallucination Analysis

This repository contains the code and experiments for a comparative study of
hallucinations in **classical RAG** (dense vector similarity over text chunks)
versus **GraphRAG** (structural traversal over an ontology-constrained Knowledge
Graph). The goal is not to eliminate hallucinations but to characterise *when*,
*why*, and *how* they appear in each paradigm, using two complementary reference
benchmarks.

**Authors:** Flavien Mathieu, Raphael Partouche
**Supervisors:** Raphael Troncy, Pasquale Lisena, Carmelle Meli Songuon, Thibault Ehrhart



## Project overview

The project is structured in three sequential tasks:

1. **Task 1 — Question generation.** Enrich the two reference benchmarks with a
   balanced set of single-hop and multi-step questions, half generated from the
   source text (fine-tuned T5 model) and half from the Knowledge Graph
   (strictly-prompted LLM).
2. **Task 2 — Pipelines.** Implement a classical RAG pipeline and a GraphRAG
   pipeline sharing the same generator LLM on the same source data.
3. **Task 3 — Evaluation.** Run both pipelines on the Task 1 question set,
   evaluate them quantitatively with the MIRAGE framework, and categorise the
   hallucinations qualitatively.



## Repository structure

```
.
├── data/                           # Datasets (gitignored, see below)
│   ├── Text2KGBench_LettrIA/
│   └── OSKGC/
├── stats/                          # Category-level distribution of each dataset
│   ├── stats_Text2KG.py
│   ├── stats_OSKGC.py
│   └── results.md
├── task1_questions_generation/     # Task 1: question generation
│   ├── data_loader.py              #   common dataset loaders + proportional sampler
│   ├── generate_t5.py              #   text-based QG with valhalla/t5-small-qa-qg-hl
│   ├── generate_llm.py             #   graph-based QG with GPT-4o
│   ├── extract.py                  #   helper to flatten T5 qa_pairs into a simple JSON
│   └── output/                     #   generated QA pairs (gitignored)
├── task2_setup_rag/                # Task 2: RAG / GraphRAG pipelines (in progress)
│   ├── rag/                        #   text-chunking + dense-retrieval pipeline
│   ├── graph_rag/                  #   knowledge-graph retrieval pipeline
│   ├── llm/                        #   shared LLM interface (Qwen)
│   └── run_pipelines.py            #   runs both pipelines on the task 1 questions
└── .env                            # Local paths and API keys (not committed)

```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Mflavien01/GraphRAG-Hallucination-Analysis.git
cd GraphRAG-Hallucination-Analysis
```

### 2. Install the Python dependencies

```bash
pip install -r requirements.txt
```

The main dependencies are :

- `transformers` 
- `torch`
- `openai`
- `pandas`
- `numpy`
- `python-dotenv`

### 3. Download the datasets

The `data/` folder is gitignored. Both datasets must be downloaded manually.

#### Text2KGBench-LettrIA

Download the dataset from the official Google Drive link provided by the authors:
https://drive.google.com/file/d/1GCEEHzHkTkaMbjgrKfYgIcZVE4-VixPe/view?usp=drive_link

Unzip the archive and place it under `data/` so that the final layout is:
```
data/Text2KGBench_LettrIA/
├── airport/
│   └── ground_truth.jsonl
├── artist/
│   └── ground_truth.jsonl
├── ...
└── writtenwork/
└── ground_truth.jsonl
```
Each `ground_truth.jsonl` file contains one JSON object per line, pairing a
source sentence with its list of typed triples.

#### OSKGC

Clone the official OSKGC repository into `data/OSKGC`:

```bash
git clone https://github.com/HeraclesWang/OSKGC.git data/OSKGC
```

The XML entries used by the project live at
`data/OSKGC/benchmark/data/dev/*.xml`.

### 4. Configure the environment variables

Create a `.env` file at the root of the repository:

```bash
LETTRIA_DIR=data/Text2KGBench_LettrIA
OSKGC_DIR=data/OSKGC/benchmark/data/dev

# OpenAI API
OPENAI_API_KEY=your_key_here
```

## Usage

### Dataset statistics

Print the category-level distribution of each dataset (sentences, triples,
percentage share):

```bash
python stats/stats_Text2KG.py
python stats/stats_OSKGC.py
```

The output drives the proportional stratified sampler used in Task 1.

### Task 1 — Question generation

Generate text-based questions with the fine-tuned T5 model:

```bash
python task1_questions_generation/generate_t5.py
```

Generate graph-based questions with GPT-4o (requires a valid API key in `.env`):

```bash
python task1_questions_generation/generate_llm.py
```

Both scripts produce one `.jsonl` file per dataset in
`task1_questions_generation/output/` (`questions_t5_*.jsonl` and
`questions_llm_*.jsonl`), containing 100 entries each. Each line carries the
originating sentence, the originating triples, and the generated question(s).


## References

- **Text2KGBench-LettrIA** — Plu et al. *A Refined Benchmark for Text2Graph
  Systems*. ISWC 2025. [paper](https://ceur-ws.org/Vol-4041/paper3.pdf)
- **OSKGC** — Wang and Iwaihara. *A Benchmark for Ontology Schema-based Knowledge
  Graph Construction from Text*. ISWC 2025.
  [paper](https://ceur-ws.org/Vol-4041/paper1.pdf) —
  [repository](https://github.com/HeraclesWang/OSKGC)
- **T5 QG model** — `valhalla/t5-small-qa-qg-hl` on
  [Hugging Face](https://huggingface.co/valhalla/t5-small-qa-qg-hl), distributed
  through [patil-suraj/question_generation](https://github.com/patil-suraj/question_generation)
- **MIRAGE** — Vendeville et al. *A Framework for the Quantitative Evaluation of
  Hallucinations in RAG*.
  [repository](https://github.com/bVendeville/MIRAGE)
