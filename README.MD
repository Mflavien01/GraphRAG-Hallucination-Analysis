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
2. **Task 2 — Pipelines.** Implement a classical RAG pipeline, a hybrid RAG
   pipeline (dense + BM25), and a GraphRAG pipeline, all sharing the same
   generator LLM (Qwen3 served through the Eurecom gateway) on the same source
   data.
3. **Task 3 — Evaluation.** Run all pipelines on the Task 1 question set,
   evaluate them quantitatively with the MIRAGE framework (FactCC faithfulness
   + SBERT cosine sanity check), and categorise the hallucinations
   qualitatively.



## Repository structure

```
.
├── data/                          # Datasets (gitignored, see below)
├── stats/                         # Category-level dataset distributions
├── task1_questions_generation/    # Task 1: T5 + LLM question generation
├── task2_setup_rag/               # Task 2: RAG, hybrid RAG, GraphRAG pipelines
├── task3_evaluation/              # Task 3: MIRAGE / FactCC evaluation
├── requirements.txt
└── .env                           # Local paths and API keys (not committed)
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

The main dependencies are:

- `transformers`, `torch`, `sentencepiece`
- `sentence-transformers`, `faiss-cpu`
- `openai` (Task 1 graph-based QG with GPT-4o)
- `rdflib`, `networkx` (GraphRAG)
- `langchain-*` (chunking + experimental graph builders)
- `rapidfuzz`, `json-repair`
- `pandas`, `numpy`, `python-dotenv`
- `datasets` (HuggingFace, used by MIRAGE)

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

# OpenAI API (Task 1 graph-based QG with GPT-4o)
OPENAI_API_KEY=your_openai_key_here

# Eurecom LLM gateway (Task 2 generator LLM, Qwen3)
EURECOM_LLM_URL=https://...
EURECOM_LLM_KEY=your_eurecom_key_here
EURECOM_LLM_MODEL=qwen3-instruct
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

### Task 2 — RAG / GraphRAG pipelines

Run all pipelines (classical RAG, hybrid RAG, GraphRAG) on the four Task 1
question files. The script is checkpointed: a crash or gateway outage can be
resumed by re-running the same command.

```bash
python task2_setup_rag/run_pipelines.py
```

Outputs land in `task2_setup_rag/output/`:
- `rag_results.jsonl` — dense vector retrieval
- `rag_hybrid_results.jsonl` — dense + BM25 fusion
- `graphrag_results.jsonl` — ontology-constrained graph traversal

Each line contains the question, the retrieved context, the generated answer
(`llm_answer`), and the ground-truth answer, joined by `source_id` to the
originating sentence in Task 1.

### Task 3 — MIRAGE evaluation

**Step 1 — build MIRAGE-ready CSVs** from the Task 2 outputs. The `source`
column holds the original ground-truth sentence (not the retrieved context),
the `gen` column holds the LLM-generated answer (`llm_answer`), and
`ground_truth` holds the reference answer.

```bash
python task3_evaluation/prepare_mirage_input.py
```

Outputs: `task3_evaluation/inputs/{rag,graphrag,rag_hybrid}_input.csv`.

**Step 2 — run FactCC** on each pipeline. The script appends a
`faithfulness = 1 − score` column to the HuggingFace dataset, since MIRAGE's
raw `score` is P(hallucination), not P(faithful).

```bash
python task3_evaluation/run_mirage.py --pipeline both       # rag + graphrag
python task3_evaluation/run_mirage.py --pipeline hybrid     # rag_hybrid only
```

Outputs land in `task3_evaluation/outputs/factcc_{rag,graphrag,rag_hybrid}/`
as HuggingFace Arrow datasets.

**Step 3 — analyse and compare.** Global mean faithfulness, per-`hop_type`
breakdown, critical-case rate, faithfulness distribution by threshold,
hybrid-vs-baseline comparison, and the Pearson(faithfulness, sbert_cosine)
sanity check.

```bash
python task3_evaluation/analyze_results.py
```

The standalone diagnostics (`diagnostic_factcc.py`, `diagnostic_sbert.py`) are
kept in-repo for traceability of the FactCC orientation fix (see
[FactCC score direction memo](task3_evaluation/)).


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
- **FactCC** — Kryściński et al. *Evaluating the Factual Consistency of
  Abstractive Text Summarization*.
  [model](https://huggingface.co/manueldeprada/FactCC)
- **SBERT** — Reimers and Gurevych. *Sentence-BERT: Sentence Embeddings using
  Siamese BERT-Networks*.
  [sentence-transformers](https://www.sbert.net/)
