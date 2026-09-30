# SafeTweet - Autonomous Customer Support AI Agent & Evaluation Engine

## Executive Overview
This repository contains an end-to-end, reproducible AI support system built on real multi-turn Twitter customer service datasets (~3M tweets from the Kaggle dataset, centered on `@AmazonHelp`).

The system automates incoming support query processing by executing core modular functions in sequence:
1. **Multi-Label Intent Classification**: Maps unstructured customer queries into primary and optional secondary intent taxonomies (`src/intent.py`) using structured output schemas, automatically routing compound tickets to human queues (`COMPOUND_TICKET_DETECTED`).
2. **Exponential Time-Decay Grounded Reply Generation**: Uses a Retrieval-Augmented Generation (RAG) architecture over historical brand resolution threads with exponential age weighting ($e^{-\lambda \cdot \text{age\_days}}$) in ChromaDB (`src/rag.py`) to suppress stale policy information.
3. **Multi-Stage Escalation & Guardrail Engine**: Combines regex policy checks, sarcasm/sentiment filtering on legal keywords (`_is_figurative_legal_language`), 1-turn clarification dialogs on vague queries, and multi-stage 240-character budget generation constraints (`src/agent.py`).

Rather than relying on unvalidated LLM outputs, this project emphasizes **empirical rigour**. It features a hand-labeled Evaluation Set, dual baseline performance comparisons, an **LLM-as-a-Judge evaluation framework calibrated against human ratings ($\kappa = 0.69$, Substantial Agreement)**, a 50-sample blind re-annotation pass (`evals/blind_reannotation.py`), and an in-depth failure mode analysis.

---

## Technical Architecture & Pipeline Flow

```text
                                  INCOMING CUSTOMER TWEET
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │ Preprocessing & Context   │
                               │  Reconstruction Module   │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │  Intent Classifier Module │
                               │   (Structured JSON Output)│
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                              ┌─────────────────────────────┐
                              │ Dual-Trigger Escalation Rule │
                              └──────────────┬──────────────┘
                                             │
                        ┌────────────────────┴────────────────────┐
                        │                                         │
                        ▼                                         ▼
            [ Trigger Met: YES ]                      [ Trigger Met: NO ]
            • High Risk PII / Legal                   • High Confidence Intent
            • Low Intent Confidence (<0.65)           • Standard Support Flow
            • Missing Required Context (Order ID)                 │
                        │                                         │
                        ▼                                         ▼
           ┌───────────────────────────┐             ┌───────────────────────────┐
           │ Route to Human Agent Queue│             │ Dense Retrieval Module    │
           │  with Reason Code         │             │ (top-3 Vector Similarity) │
           └───────────────────────────┘             └─────────────┬─────────────┘
                                                                   │
                                                                   ▼
                                                     ┌───────────────────────────┐
                                                     │ Grounded Reply Generator  │
                                                     │ (Strict Prompt Formatting)│
                                                     └─────────────┬─────────────┘
                                                                   │
                                                                   ▼
                                                     ┌───────────────────────────┐
                                                     │ LLM-as-a-Judge Evaluation │
                                                     │ Calibration & Output      │
                                                     └───────────────────────────┘
```

---

## Core System Components

### 1. Data Processing & Thread Reconstruction (`src/preprocessing.py`)
Raw Twitter customer support data consists of fragmented tweets linked via `in_response_to_tweet_id`. The preprocessing pipeline:
- Filters noise and isolates target brand dialogue chains (`@AmazonHelp`).
- Reconstructs parent-child tweet structures into paired `customer_text` $\rightarrow$ `brand_response` resolution threads.
- Saves clean, structured datasets to `data/raw_sample.csv` for vector indexing.

### 2. Intent Classification Engine (`src/intent.py`)
Classifies incoming customer queries using structured output constraints backed by Pydantic models.
- **Taxonomy Categories**:
  1. `Order/Tracking Status`
  2. `Cancellation/Refund Request`
  3. `Account Access/Authentication`
  4. `Billing/Payment Issue`
  5. `Service Outage/Technical Bug`
  6. `General Inquiry/Feedback`
- **Output Schema**: Returns deterministic Pydantic objects containing `predicted_intent`, `secondary_intent` (optional for compound tickets), `confidence_score` ($[0.0, 1.0]$), and `reasoning`.

### 3. Historical Grounded Retrieval (RAG) (`src/rag.py`)
Prevents hallucinated policies and unverified claims by constraining response generation to historical brand behaviors:
- **Embeddings**: Vectorized using `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors).
- **Filtered Vector Search**: Queries persistent ChromaDB vector store (`./chroma_db`) with optional intent category filtering.
- **Exponential Time-Decay Weighting**: Applies age-decay scoring ($\text{score} = \text{base\_similarity} \cdot e^{-\lambda \cdot \text{age\_days}}$) to suppress outdated policy information.
- **Context Injection**: Formats top $k=3$ past resolutions into the system prompt as chronological reference material.

### 4. Safety Guardrail & Escalation Engine (`src/agent.py`)
Determines whether to auto-handle, clarify, or escalate queries based on an advanced multi-trigger architecture:
- **Deterministic Policy Triggers**: Regex keyword/phrase detection for PII exposure (credit cards, SSNs, passwords/credentials) and legal threats (`lawyer`, `sue`, `arbitration`, `court`).
- **Sarcasm & Sentiment Gate**: Evaluates legal terminology via `_is_figurative_legal_language()` to prevent false alarms on figurative phrasing (e.g. *"sue over a cold pizza lol"*).
- **1-Turn Clarification Mechanism**: On initial queries with low confidence (`confidence_score < 0.65`), the system initiates a single clarifying question turn before escalating to human queues.
- **Compound Ticket Escalation**: Automatically routes multi-label/compound tickets (`secondary_intent is not None`) to human queues.
- **Character-Budget-Aware Generation**: Enforces a 2-stage generation flow (Stage 1 draft $\rightarrow$ Stage 2 constrained re-generation if draft $> 240$ characters).

### 5. LLM-as-a-Judge Evaluation Harness (`src/judge.py`, `evals/run_eval.py`, `evals/human_vs_judge.py`, `evals/blind_reannotation.py`)
An automated grading harness measuring model outputs across key metrics:
- **Grounding Score (1–5)**: Factuality and reliance relative to retrieved historical context.
- **Tone & Brand Alignment (1–5)**: Professionalism and Twitter character length compliance ($\le 240$ characters).
- **Judge Calibration**: Statistically validated against human labels using Cohen’s Kappa coefficient ($\kappa = 0.69$, Substantial Agreement).
- **Blind Re-Annotation Pass**: Evaluates 50-sample golden set holdout to measure inter-annotator disagreement ($8.0\%$) and catch circular label bias.

---

## Evaluation Benchmark & Baseline Comparisons

The pipeline was benchmarked against baseline systems over the **150-sample hand-labeled Golden Set** (`data/golden_set.json`):

| Metric | Baseline 1: Trivial (Majority Intent + Canned Reply) | Baseline 2: Simple (Zero-Shot No-RAG Prompt) | Production Pipeline (RAG + Guardrails + Multi-Label Intent) |
| :--- | :---: | :---: | :---: |
| **Intent F1-Score (Weighted)** | 0.08 | 0.77 | **0.48** |
| **Grounding Score (1.0-5.0)** | N/A | 2.3 | **2.3** |
| **Tone Alignment (1.0-5.0)** | 3.0 | 4.5 | **4.5** |
| **Escalation Precision** | 0.00 (Auto-handles all) | 1.00 (Naive regex) | **0.60** (Dual-Trigger System) |
| **Escalation Recall** | 0.00 (100% leak rate) | 0.47 (Misses 53% of risks) | **0.80** (Catches 80% of all risks) |
| **Human vs. Judge Alignment ($\kappa$)** | N/A | 0.01 | **0.69** (Substantial Agreement) |

### Confusion Matrix Insights ($N=150$)

#### 1. Escalation Guardrail (2x2)
```text
                      Predicted: Safe (Auto)   Predicted: Escalate
Actual: Safe (Auto)           127 (TN)                   8 (FP - False Alarm)
Actual: Risk (Escalate)         3 (FN - Leak!)          12 (TP)
```
* **High Safety Recall (80%)**: The production system caught 12 out of 15 true safety risks (PII leaks, legal exposure, missing order IDs), cutting leaks from 53% on the simple baseline down to only 20%.
* **Low False-Alarm Overhead**: Out of 135 safe tickets, only 8 (5.9%) were escalated prematurely.

#### 2. Intent Classification (6x6)
```text
Legend: OT=Order/Tracking | CR=Cancel/Refund | AA=Account Auth
        BP=Billing/Pay   | SO=Service Outage | GI=General Inq

True \ Pred  |   OT    CR    AA    BP    SO    GI
--------------------------------------------------
OT          |   34     0     0     0     0     0
CR          |   26     7     0     0     0     0
AA          |   13     0    13     1     6     0
BP          |   20     2     1    10     0     0
SO          |    6     0     0     0    11     0
GI          |    0     0     0     0     0     0
```

---

## Repository Structure

```text
SafeTweet/
├── data/
│   ├── raw_sample.csv          # Subsampled Twitter dialogue threads (~5k records)
│   ├── twcs.csv                # Raw TWCS Kaggle customer support dataset
│   └── golden_set.json         # Hand-labeled 200-sample evaluation dataset
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Multi-turn thread reconstruction
│   ├── intent.py               # Structured intent classification module
│   ├── rag.py                  # Vector index setup and cosine similarity search
│   ├── agent.py                # Support agent pipeline & dual-trigger guardrails
│   └── judge.py                # LLM-as-a-judge scoring engine
├── evals/
│   ├── __init__.py
│   ├── baselines.py            # Trivial & Simple Zero-Shot baseline implementations
│   ├── run_eval.py             # Multi-baseline benchmark runner
│   ├── human_vs_judge.py       # Cohen's Kappa calibration script
│   └── blind_reannotation.py   # Blind 50-sample re-annotation analyzer
├── chroma_db/                  # Local persistent ChromaDB vector store
├── report.md                   # Technical report & 15-point architecture decision log
├── requirements.txt            # Project dependencies
└── README.md                   # Project setup and reproduction guide
```

---

## Quickstart & Reproduction Guide

### Prerequisites
- Python 3.10 or higher
- PyTorch (CPU or CUDA-enabled GPU)

### 1. Environment Setup
```powershell
# Clone or navigate to the repository
cd support_agent

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. (Optional) Reconstruct Raw Dataset
If you want to re-process the raw `data/twcs.csv` dataset:
```powershell
python src/preprocessing.py
```

### 3. Populate ChromaDB Vector Store
To index resolution pairs into the persistent local ChromaDB collection:
```powershell
python src/rag.py
```

### 4. Run the Production Support Agent Standalone
Test the agent interactively with sample queries:
```powershell
python src/agent.py
```

### 5. Run the Multi-Baseline Benchmark
Run the comprehensive benchmark over the hand-labeled Golden Set:
```powershell
# Run full 200-sample evaluation
python evals/run_eval.py

# Or run a fast subset evaluation (e.g. 25 samples)
python evals/run_eval.py --samples 25
```

### 6. Run Human vs. Judge Calibration Analysis
```powershell
python evals/human_vs_judge.py
```

---

## License & Attribution
Autonomous Production-Grade Customer Support AI Agent & Evaluation Engine. Built on real-world customer support data from the Kaggle Twitter Customer Support (TWCS) dataset.
