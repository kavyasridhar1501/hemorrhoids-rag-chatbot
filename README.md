## Patient RAG Chatbot: Hemorrhoids & Constipation

A patient-friendly RAG chatbot focused on hemorrhoids and constipation. It provides empathetic guidance, detects red flags, and grounds answers in trusted medical sources (ACG, ASCRS, AGA). The system includes persistent conversation memory and a rigorous evaluation pipeline (LLM-as-judge + human/doctor review).

### Key Features
- Patient-centered responses with safety-first behavior and red-flag escalation
- Retrieval-Augmented Generation (RAG) over curated medical documents
- Persistent conversation memory across sessions
- Evaluation framework with both LLM-as-judge and human/doctor review
- Local FAISS vectorstore; documents live in `documents/`
- LoRA fine-tuning of Med42-8B, distilled from the Claude + RAG chatbot and benchmarked against the base model (`lora_finetune/`)

### Tech Stack
- LLM: Claude (primary); Med42-8B via Ollama for comparison, and as a LoRA fine-tuning target
- RAG: LangChain + FAISS + local HuggingFace embeddings (`sentence-transformers/all-MiniLM-L6-v2`, no API key/cost)
- Evaluation: Custom test runner, LLM-as-judge (Claude), optional human/doctor scoring
- Fine-tuning: 4-bit QLoRA (`peft` + `bitsandbytes`) on `m42-health/Llama3-Med42-8B`

---

## Getting Started

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Environment variables
Create a `.env` file in the project root:
```
ANTHROPIC_API_KEY=your_anthropic_key
# Optional patient context via Supabase
SUPABASE_URL=...
SUPABASE_ANON_KEY=...
HEMS_USER_ID=...                              
```
Embeddings run locally (`sentence-transformers/all-MiniLM-L6-v2`) — no OpenAI key needed.

### 3) Prepare documents and build the vectorstore
Add PDFs/TXT/MD/CSV/HTML to `documents/`, then run:
```bash
python rag_setup.py
```
This creates `faiss_index/` used at runtime.

### 4) Run the chatbot
```bash
python patient_chatbot.py
```
- If the vectorstore is missing, run `rag_setup.py` first.

### 5) (Optional) Fine-tune Med42-8B with LoRA
Two fine-tuning tasks, both QLoRA on `m42-health/Llama3-Med42-8B`, and both
need a CUDA GPU (a free-tier Colab T4 is enough):
```bash
# Free-text: distills the Claude+RAG chatbot's answers into a training set
python lora_finetune/run_pipeline.py

# Scoped: red-flag/triage JSON extraction, scored deterministically
# (precision/recall/F1) instead of by an LLM judge
python lora_finetune/run_pipeline.py --task extraction
```
See `lora_finetune/README.md` for the individual steps, hyperparameters,
and current results if you'd rather run it piece by piece.

---

## Project Architecture

**Knowledge base setup** (`rag_setup.py`):
```
Medical Documents  ->  Chunking          ->  Embeddings           ->  FAISS Vector Index
(PDF/TXT/MD/DOCX)      (1000 chars,          (local sentence-          (stored on disk)
                        200 overlap)          transformers)
```

**Q&A flow** (`patient_chatbot.py`):
```
User Question -> Retriever -> Prompt Builder -> LLM Response -> Safety Check -> Save Conversation
                 (top-4        (system rules     (Claude /                     History
                  chunks)       + retrieved        Med42-v2)
                                 snippets +
                                 conversation
                                 history +
                                 user question)
```

1. User asks a question
2. Retriever fetches relevant chunks from FAISS
3. Prompt composes: system safety policy + patient context + retrieved medical context + chat history
4. LLM generates response (Claude by default)
5. Safety check flags severe symptoms; red-flag post-processor appends urgent guidance if needed
6. Conversation history is saved

---

## Evaluation Methodology

### Models Compared
- Claude (RAG): primary production configuration
- Med42-8B (Ollama, RAG): baseline comparison

### Datasets
- Curated test cases covering: common queries, red flags, edge cases, emotional support, follow-ups, myths, pregnancy safety
- Optional forum-derived cases (see `testing_framework.py` web scraping utilities)

### LLM-as-Judge (Automated)
- Implemented in `testing_framework.py::LLMJudgeEvaluator`
- Uses Claude with a structured rubric across five dimensions:
  - Medical Accuracy, Safety & Red Flags, Patient-Friendliness, Actionability, Scope
- Produces JSON with dimension scores, overall percentage, and PASS/REVISE/FAIL
- Strength: fast, scalable early signal
- Limitation: tends to overestimate weaker models and can miss subtle safety/clarity failures; not a replacement for human review

### Human/Doctor Review
- Implemented in `testing_framework.py::HumanEvaluationInterface`
- Clinician evaluates a subset for:
  - Medical accuracy/safety
  - Clarity and actionability
  - Appropriateness of scope and empathy
- Considered the source of truth when disagreeing with LLM-as-judge

### How to Re-run the Evaluation
```bash
python test_runner.py
```
Follow the prompts to:
- Generate responses
- Run LLM-as-judge
- Optionally run human evaluation and save results to `test_results/`

---

## Results Summary

### Dimension scores (out of 10)

**Claude (Sonnet)**

| Dimension              | Human Evaluation | LLM-as-Judge |
|-------------------------|:---:|:---:|
| Medical Accuracy        | 8.71 | 9.12 |
| Safety                  | 8.68 | 9.03 |
| Patient-Friendliness    | 9.03 | 9.44 |
| Actionability           | 9.00 | 9.31 |
| Scope Appropriateness   | 8.89 | 9.27 |
| **Total Score (Avg)**   | **8.86 / 10** | **9.23 / 10** |
| **Percentage**          | **88.6%** | **92.3%** |

**Med42-8B (Ollama)**

| Dimension              | Human Evaluation | LLM-as-Judge |
|-------------------------|:---:|:---:|
| Medical Accuracy        | 7.11 | 8.02 |
| Safety                  | 7.79 | 8.21 |
| Patient-Friendliness    | 7.05 | 8.51 |
| Actionability           | 7.34 | 8.42 |
| Scope Appropriateness   | 7.89 | 8.49 |
| **Total Score (Avg)**   | **7.44 / 10** | **8.33 / 10** |
| **Percentage**          | **74.4%** | **83.3%** |

### Pass rates: LLM-as-Judge vs. human/doctor review

**Claude Model**

| Verdict | Human | LLM Judge | Observation |
|---|:---:|:---:|---|
| PASS    | 29 | 35 | +6 extra passes |
| REVISE  | 8  | 3  | Misses 5 |
| FAIL    | 1  | 0  | Misses 1 |

**Med42-8B Model**

| Verdict | Human | LLM Judge | Observation |
|---|:---:|:---:|---|
| PASS    | 10 | 20 | +10 extra passes |
| REVISE  | 17 | 17 | Same number but mismatched |
| FAIL    | 11 | 1  | Misses 10 (91%!) |

### LLM-as-judge reliability analysis

| Metric | Claude Eval | Llama Eval | Interpretation |
|---|:---:|:---:|---|
| Score Correlation (LLM vs Human) | 0.18 | 0.20 | Weak agreement |
| Verdict Match Rate | 74% | 34% | LLM mis-judges Llama heavily |
| Average Over-scoring | +2.8 pts | +8.7 pts | LLM is overly generous, especially for Med42 |
| PASS mis-classification | 8 cases | 17 cases | Many unsafe answers incorrectly marked PASS |

The LLM-as-judge system **cannot be trusted alone**, especially for medical evaluation. It often overrates the Med42-8B outputs and marks unsafe responses as acceptable.

Key findings:
- Claude (RAG) consistently outperforms Med42-8B across all evaluation dimensions.
- The performance gap is large, meaningful, and statistically significant.
- Claude delivers clearer, more accurate, and more actionable medical advice.
- Med42-8B produces too many unclear, incomplete, or unsafe responses.
- Claude shows far higher pass rates and more stable behavior across test cases.
- Med42-8B frequently misses red flags and struggles with condition identification.
- LLM-as-judge regularly overestimates Med42-8B’s quality and misses failures.
- Automated judging can support quick screening but cannot replace human review.

### LoRA fine-tuning results (red-flag/triage extraction)

Scoped task: fine-tune Med42-8B to extract `{red_flags, urgency, reasoning}`
JSON from a patient message, scored deterministically (not by an LLM judge)
against a 68-example held-out eval set.

| metric | base Med42-8B | LoRA Med42-8B | delta |
|---|:---:|:---:|:---:|
| strict JSON validity | 92.6% | 95.6% | +3.0 pts |
| micro F1 (red-flag detection) | 0.869 | 0.911 | +0.042 |
| urgency exact-match accuracy | 95.6% | 95.6% | +0.0 pts |
| strict exact-match rate | 76.5% | 79.4% | +2.9 pts |

- 313 train / 68 val, 6 epochs on a free Colab T4
- Every metric moved in LoRA's favor, none regressed
- An earlier, unscoped attempt (imitating full free-text answers instead of
  a fixed JSON schema) came back statistically indistinguishable from base

Full breakdown, caveats, and dataset-size rationale: `lora_finetune/README.md`.

---

## Repository Guide
- `patient_chatbot.py`: main chat loop; safety logic and RAG chain
- `conversation_memory.py`: lightweight persistent memory
- `rag_setup.py`: document ingestion and FAISS vectorstore creation
- `testing_framework.py`: web scraping utilities, LLM-as-judge, human evaluator
- `test_runner.py`: end-to-end generation + evaluation orchestrator
- `documents/`: source medical material (guidelines, patient center)
- `faiss_index/`: persisted vectorstore (generated)
- `test_data/` and `test_results/`: inputs and outputs for evaluation
- `integrations/supabase_utils.py`: optional patient context integration
- `lora_finetune/`: LoRA fine-tuning pipeline for Med42-8B - `run_pipeline.py` runs it end to end; see its own README for individual steps and current status
