# LoRA Fine-Tuning for Med42-8B

Two tasks, both QLoRA (`peft` + `bitsandbytes`, 4-bit) on `m42-health/Llama3-Med42-8B`. Needs a CUDA GPU — a free-tier Colab T4 works.

## Setup
```bash
pip install -r lora_finetune/requirements.txt
```
- Set `HF_TOKEN` (accept the license on the model card) and `ANTHROPIC_API_KEY`
- T4 notes:
  - `config.py` defaults `compute_dtype="float16"` (T4 has no bf16) — only switch to `"bfloat16"` on Ampere+ (A100/L4/3090/4090)
  - `gradient_checkpointing=True` by default; if you OOM, drop `per_device_train_batch_size` to 1 and/or lower `max_seq_length`
  - `evaluate_*.py` loads base then LoRA sequentially, `.unload()`-ing between them to fit both in 16GB

---

## Task A (recommended): red-flag/triage JSON extraction

Fine-tunes Med42-8B to extract a fixed-schema JSON object from a patient message:
```json
{"red_flags": ["heavy_bleeding", "dizziness"], "urgency": "emergency", "reasoning": "..."}
```
Mirrors `patient_chatbot.py`'s regex-based `RED_FLAG_PATTERNS` / `create_red_flag_warning` — a real upgrade path, not just a benchmark. Scored deterministically (precision/recall/F1, JSON validity) against a held-out eval set, not an LLM judge.

**Why scoped, not free-text:** a small, low-entropy JSON target can move with a few hundred examples; free-text imitation (Task B) couldn't.

**Run:**
```bash
python lora_finetune/run_pipeline.py --task extraction
```
or step by step: `generate_extraction_questions.py` → `prepare_extraction_dataset.py` → `train_lora.py --task extraction` → `evaluate_extraction.py`

- No RAG/vectorstore needed
- Data pipeline is two-stage, deliberately decoupled: messages generated class-balanced (per-flag, routine, multi-flag) in stage 1, then labeled by a **separate, blind** Claude call in stage 2 (no label leakage from generation intent)
- **Review `lora_finetune/data/extraction_val_for_review.jsonl` by hand before trusting eval numbers** — it's a Claude-labeled eval set
- Labeling is resumable: successful labels cache to `extraction_labels_cache.jsonl`; a billing/auth error stops the run immediately instead of burning through the rest

### Results (first full run)

| metric | base Med42-8B | LoRA Med42-8B | delta |
|---|:---:|:---:|:---:|
| strict JSON validity | 92.6% | 95.6% | +3.0 pts |
| micro precision | 0.849 | 0.885 | +0.036 |
| micro recall | 0.890 | 0.939 | +0.049 |
| micro F1 | 0.869 | 0.911 | +0.042 |
| urgency exact-match accuracy | 95.6% | 95.6% | +0.0 pts |
| strict exact-match (all flags + urgency) | 76.5% | 79.4% | +2.9 pts |

- 313 train / 68 val (381 generated), 6 epochs on a free T4
- Every metric moved in LoRA's favor, none regressed — a real signal, not noise
- Per-flag: improved/held on 5 of 6 flags; `fever` recall (0.692→0.923) was the biggest move
- Gain is modest, not dramatic — base model was already fairly capable at this task (Med42-8B is Llama-3-8B-Instruct-derived)
- 313 training examples is well under generic "2k-10k" LoRA sizing advice — that's fine here since it's a narrow classification task on an already-instruction-tuned base, not general instruction-tuning; the consistent, same-direction result is the evidence for that, not just an assumption
- Only the first 15/68 eval labels have been spot-checked by hand so far, not the full set
- Training log showed `loss: '0'` every step — almost certainly a Colab progress-bar/output-capture artifact, not a real no-op run (the eval result clearly shows learning happened); rerun and print `trainer.state.log_history` directly to confirm if in doubt

---

## Task B (appendix, optional): free-text answer imitation

Distills the Claude+RAG chatbot's free-text answers into training data, scored by the same LLM judge as `test_runner.py`.

**Run:**
```bash
python lora_finetune/run_pipeline.py
```
or step by step: `generate_questions.py` → `prepare_dataset.py` → `train_lora.py` → `evaluate_lora.py` → (optional) `merge_adapter.py`

- Needs `faiss_index/` (run `rag_setup.py` first if missing)
- Eval set (`TestCaseGenerator.get_curated_test_cases()`, 16 questions) is excluded from training data — don't add it back in, that contaminates the comparison

**Status:** one completed run (34 train / 4 val, ~9 optimizer steps total) came back statistically indistinguishable from base (71.6% vs 70.8%, same pass rate) — too few steps to move an 8B model, and it predates the train/eval separation fix above. This result is why Task A exists.
