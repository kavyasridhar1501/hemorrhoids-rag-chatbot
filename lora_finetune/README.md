# LoRA Fine-Tuning for Med42-8B

Rebuilt from scratch. The original exploration mentioned in the top-level
README was deleted from the author's machine before it was ever committed to
git, so no prior code or results could be recovered — this is a new
implementation, not a restoration.

## Approach
`patient_chatbot.py` already produces safety-first, patient-friendly,
RAG-grounded answers via Claude. Instead of hand-labeling a new dataset,
`prepare_dataset.py` distills those answers into instruction/response pairs
and uses them to fine-tune `m42-health/Llama3-Med42-8B` with LoRA, so the
adapter learns to imitate Claude's tone, safety behavior, and red-flag
handling on top of Med42's medical domain knowledge. The result is then
scored against the un-tuned base model with the same Claude LLM-as-judge
(`testing_framework.LLMJudgeEvaluator`) used by `test_runner.py`.

## Requirements
Needs a CUDA GPU with roughly 16GB+ VRAM (Med42-8B in 4-bit plus the LoRA
adapter). **This has not been run** — the environment that generated this
code has no GPU and no `torch` installed, so it could only be written and
structurally checked, not executed. Run the steps below on your own GPU
machine.

```bash
pip install -r lora_finetune/requirements.txt
```

The base model is gated on Hugging Face; set `HF_TOKEN` (and accept the
model license on its model card) if `from_pretrained` fails with an auth
error.

### Running on Google Colab (free tier)
A free-tier T4 (16GB VRAM) should fit this, with two things to know:

- **T4 is Turing, not Ampere — it has no `bfloat16` support.** `config.py`
  defaults `compute_dtype` to `"float16"` for exactly this reason. Only
  switch it to `"bfloat16"` if you're on an A100/L4/3090/4090 (Colab Pro
  or your own box).
- `gradient_checkpointing` is on by default in `config.py` — needed to fit
  the activation memory of an 8B model on 16GB. If you still hit an OOM,
  drop `per_device_train_batch_size` to 1 and/or `max_seq_length` to 512-768
  in `config.py` (raise `gradient_accumulation_steps` to compensate).

Also note: `evaluate_lora.py` loads the base model and the LoRA-adapted
model one after another in the same process and calls `.unload()` between
them to free VRAM — if you adapt it into a notebook, keep that pattern (or
restart the runtime between the two) rather than holding both in memory
at once.

Mount time and GPU availability aren't guaranteed on the free tier, but
given how small this project's dataset is (the curated + scraped test
cases, well under a thousand examples), a few epochs of training and the
eval pass should each finish in well under an hour once a GPU is assigned.

## Steps

1. **Build the training data** (needs `ANTHROPIC_API_KEY` and an existing
   `faiss_index/` — run `rag_setup.py` first if missing; embeddings run
   locally, no OpenAI key needed):
   ```bash
   python lora_finetune/prepare_dataset.py
   ```
   Writes `lora_finetune/data/train.jsonl` and `val.jsonl`.

2. **Fine-tune**:
   ```bash
   python lora_finetune/train_lora.py
   ```
   Saves the adapter to `lora_finetune/adapter/`. Hyperparameters live in
   `lora_finetune/config.py`.

3. **Evaluate base vs. LoRA**:
   ```bash
   python lora_finetune/evaluate_lora.py
   ```
   Writes `test_results/lora_vs_base_results.json` with per-dimension
   scores, pass rates, and the base-vs-LoRA delta.

4. **(Optional) Merge for deployment**:
   ```bash
   python lora_finetune/merge_adapter.py
   ```
   Merges the adapter into the base weights (`lora_finetune/merged_model/`)
   so it can be converted to GGUF for Ollama, the same way the base
   Med42-8B is served elsewhere in this project.

## Status
No metrics yet — this needs to run on GPU hardware first. Once it has,
replace this section (and the note in the top-level README) with the
actual pass rates / dimension scores from
`test_results/lora_vs_base_results.json`.
