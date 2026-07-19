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

## Train/eval separation
`evaluate_lora.py` scores against `TestCaseGenerator.get_curated_test_cases()`
(16 hand-written questions). `prepare_dataset.py` deliberately excludes that
same set from training data — training only draws from the scraped/synthetic
`test_data/*.json` files. Don't add the curated cases back into training; that
would contaminate the comparison.

## Steps

Run the whole sequence below in one command with:
```bash
python lora_finetune/run_pipeline.py
```
It skips the vectorstore build if `faiss_index/` already exists, and stops
with a clear error if any step fails. Or run the steps individually if you
want more control:

1. **(Optional but recommended) Generate more training questions** — the
   scraped `test_data/*.json` files alone are a small pool (~20 questions
   after dedup), which isn't enough to meaningfully fine-tune an 8B model.
   This calls Claude to generate ~150+ additional diverse synthetic patient
   questions across categories, written to `test_data/synthetic_test_cases.json`:
   ```bash
   python lora_finetune/generate_questions.py
   ```

2. **Build the training data** (needs `ANTHROPIC_API_KEY` and an existing
   `faiss_index/` — run `rag_setup.py` first if missing; embeddings run
   locally, no OpenAI key needed):
   ```bash
   python lora_finetune/prepare_dataset.py
   ```
   Writes `lora_finetune/data/train.jsonl` and `val.jsonl`.

3. **Fine-tune**:
   ```bash
   python lora_finetune/train_lora.py
   ```
   Saves the adapter to `lora_finetune/adapter/`. Hyperparameters live in
   `lora_finetune/config.py`. With only ~30-40 training examples this works
   out to single-digit total optimizer steps, too few for the adapter to
   learn anything reliable — step 1 above exists specifically to fix that.

4. **Evaluate base vs. LoRA**:
   ```bash
   python lora_finetune/evaluate_lora.py
   ```
   Writes `test_results/lora_vs_base_results.json` with per-dimension
   scores, pass rates, and the base-vs-LoRA delta.

5. **(Optional) Merge for deployment**:
   ```bash
   python lora_finetune/merge_adapter.py
   ```
   Merges the adapter into the base weights (`lora_finetune/merged_model/`)
   so it can be converted to GGUF for Ollama, the same way the base
   Med42-8B is served elsewhere in this project.

## Status (free-text task)
First full run (34 train / 4 val examples, 3 epochs, ~9 optimizer steps
total) completed but was inconclusive: LoRA scored within ~1 point of the
base model (71.6% vs 70.8%, identical 12.5% pass rate) — a difference small
enough to be judge noise rather than a real effect, consistent with too few
training steps to move an 8B model's behavior. That run also predates the
train/eval separation fix above, so its result shouldn't be trusted even
as a null result.

A follow-up run generated 156 synthetic questions (step 1) to scale up the
training set, but the training/evaluation steps after that weren't
confirmed to complete cleanly - re-run `python lora_finetune/run_pipeline.py`
end to end and check `test_results/lora_vs_base_results.json` for a result
that can actually be attributed to the larger dataset before reporting a
number anywhere.

This diffuse "imitate the whole free-text answer" task is also just a hard
one to move with a handful of LoRA steps - see the scoped task below, which
exists specifically to fix that.

## Scoped task: red-flag/triage JSON extraction

Instead of imitating Claude's full free-text answer, this task fine-tunes
Med42-8B to extract a small, fixed-schema JSON object from a patient
message:
```json
{"red_flags": ["heavy_bleeding", "dizziness"], "urgency": "emergency", "reasoning": "..."}
```
using the same flag taxonomy and critical/non-urgent grouping as
`patient_chatbot.py`'s regex-based `RED_FLAG_PATTERNS` /
`create_red_flag_warning` (see `extraction_schema.py`). A model that does
this reliably is a direct upgrade path for that regex logic, not just a
benchmark exercise.

**Why a separate task from the one above:** free-text generation is a huge,
high-entropy output space - a handful of LoRA steps can't visibly shift it,
which is exactly why the run above came back as noise. A 6-flag-set +
3-class JSON target is a small, low-entropy space that a few hundred
examples and a few epochs can actually move, and - critically - it can be
**scored deterministically** (precision/recall/F1, JSON-validity rate)
instead of relying on a 16-question LLM judge where a ~1-point difference
is meaningless.

### Data pipeline (two-stage, deliberately decoupled)
1. `generate_extraction_questions.py` — generates a *class-balanced* set of
   patient messages via Claude: ~40 per red flag, 80 routine (no red flags
   at all), and multi-flag combinations (~60), written to
   `test_data/extraction_messages.json`. Class balance matters here more
   than in the free-text task - without it the model just learns to always
   predict "routine".
2. `prepare_extraction_dataset.py` — labels each message with a **separate,
   blind** Claude call (the labeling prompt never sees which flag a message
   was generated for) using the same schema/few-shot format the fine-tuned
   model is trained to produce. This avoids label leakage from the
   generation step. Builds a stratified train/val split (every flag and
   urgency level represented in both) and writes
   `lora_finetune/data/extraction_{train,val}.jsonl`, plus
   `extraction_val_for_review.jsonl` - a human-readable dump of the eval
   set's labels. **Hand-review that file before trusting
   `evaluate_extraction.py`'s numbers** - a Claude-labeled eval set is only
   as trustworthy as the labeler, and this one hasn't been manually
   verified yet.

No RAG/vectorstore step is needed for this task at all (unlike the
free-text task) — it's a direct classification task over the raw patient
message, not a retrieval-augmented answer.

### Run it
```bash
python lora_finetune/run_pipeline.py --task extraction
```
or step by step:
```bash
python lora_finetune/generate_extraction_questions.py
python lora_finetune/prepare_extraction_dataset.py
python lora_finetune/train_lora.py --task extraction
python lora_finetune/evaluate_extraction.py
```
Uses `ExtractionLoRAConfig` in `config.py` (shorter `max_seq_length=256`
since completions are compact JSON, not paragraphs; more epochs since each
step is cheaper). Adapter saves to `lora_finetune/adapter_extraction/`,
results to `test_results/lora_vs_base_extraction_results.json`.

### Metrics reported
For base vs. LoRA Med42-8B: strict JSON-validity rate, micro-averaged
flag precision/recall/F1 (plus per-flag breakdown), urgency-level exact-
match accuracy, and strict exact-match rate (all flags + urgency correct).
These are computed directly from parsed output against gold labels, not
judged by an LLM.

### Status
Not yet run — this task was added after the free-text run above and needs
a Colab session to produce real numbers. Once it's run, replace this
section with the actual base-vs-LoRA table and, per the section above,
don't report a number here until `extraction_val_for_review.jsonl` has
actually been reviewed.
