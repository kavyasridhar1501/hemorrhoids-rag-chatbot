"""
Builds the red-flag/triage JSON-extraction fine-tuning dataset.

Two-stage design, deliberately decoupled:
  1. generate_extraction_questions.py writes raw patient messages, each
     tagged with a `target_category` that is only a generation hint - used
     here purely to build a stratified split.
  2. This script labels each message with a SEPARATE, blind Claude call
     (the labeling prompt never sees target_category) using the same
     schema the fine-tuned model will be trained to produce. A message
     generated "for" heavy_bleeding that Claude's blind read doesn't
     actually judge that way ends up with the label Claude assigned, not
     the generation intent - avoiding label leakage from the generation step.

Writes lora_finetune/data/extraction_{train,val}.jsonl for train_lora.py,
plus a human-readable extraction_val_for_review.jsonl. Hand-check that file
before trusting evaluate_extraction.py's numbers - an unverified
Claude-labeled eval set is only as trustworthy as the labeler.
"""
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import anthropic
from dotenv import load_dotenv
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from extraction_schema import EXTRACTION_SYSTEM_PROMPT, extract_json, is_fatal_account_error
from config import ExtractionLoRAConfig

MESSAGES_PATH = Path("test_data/extraction_messages.json")
CACHE_PATH = Path("lora_finetune/data/extraction_labels_cache.jsonl")
EVAL_PER_CATEGORY = 10
MAX_RETRIES = 3

# Fixed few-shot examples for the labeling call - not model-generated, so
# they act as a stable anchor for what "correct" labeling looks like.
FEW_SHOT = [
    {"role": "user", "content": "I've been having really bad hemorrhoid pain, it's excruciating and I can barely sit down."},
    {"role": "assistant", "content": '{"red_flags": ["severe_pain"], "urgency": "emergency", "reasoning": "Patient describes excruciating pain, a critical red flag."}'},
    {"role": "user", "content": "What foods should I avoid to prevent hemorrhoid flare-ups?"},
    {"role": "assistant", "content": '{"red_flags": [], "urgency": "routine", "reasoning": "General prevention question with no concerning symptoms."}'},
    {"role": "user", "content": "I noticed my stool was black and tarry this morning and I also feel really dizzy."},
    {"role": "assistant", "content": '{"red_flags": ["black_stool", "dizziness"], "urgency": "emergency", "reasoning": "Black stool plus dizziness could indicate significant GI bleeding."}'},
]


def label_message(client, text: str) -> Dict:
    last_err: Optional[Exception] = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.messages.create(
                model="claude-sonnet-5",
                # 300 was too tight - verbose reasoning could truncate the
                # completion before the closing "}", making otherwise-valid
                # JSON unparseable and silently dropping the example.
                max_tokens=500,
                system=EXTRACTION_SYSTEM_PROMPT,
                messages=FEW_SHOT + [{"role": "user", "content": text}],
            )
            raw = next(block.text for block in resp.content if block.type == "text")
            return extract_json(raw)
        except Exception as e:
            if is_fatal_account_error(e):
                raise
            status = getattr(e, "status_code", None)
            if status in (429, 500, 502, 503, 529) and attempt < MAX_RETRIES:
                wait = 2 ** (attempt + 1)
                print(f"    transient error ({status}), retrying in {wait}s...")
                time.sleep(wait)
                last_err = e
                continue
            raise
    raise last_err


def stratified_split(messages: List[Dict], eval_per_category: int, seed: int):
    by_category = defaultdict(list)
    for m in messages:
        by_category[m["target_category"]].append(m)

    rng = random.Random(seed)
    train, val = [], []
    for items in by_category.values():
        rng.shuffle(items)
        n_eval = min(eval_per_category, max(1, len(items) // 5))
        val += items[:n_eval]
        train += items[n_eval:]
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def load_cache() -> Dict[str, Dict]:
    if not CACHE_PATH.exists():
        return {}
    cache = {}
    with open(CACHE_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                cache[row["id"]] = row["label"]
    return cache


def append_cache(message_id: str, label: Dict):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": message_id, "label": label}, ensure_ascii=False) + "\n")


def build_example(tokenizer, message: str, label: Dict) -> Dict:
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    completion = json.dumps(label, ensure_ascii=False)
    return {"prompt": prompt, "completion": completion}


def main():
    cfg = ExtractionLoRAConfig()
    if not MESSAGES_PATH.exists():
        print(f"{MESSAGES_PATH} not found - run generate_extraction_questions.py first.")
        return

    with open(MESSAGES_PATH, encoding="utf-8") as f:
        messages = json.load(f)["messages"]

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)

    cache = load_cache()
    if cache:
        print(f"Resuming from cache: {len(cache)} messages already labeled in a previous run.")

    labeled = []
    stopped_early = False
    for i, m in enumerate(messages, 1):
        if m["id"] in cache:
            labeled.append({**m, "label": cache[m["id"]]})
            continue
        try:
            label = label_message(client, m["text"])
        except Exception as e:
            if is_fatal_account_error(e):
                print(f"\nFatal error at [{i}/{len(messages)}]: {e}")
                print(
                    "This looks like an account issue (credits/auth), not a per-message "
                    "problem - stopping here instead of burning through the rest.\n"
                    f"{len(labeled)} messages labeled and cached to {CACHE_PATH} so far. "
                    "Fix the issue and re-run this script - it resumes from the cache "
                    "instead of relabeling (and re-paying for) messages already done."
                )
                stopped_early = True
                break
            print(f"  [{i}/{len(messages)}] skip '{m['text'][:50]}...': {e}")
            continue
        append_cache(m["id"], label)
        labeled.append({**m, "label": label})
        if i % 25 == 0:
            print(f"  labeled {i}/{len(messages)}")

    if not labeled:
        print("No examples labeled - check ANTHROPIC_API_KEY.")
        return

    if stopped_early:
        print(
            f"\nBuilding a dataset from the {len(labeled)}/{len(messages)} messages labeled "
            "so far - it will be smaller and less class-balanced than a full run. Re-run "
            "this script after fixing the account issue to top it up before training."
        )

    train_msgs, val_msgs = stratified_split(labeled, EVAL_PER_CATEGORY, cfg.seed)

    out_dir = Path("lora_finetune/data")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "extraction_train.jsonl", "w", encoding="utf-8") as f:
        for m in train_msgs:
            f.write(json.dumps(build_example(tokenizer, m["text"], m["label"]), ensure_ascii=False) + "\n")

    with open(out_dir / "extraction_val.jsonl", "w", encoding="utf-8") as f:
        for m in val_msgs:
            row = build_example(tokenizer, m["text"], m["label"])
            row["message"] = m["text"]
            row["gold"] = m["label"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(out_dir / "extraction_val_for_review.jsonl", "w", encoding="utf-8") as f:
        for m in val_msgs:
            f.write(json.dumps(
                {"text": m["text"], "target_category": m["target_category"], "label": m["label"]},
                ensure_ascii=False,
            ) + "\n")

    print(f"\nWrote {len(train_msgs)} train / {len(val_msgs)} val examples to {out_dir}/")
    print(f"Review {out_dir}/extraction_val_for_review.jsonl by hand before trusting evaluate_extraction.py's numbers.")


if __name__ == "__main__":
    main()
