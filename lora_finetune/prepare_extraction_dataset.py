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
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import anthropic
from dotenv import load_dotenv
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from extraction_schema import EXTRACTION_SYSTEM_PROMPT, extract_json
from config import ExtractionLoRAConfig

MESSAGES_PATH = Path("test_data/extraction_messages.json")
EVAL_PER_CATEGORY = 10

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
    resp = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=300,
        system=EXTRACTION_SYSTEM_PROMPT,
        messages=FEW_SHOT + [{"role": "user", "content": text}],
    )
    raw = next(block.text for block in resp.content if block.type == "text")
    return extract_json(raw)


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

    labeled = []
    for i, m in enumerate(messages, 1):
        try:
            label = label_message(client, m["text"])
        except Exception as e:
            print(f"  [{i}/{len(messages)}] skip '{m['text'][:50]}...': {e}")
            continue
        labeled.append({**m, "label": label})
        if i % 25 == 0:
            print(f"  labeled {i}/{len(messages)}")

    if not labeled:
        print("No examples labeled - check ANTHROPIC_API_KEY.")
        return

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
