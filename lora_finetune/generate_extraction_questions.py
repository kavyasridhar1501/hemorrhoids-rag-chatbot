"""
Generates a class-balanced set of patient messages for the red-flag/triage
JSON-extraction fine-tuning task: one call per red flag (messages that
clearly exhibit that symptom), one for routine messages (no red flags at
all), and several for messages combining two flags at once (the case a
classifier is most likely to get wrong).

Output: test_data/extraction_messages.json - a flat list of
{"id", "text", "target_category"}. `target_category` is only a generation
hint used later to build a stratified train/eval split in
prepare_extraction_dataset.py; the actual training label comes from a
separate, blind labeling pass there, not from this file.
"""
import json
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from extraction_schema import RED_FLAGS, FLAG_DESCRIPTIONS

MESSAGES_PER_FLAG = 40
ROUTINE_MESSAGES = 80
MULTI_FLAG_MESSAGES = 60
OUTPUT_PATH = Path("test_data/extraction_messages.json")

FLAG_PROMPT = """Generate {n} realistic patient messages to a hemorrhoids/constipation chatbot. Each message must clearly describe: {description}.

Guidelines:
- Write as an actual worried or matter-of-fact patient would (first person, casual language)
- Vary severity phrasing, sentence length, and how much other context is included
- Most messages should mention hemorrhoid/constipation context alongside the symptom, but not all
- Do not repeat phrasing patterns from message to message
- Return ONLY a JSON array of {n} strings, no markdown fences, no other text"""

ROUTINE_PROMPT = """Generate {n} realistic patient messages to a hemorrhoids/constipation chatbot that do NOT mention any of the following: severe/unbearable pain, heavy bleeding or blood clots, fever, 3+ days without a bowel movement, black/tarry stool, or dizziness/fainting.

These should be everyday questions about symptoms, home treatment, diet, prevention, or general concern - normal severity, nothing urgent.

Guidelines:
- Write as an actual patient would (first person, casual language, varied tone)
- Vary length and specificity
- Return ONLY a JSON array of {n} strings, no markdown fences, no other text"""

MULTI_FLAG_PROMPT = """Generate {n} realistic patient messages to a hemorrhoids/constipation chatbot, each describing TWO of these symptoms together in the same message:
- {flag1}: {desc1}
- {flag2}: {desc2}

Guidelines:
- Write as an actual worried patient would (first person, casual language)
- Both symptoms must be clearly present in each message
- Return ONLY a JSON array of {n} strings, no markdown fences, no other text"""


def call_claude(client, prompt: str) -> list:
    message = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    text = next(block.text for block in message.content if block.type == "text")
    json_str = text[text.find("["):text.rfind("]") + 1]
    json_str = re.sub(r"[\x00-\x1f]+", " ", json_str)
    return json.loads(json_str)


def main():
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    rng = random.Random(42)
    messages = []

    for flag in RED_FLAGS:
        print(f"Generating {MESSAGES_PER_FLAG} messages for '{flag}'...")
        try:
            texts = call_claude(
                client, FLAG_PROMPT.format(n=MESSAGES_PER_FLAG, description=FLAG_DESCRIPTIONS[flag])
            )
        except Exception as e:
            print(f"  Failed: {e}")
            continue
        for i, text in enumerate(texts, 1):
            messages.append({"id": f"{flag}_{i:03d}", "text": text.strip(), "target_category": flag})
        print(f"  +{len(texts)}")

    print(f"Generating {ROUTINE_MESSAGES} routine messages...")
    try:
        texts = call_claude(client, ROUTINE_PROMPT.format(n=ROUTINE_MESSAGES))
        for i, text in enumerate(texts, 1):
            messages.append({"id": f"routine_{i:03d}", "text": text.strip(), "target_category": "routine"})
        print(f"  +{len(texts)}")
    except Exception as e:
        print(f"  Failed: {e}")

    flag_pairs = list(zip(RED_FLAGS, RED_FLAGS[1:] + RED_FLAGS[:1]))
    rng.shuffle(flag_pairs)
    per_pair = max(1, MULTI_FLAG_MESSAGES // len(flag_pairs))
    for f1, f2 in flag_pairs:
        print(f"Generating {per_pair} multi-flag messages for '{f1}+{f2}'...")
        try:
            texts = call_claude(
                client,
                MULTI_FLAG_PROMPT.format(
                    n=per_pair, flag1=f1, desc1=FLAG_DESCRIPTIONS[f1],
                    flag2=f2, desc2=FLAG_DESCRIPTIONS[f2],
                ),
            )
            for i, text in enumerate(texts, 1):
                messages.append({
                    "id": f"multi_{f1}_{f2}_{i:03d}",
                    "text": text.strip(),
                    "target_category": "multi_flag",
                })
            print(f"  +{len(texts)}")
        except Exception as e:
            print(f"  Failed: {e}")

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "generated_date": datetime.now().isoformat(),
            "total_messages": len(messages),
            "messages": messages,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(messages)} messages to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
