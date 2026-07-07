"""
Builds instruction/response training pairs for LoRA fine-tuning of Med42-8B.

Targets are distilled from the existing RAG chatbot (patient_chatbot.py) with
its LLM swapped to Gemini (free tier) instead of Claude: each patient question
is answered by the RAG-grounded, safety-prompted chatbot, and that answer
becomes the training completion. This teaches Med42-8B to imitate the
teacher's safety-first, patient-friendly, red-flag-aware behavior on top of
its own medical domain knowledge, rather than requiring a hand-labeled
dataset.
"""
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from patient_chatbot import PatientChatbot, load_vectorstore, SYSTEM_PROMPT
from test_runner import TestCaseGenerator
from config import LoRAConfig

load_dotenv()


def collect_questions() -> List[Dict]:
    # Deliberately excludes TestCaseGenerator.get_curated_test_cases() - those
    # 16 questions are the held-out set evaluate_lora.py scores against, and
    # must never appear in training data or the comparison is contaminated.
    cases = []
    for filename in ["forum_test_cases.json", "google_test_cases.json", "manual_test_cases.json", "synthetic_test_cases.json"]:
        cases += TestCaseGenerator.load_forum_cases(f"test_data/{filename}")

    seen = set()
    unique = []
    for case in cases:
        question = case["question"].strip()
        if question and question not in seen:
            seen.add(question)
            unique.append(case)
    return unique


def build_examples(tokenizer, vectorstore_path: str = "./faiss_index") -> List[Dict]:
    vectorstore = load_vectorstore(vectorstore_path)
    teacher_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )
    chatbot = PatientChatbot(vectorstore, patient_id="lora_dataset_builder", llm=teacher_llm)

    examples = []
    for case in collect_questions():
        question = case["question"]
        try:
            answer = chatbot.chat(question)
        except Exception as e:
            print(f"Skipping '{question[:60]}...': {e}")
            continue

        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        examples.append({
            "prompt": prompt,
            "completion": answer.strip(),
            "category": case.get("category", "unknown"),
        })
    return examples


def main():
    cfg = LoRAConfig()
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)

    examples = build_examples(tokenizer)
    if not examples:
        print("No examples generated - check GEMINI_API_KEY and faiss_index/.")
        return

    random.Random(cfg.seed).shuffle(examples)
    split = max(1, int(len(examples) * 0.9))
    train, val = examples[:split], examples[split:]

    out_dir = Path("lora_finetune/data")
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, split_examples in [("train.jsonl", train), ("val.jsonl", val)]:
        with open(out_dir / name, "w", encoding="utf-8") as f:
            for ex in split_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(train)} train / {len(val)} val examples to {out_dir}/")


if __name__ == "__main__":
    main()
