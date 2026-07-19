"""
Compares base Med42-8B against the LoRA-adapted Med42-8B on the red-flag/
triage extraction task, using deterministic metrics (strict JSON validity,
per-flag precision/recall/F1, urgency accuracy, exact-match rate) computed
against the Claude-labeled, hand-reviewable eval set from
prepare_extraction_dataset.py - not an LLM judge. A fixed-schema
classification task can be scored exactly, which is the whole point of
scoping the fine-tuning task down to this instead of free-text imitation.
"""
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from extraction_schema import RED_FLAGS, EXTRACTION_SYSTEM_PROMPT, extract_json

from generate_lora import Med42Generator
from config import ExtractionLoRAConfig

VAL_PATH = Path("lora_finetune/data/extraction_val.jsonl")


def load_val() -> List[Dict]:
    with open(VAL_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def generate_all(generator: Med42Generator, examples: List[Dict], label: str) -> List[str]:
    print(f"\nGenerating {label} responses for {len(examples)} examples...")
    outputs = []
    for i, ex in enumerate(examples, 1):
        try:
            outputs.append(generator.generate(ex["message"], max_new_tokens=150))
        except Exception as e:
            print(f"  [{i}/{len(examples)}] error: {e}")
            outputs.append("")
        if i % 10 == 0 or i == len(examples):
            print(f"  [{i}/{len(examples)}]")
    return outputs


def score(examples: List[Dict], raw_outputs: List[str]) -> Dict:
    strict_valid = 0
    tp = {f: 0 for f in RED_FLAGS}
    fp = {f: 0 for f in RED_FLAGS}
    fn = {f: 0 for f in RED_FLAGS}
    urgency_correct = 0
    exact_match = 0
    n = len(examples)

    for ex, raw in zip(examples, raw_outputs):
        gold = ex["gold"]
        gold_flags = set(gold["red_flags"])

        try:
            json.loads(raw.strip())
            strict_valid += 1
        except Exception:
            pass

        try:
            pred = extract_json(raw)
        except Exception:
            pred = {"red_flags": [], "urgency": "routine", "reasoning": ""}
        pred_flags = set(pred["red_flags"])

        for flag in RED_FLAGS:
            if flag in pred_flags and flag in gold_flags:
                tp[flag] += 1
            elif flag in pred_flags and flag not in gold_flags:
                fp[flag] += 1
            elif flag not in pred_flags and flag in gold_flags:
                fn[flag] += 1

        if pred["urgency"] == gold["urgency"]:
            urgency_correct += 1
        if pred_flags == gold_flags and pred["urgency"] == gold["urgency"]:
            exact_match += 1

    total_tp, total_fp, total_fn = sum(tp.values()), sum(fp.values()), sum(fn.values())
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    per_flag = {}
    for flag in RED_FLAGS:
        p = tp[flag] / (tp[flag] + fp[flag]) if (tp[flag] + fp[flag]) else 0.0
        r = tp[flag] / (tp[flag] + fn[flag]) if (tp[flag] + fn[flag]) else 0.0
        per_flag[flag] = {
            "precision": round(p, 3),
            "recall": round(r, 3),
            "f1": round(2 * p * r / (p + r), 3) if (p + r) else 0.0,
            "support": tp[flag] + fn[flag],
        }

    return {
        "n": n,
        "strict_json_validity_rate": round(strict_valid / n, 3),
        "micro_precision": round(precision, 3),
        "micro_recall": round(recall, 3),
        "micro_f1": round(f1, 3),
        "urgency_accuracy": round(urgency_correct / n, 3),
        "exact_match_rate": round(exact_match / n, 3),
        "per_flag": per_flag,
    }


def main():
    cfg = ExtractionLoRAConfig()
    examples = load_val()
    if not examples:
        print(f"No eval examples in {VAL_PATH} - run prepare_extraction_dataset.py first.")
        return

    base_gen = Med42Generator(adapter_path=None, cfg=cfg, system_prompt=EXTRACTION_SYSTEM_PROMPT)
    base_raw = generate_all(base_gen, examples, "base Med42-8B")
    base_scores = score(examples, base_raw)
    base_gen.unload()
    del base_gen

    lora_gen = Med42Generator(adapter_path=cfg.output_dir, cfg=cfg, system_prompt=EXTRACTION_SYSTEM_PROMPT)
    lora_raw = generate_all(lora_gen, examples, "LoRA Med42-8B (extraction)")
    lora_scores = score(examples, lora_raw)
    lora_gen.unload()
    del lora_gen

    comparison = {
        "base_med42": base_scores,
        "lora_med42": lora_scores,
        "improvement": {
            "micro_f1": round(lora_scores["micro_f1"] - base_scores["micro_f1"], 3),
            "urgency_accuracy": round(lora_scores["urgency_accuracy"] - base_scores["urgency_accuracy"], 3),
            "exact_match_rate": round(lora_scores["exact_match_rate"] - base_scores["exact_match_rate"], 3),
            "strict_json_validity_rate": round(
                lora_scores["strict_json_validity_rate"] - base_scores["strict_json_validity_rate"], 3
            ),
        },
    }

    out_dir = Path("test_results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "lora_vs_base_extraction_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "comparison": comparison,
            "base_raw_outputs": base_raw,
            "lora_raw_outputs": lora_raw,
        }, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("LORA vs BASE MED42-8B - red-flag/triage extraction")
    print("=" * 70)
    print(f"Base  strict JSON valid: {base_scores['strict_json_validity_rate']:.1%}  "
          f"micro-F1: {base_scores['micro_f1']:.3f}  "
          f"urgency acc: {base_scores['urgency_accuracy']:.1%}  "
          f"exact match: {base_scores['exact_match_rate']:.1%}")
    print(f"LoRA  strict JSON valid: {lora_scores['strict_json_validity_rate']:.1%}  "
          f"micro-F1: {lora_scores['micro_f1']:.3f}  "
          f"urgency acc: {lora_scores['urgency_accuracy']:.1%}  "
          f"exact match: {lora_scores['exact_match_rate']:.1%}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
