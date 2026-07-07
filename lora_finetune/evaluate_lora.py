"""
Compares base Med42-8B against the LoRA-adapted Med42-8B on the curated
test set, scored by the same Claude LLM-as-judge (testing_framework.LLMJudgeEvaluator)
used elsewhere in this repo.
"""
import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from test_runner import TestCaseGenerator
from testing_framework import LLMJudgeEvaluator

from generate_lora import Med42Generator
from config import LoRAConfig


def run_variant(generator: Med42Generator, test_cases: List[dict], label: str) -> List[str]:
    print(f"\nGenerating {label} responses for {len(test_cases)} cases...")
    responses = []
    for i, case in enumerate(test_cases, 1):
        try:
            responses.append(generator.generate(case["question"]))
            print(f"  [{i}/{len(test_cases)}] ok")
        except Exception as e:
            print(f"  [{i}/{len(test_cases)}] error: {e}")
            responses.append("")
    return responses


def main():
    cfg = LoRAConfig()
    test_cases = TestCaseGenerator.get_curated_test_cases()
    evaluator = LLMJudgeEvaluator()

    base_gen = Med42Generator(adapter_path=None, cfg=cfg)
    base_responses = run_variant(base_gen, test_cases, "base Med42-8B")
    base_eval = evaluator.batch_evaluate(test_cases, base_responses)
    del base_gen

    lora_gen = Med42Generator(adapter_path=cfg.output_dir, cfg=cfg)
    lora_responses = run_variant(lora_gen, test_cases, "LoRA Med42-8B")
    lora_eval = evaluator.batch_evaluate(test_cases, lora_responses)
    del lora_gen

    comparison = {
        "base_med42": base_eval["summary"],
        "lora_med42": lora_eval["summary"],
        "improvement": {
            "average_score": lora_eval["summary"]["average_score"] - base_eval["summary"]["average_score"],
            "pass_rate": lora_eval["summary"]["pass_rate"] - base_eval["summary"]["pass_rate"],
        },
    }

    out_dir = Path("test_results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "lora_vs_base_results.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "comparison": comparison,
            "base_detailed": base_eval["detailed_results"],
            "lora_detailed": lora_eval["detailed_results"],
        }, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("LORA vs BASE MED42-8B")
    print("=" * 60)
    print(f"Base  avg score: {base_eval['summary']['average_score']:.1f}%  pass rate: {base_eval['summary']['pass_rate']:.1f}%")
    print(f"LoRA  avg score: {lora_eval['summary']['average_score']:.1f}%  pass rate: {lora_eval['summary']['pass_rate']:.1f}%")
    print(f"Delta avg score: {comparison['improvement']['average_score']:+.1f} pts  pass rate: {comparison['improvement']['pass_rate']:+.1f} pts")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
