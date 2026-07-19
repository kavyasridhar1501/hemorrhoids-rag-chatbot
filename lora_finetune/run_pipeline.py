"""
End-to-end LoRA fine-tuning pipeline runner.

Generates synthetic training questions, builds the RAG vectorstore (if
missing), distills training data from the Claude + RAG chatbot, fine-tunes
Med42-8B with LoRA, and evaluates it against the base model - the same
sequence of steps documented in lora_finetune/README.md, run as one command
instead of one script at a time.

Usage (from the repo root, after cloning):
    pip install -r requirements.txt -r lora_finetune/requirements.txt
    python lora_finetune/run_pipeline.py

Requires ANTHROPIC_API_KEY and HF_TOKEN in the environment (.env or
exported) and a CUDA GPU (T4 16GB minimum) for the train/evaluate steps.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run(cmd, label):
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        print(f"\nStep failed: {label}")
        sys.exit(result.returncode)


def build_vectorstore_if_missing():
    if (REPO_ROOT / "faiss_index" / "index.faiss").exists():
        print("faiss_index/ already exists, skipping vectorstore build.")
        return

    run(["python", "medical_scraper.py"], "Scraping trusted medical sources")

    documents_dir = REPO_ROOT / "documents"
    documents_dir.mkdir(exist_ok=True)
    for txt_file in (REPO_ROOT / "medical_articles").glob("*.txt"):
        shutil.copy(txt_file, documents_dir)

    n_docs = len(list(documents_dir.glob("*.txt")))
    print(f"{n_docs} articles copied into documents/")
    if n_docs == 0:
        print("Scraper collected 0 articles - check medical_articles/blocked_urls.txt")
        sys.exit(1)

    run(["python", "rag_setup.py"], "Building the RAG vectorstore")


def run_extraction_task():
    """Scoped red-flag/triage JSON-extraction task - no RAG/vectorstore
    dependency at all, since it's a direct classification task."""
    run(["python", "lora_finetune/generate_extraction_questions.py"], "Generating class-balanced red-flag messages")
    run(["python", "lora_finetune/prepare_extraction_dataset.py"], "Labeling + building extraction train/val data")
    run(["python", "lora_finetune/train_lora.py", "--task", "extraction"], "Fine-tuning Med42-8B on red-flag extraction")
    run(["python", "lora_finetune/evaluate_extraction.py"], "Evaluating base vs. LoRA on red-flag extraction")
    print(
        "\nPipeline complete.\n"
        "  Adapter: lora_finetune/adapter_extraction/\n"
        "  Results: test_results/lora_vs_base_extraction_results.json\n"
        "\nHand-review lora_finetune/data/extraction_val_for_review.jsonl "
        "before trusting those numbers - it's the eval set's labels in a "
        "readable format.\n"
        "\nIf running on an ephemeral environment (e.g. Colab), copy "
        "lora_finetune/adapter_extraction/ somewhere durable now - it will "
        "not survive a runtime restart."
    )


def run_chat_task():
    run(["python", "lora_finetune/generate_questions.py"], "Generating synthetic training questions")
    build_vectorstore_if_missing()
    run(["python", "lora_finetune/prepare_dataset.py"], "Building LoRA training data")
    run(["python", "lora_finetune/train_lora.py"], "Fine-tuning Med42-8B with LoRA")
    run(["python", "lora_finetune/evaluate_lora.py"], "Evaluating base vs. LoRA")
    print(
        "\nPipeline complete.\n"
        "  Adapter: lora_finetune/adapter/\n"
        "  Results: test_results/lora_vs_base_results.json\n"
        "\nIf running on an ephemeral environment (e.g. Colab), copy "
        "lora_finetune/adapter/ somewhere durable now - it will not "
        "survive a runtime restart."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", choices=["chat", "extraction"], default="chat",
        help="chat = free-text answer imitation (original pipeline, default); "
             "extraction = scoped red-flag/triage JSON classification",
    )
    args = parser.parse_args()
    if args.task == "extraction":
        run_extraction_task()
    else:
        run_chat_task()


if __name__ == "__main__":
    main()
