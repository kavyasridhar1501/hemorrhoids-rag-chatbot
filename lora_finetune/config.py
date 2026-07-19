from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class LoRAConfig:
    base_model: str = "m42-health/Llama3-Med42-8B"
    output_dir: str = "lora_finetune/adapter"

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    load_in_4bit: bool = True
    # T4 (free-tier Colab) is Turing architecture and has no bf16 support -
    # use float16 there. Switch to "bfloat16" on Ampere+ (A100/3090/4090/L4).
    compute_dtype: str = "float16"
    gradient_checkpointing: bool = True

    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_seq_length: int = 512
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_strategy: str = "epoch"
    # In-training eval switches the model between train/eval mode each epoch,
    # which adds memory churn on top of an already-tight T4 budget - skip it
    # here and rely on evaluate_lora.py for the real base-vs-LoRA comparison.
    eval_strategy: str = "no"

    train_file: str = "lora_finetune/data/train.jsonl"
    val_file: str = "lora_finetune/data/val.jsonl"
    seed: int = 42

    @property
    def torch_dtype(self) -> torch.dtype:
        return getattr(torch, self.compute_dtype)


@dataclass
class ExtractionLoRAConfig(LoRAConfig):
    """Config for the scoped red-flag/triage JSON-extraction task (see
    extraction_schema.py) instead of free-text chatbot-answer imitation.

    Completions are a few dozen tokens of compact JSON rather than a full
    paragraph, so max_seq_length shrinks and the effective batch can grow -
    both make each optimizer step cheaper, so more of them fit in a T4
    session than the free-text task's ~9 total steps did.
    """
    output_dir: str = "lora_finetune/adapter_extraction"
    max_seq_length: int = 256
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 6

    train_file: str = "lora_finetune/data/extraction_train.jsonl"
    val_file: str = "lora_finetune/data/extraction_val.jsonl"
