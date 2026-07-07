from dataclasses import dataclass, field
from typing import List


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

    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 1024
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"

    train_file: str = "lora_finetune/data/train.jsonl"
    val_file: str = "lora_finetune/data/val.jsonl"
    seed: int = 42
