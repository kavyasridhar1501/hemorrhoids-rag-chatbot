"""
LoRA fine-tuning of Med42-8B on the distilled patient Q&A pairs from
prepare_dataset.py. Requires a CUDA GPU (bitsandbytes 4-bit quantization).
"""
import json
import os
from pathlib import Path
from typing import List

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig as PeftLoraConfig, get_peft_model, prepare_model_for_kbit_training

from config import LoRAConfig


def load_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def tokenize_example(example, tokenizer, max_length):
    prompt_ids = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(
        example["completion"] + tokenizer.eos_token, add_special_tokens=False
    )["input_ids"]

    input_ids = (prompt_ids + completion_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + completion_ids)[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def build_dataset(path, tokenizer, max_length):
    rows = load_jsonl(path)
    dataset = Dataset.from_list(rows)
    return dataset.map(
        lambda ex: tokenize_example(ex, tokenizer, max_length),
        remove_columns=dataset.column_names,
    )


def main():
    cfg = LoRAConfig()

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=cfg.torch_dtype,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=quant_config if cfg.load_in_4bit else None,
        device_map="auto",
        torch_dtype=cfg.torch_dtype,
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg.gradient_checkpointing
    )

    peft_config = PeftLoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_dataset = build_dataset(cfg.train_file, tokenizer, cfg.max_seq_length)
    val_dataset = (
        build_dataset(cfg.val_file, tokenizer, cfg.max_seq_length)
        if Path(cfg.val_file).exists()
        else None
    )

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        eval_strategy=cfg.eval_strategy if val_dataset is not None else "no",
        fp16=cfg.compute_dtype == "float16",
        bf16=cfg.compute_dtype == "bfloat16",
        report_to=[],
        seed=cfg.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )

    trainer.train()
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"Adapter saved to {cfg.output_dir}")


if __name__ == "__main__":
    main()
