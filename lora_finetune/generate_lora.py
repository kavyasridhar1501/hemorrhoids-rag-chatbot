"""Shared helper for generating responses with base or LoRA-adapted Med42-8B."""
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from patient_chatbot import SYSTEM_PROMPT
from config import LoRAConfig


class Med42Generator:
    def __init__(self, adapter_path: str = None, cfg: LoRAConfig = None):
        self.cfg = cfg or LoRAConfig()

        quant_config = BitsAndBytesConfig(
            load_in_4bit=self.cfg.load_in_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.base_model)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.cfg.base_model,
            quantization_config=quant_config if self.cfg.load_in_4bit else None,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.model = PeftModel.from_pretrained(base_model, adapter_path) if adapter_path else base_model
        self.model.eval()

    @torch.inference_mode()
    def generate(self, question: str, max_new_tokens: int = 512) -> str:
        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return text.strip()
