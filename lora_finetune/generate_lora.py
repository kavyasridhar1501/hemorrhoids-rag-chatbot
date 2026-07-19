"""Shared helper for generating responses with base or LoRA-adapted Med42-8B."""
import gc
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from patient_chatbot import SYSTEM_PROMPT
from config import LoRAConfig


class Med42Generator:
    def __init__(self, adapter_path: str = None, cfg: LoRAConfig = None, system_prompt: str = None):
        self.cfg = cfg or LoRAConfig()
        self.system_prompt = system_prompt or SYSTEM_PROMPT

        quant_config = BitsAndBytesConfig(
            load_in_4bit=self.cfg.load_in_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=self.cfg.torch_dtype,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.base_model)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.cfg.base_model,
            quantization_config=quant_config if self.cfg.load_in_4bit else None,
            device_map="auto",
            torch_dtype=self.cfg.torch_dtype,
        )
        self.model = PeftModel.from_pretrained(base_model, adapter_path) if adapter_path else base_model
        self.model.eval()

    def unload(self):
        """Release GPU memory - call before loading another Med42Generator in the same process."""
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def generate(self, question: str, max_new_tokens: int = 512) -> str:
        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.system_prompt},
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
