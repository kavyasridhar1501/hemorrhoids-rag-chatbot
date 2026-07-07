"""
Merges the trained LoRA adapter into the base Med42-8B weights, producing a
standalone model directory that can be converted to GGUF (e.g. via
llama.cpp's convert script) to serve through Ollama like the base model.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from config import LoRAConfig


def main():
    cfg = LoRAConfig()

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, torch_dtype=cfg.torch_dtype, device_map="cpu"
    )
    model = PeftModel.from_pretrained(base_model, cfg.output_dir)
    merged = model.merge_and_unload()

    out_dir = "lora_finetune/merged_model"
    merged.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Merged model saved to {out_dir}")


if __name__ == "__main__":
    main()
