#!/usr/bin/env python
"""06_train_lora_fp16.py — LoRA fine-tuning in FP16 (no quantization) for EXAONE / LLaMA / Qwen3
FP16 training avoids bitsandbytes CUDA ops issues on TITAN RTX (cc7.5).
Model: 8B params x 2 bytes = 16 GB, fits in 24 GB VRAM with gradient checkpointing.
"""

import argparse, json, os, sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, SFTConfig

MODEL_MAP = {
    "exaone": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
    "llama":  "meta-llama/Llama-3.1-8B-Instruct",
    "qwen3":  "Qwen/Qwen3-8B",
}
LOCAL_MAP = {
    "exaone": "/home/user/models/EXAONE-3.5-7.8B-Instruct",
    "llama":  "/home/user/models/Llama-3.1-8B-Instruct",
    "qwen3":  "/home/user/models/Qwen3-8B",
}

DATA_PATH   = Path("data/ft_dataset.json")
CKPT_BASE   = Path("checkpoints")

LORA_R      = 16
LORA_ALPHA  = 32
LORA_DROP   = 0.05
TARGET_MODS = ["q_proj", "v_proj"]

BATCH       = 1
GRAD_ACCUM  = 16
LR          = 2e-4
EPOCHS      = 3
SEED        = 42
MAX_SEQ_LEN = 512

def get_model_path(key):
    local = LOCAL_MAP[key]
    if Path(local).exists():
        print(f"Using local model: {local}")
        return local
    print(f"Local not found, using HF: {MODEL_MAP[key]}")
    return MODEL_MAP[key]

def load_dataset_from_json(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)

def format_messages(example, tokenizer):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["exaone", "llama", "qwen3"])
    args = parser.parse_args()

    model_key  = args.model
    model_path = get_model_path(model_key)
    out_dir    = CKPT_BASE / f"{model_key}-ft"
    merged_dir = CKPT_BASE / f"{model_key}-ft-merged"

    print(f"\n=== LoRA FP16 Training: {model_key} ===")
    print(f"Output: {out_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # FP16: 8B params x 2 bytes = 16 GB fits in 24 GB TITAN RTX
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": 0},
        trust_remote_code=True,
        dtype=torch.float16,
    )
    model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODS,
        lora_dropout=LORA_DROP,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    ds = load_dataset_from_json(DATA_PATH)
    ds = ds.map(lambda ex: format_messages(ex, tokenizer), remove_columns=["messages"])
    ds = ds.train_test_split(test_size=0.05, seed=SEED)
    print(f"Train: {len(ds['train'])}, Val: {len(ds['test'])}")

    sft_config = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        fp16=True,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch",
        seed=SEED,
        report_to="none",
        dataloader_num_workers=0,
        optim="adamw_torch",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        dataset_text_field="text",
        max_length=MAX_SEQ_LEN,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(out_dir))
    print(f"Adapter saved: {out_dir}")

    print("Merging adapter...")
    del model
    torch.cuda.empty_cache()

    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": 0},
        trust_remote_code=True,
        dtype=torch.float16,
    )
    merged = PeftModel.from_pretrained(base, str(out_dir))
    merged = merged.merge_and_unload()
    merged.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    print(f"Merged model saved: {merged_dir}")


if __name__ == "__main__":
    main()
