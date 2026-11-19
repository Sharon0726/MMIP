"""Experiment B: supervised LoRA tuning of the text routing node on a remote GPU."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from core import SYSTEM_PROMPT, router_input
from models import TEXT_MODEL

ROOT = Path(__file__).resolve().parent


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def train(output: Path, epochs: int = 3, max_length: int = 1024) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    if not torch.cuda.is_available():
        raise RuntimeError("訓練必須在 Colab 或遠端 CUDA GPU 上執行；本機不會訓練")
    rows = load_rows(ROOT / "data" / "train.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    examples = []
    for row in rows:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": router_input(row["turns"], row.get("image_note", ""))},
        ]
        prompt_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        full_ids = tokenizer.apply_chat_template(
            messages + [{
                "role": "assistant",
                "content": json.dumps(row["target"], ensure_ascii=False, separators=(",", ":")),
            }],
            tokenize=True, add_generation_prompt=False,
        )
        if len(full_ids) > max_length:
            raise ValueError(f"{row['id']} exceeds max_length={max_length}: {len(full_ids)}")
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        examples.append({"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels})

    class RouteDataset(Dataset):
        def __len__(self):
            return len(examples)

        def __getitem__(self, index):
            return examples[index]

    def collate(batch):
        size = max(len(x["input_ids"]) for x in batch)
        return {
            "input_ids": torch.tensor([
                x["input_ids"] + [tokenizer.pad_token_id] * (size - len(x["input_ids"]))
                for x in batch
            ]),
            "attention_mask": torch.tensor([
                x["attention_mask"] + [0] * (size - len(x["attention_mask"]))
                for x in batch
            ]),
            "labels": torch.tensor([
                x["labels"] + [-100] * (size - len(x["labels"]))
                for x in batch
            ]),
        }

    model = AutoModelForCausalLM.from_pretrained(
        TEXT_MODEL, torch_dtype=torch.float16, attn_implementation="sdpa"
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    model.print_trainable_parameters()
    args = TrainingArguments(
        output_dir=str(output / "trainer_temp"),
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=20261117,
    )
    start = time.perf_counter()
    trainer = Trainer(model=model, args=args, train_dataset=RouteDataset(), data_collator=collate)
    result = trainer.train()
    elapsed = time.perf_counter() - start
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))
    metadata = {
        "model": TEXT_MODEL, "train_examples": len(rows),
        "epochs": epochs, "steps": result.global_step,
        "train_loss": result.training_loss,
        "seconds": elapsed,
        "gpu": torch.cuda.get_device_name(0),
        "note": "SFT only; no RLHF/DPO claim",
    }
    (output / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "router_lora"))
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(train(Path(args.output), args.epochs), ensure_ascii=False, indent=2))
