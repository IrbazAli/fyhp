"""
train_kaggle_qwen_vl.py
=============================================================================
Aasaan Khata - Qwen2.5-VL-7B QLoRA Fine-Tuning Script (Kaggle / Cloud GPUs)
=============================================================================
Fine-tunes Qwen2.5-VL-7B on raw handwritten Urdu wholesale ledger images
and their ground-truth validated JSON schema.

Features:
- 4-bit NF4 Quantization (QLoRA) to train on Kaggle 16GB GPUs (T4 / P100).
- Memory-efficient gradient checkpointing.
- Weights & Biases (W&B) and Hugging Face Hub checkpoint synchronization.
- Whole-document end-to-end instruction tuning.
"""

import torch_patch
import os
import json
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List

import torch
from torch.utils.data import Dataset
from PIL import Image

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from qwen_vl_utils import process_vision_info


class LedgerDataset(Dataset):
    """
    Dataset loader for Qwen conversational JSON schema.
    Reads images and formats chat templates.
    """
    def __init__(self, json_path: str, processor, max_pixels: int = 602112):
        self.processor = processor
        self.max_pixels = max_pixels
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item["image"]
        conversations = item["conversations"]

        # Ensure image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        image = Image.open(image_path).convert("RGB")

        # Format user message with image payload and assistant response
        user_content = conversations[0]["content"].replace("<image>\n", "").strip()
        assistant_content = conversations[1]["content"]

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                        "max_pixels": self.max_pixels
                    },
                    {
                        "type": "text",
                        "text": user_content
                    }
                ]
            },
            {
                "role": "assistant",
                "content": assistant_content
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"][0]
        labels = input_ids.clone()

        # Mask user prompt tokens in loss so model only trains on JSON prediction
        # Find assistant start token
        assistant_marker = "<|im_start|>assistant\n"
        encoded_marker = self.processor.tokenizer.encode(assistant_marker, add_special_tokens=False)
        marker_len = len(encoded_marker)

        # Locate marker in input_ids
        for i in range(len(input_ids) - marker_len):
            if input_ids[i:i+marker_len].tolist() == encoded_marker:
                labels[:i+marker_len] = -100
                break

        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": inputs.get("pixel_values", [None])[0],
            "image_grid_thw": inputs.get("image_grid_thw", [None])[0]
        }


def collate_fn(batch):
    """Custom collator for dynamic length padding."""
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=151643  # Qwen pad token
    )
    labels_padded = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )

    batch_out = {
        "input_ids": input_ids_padded,
        "labels": labels_padded,
        "attention_mask": input_ids_padded.ne(151643).long()
    }

    # Stack vision tensors if present
    if "pixel_values" in batch[0] and batch[0]["pixel_values"] is not None:
        batch_out["pixel_values"] = torch.cat([b["pixel_values"] for b in batch], dim=0)
    if "image_grid_thw" in batch[0] and batch[0]["image_grid_thw"] is not None:
        batch_out["image_grid_thw"] = torch.cat([b["image_grid_thw"] for b in batch], dim=0)

    return batch_out


def train(args):
    print("=" * 60)
    print("Aasaan Khata - Qwen2.5-VL-7B QLoRA Training Pipeline")
    print(f"Base Model: {args.model_name_or_path}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 60)

    # 4-bit Quantization Config for Kaggle 16GB GPUs
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True
    )

    processor = AutoProcessor.from_pretrained(args.model_name_or_path)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )

    model = prepare_model_for_kbit_training(model)
    model.enable_input_require_grads()

    # LoRA Adapter Configuration targeting attention and projection layers
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Datasets
    train_dataset = LedgerDataset(args.train_json, processor, max_pixels=args.max_pixels)
    val_dataset = LedgerDataset(args.val_json, processor, max_pixels=args.max_pixels) if args.val_json else None

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_strategy="epoch",
        evaluation_strategy="epoch" if val_dataset else "no",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="wandb" if args.use_wandb else "none",
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id if args.push_to_hub else None,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )

    print("Beginning fine-tuning...")
    trainer.train()

    print(f"Saving fine-tuned adapter to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("Training successfully completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--train_json", type=str, default="dataset/train.json")
    parser.add_argument("--val_json", type=str, default="dataset/val.json")
    parser.add_argument("--output_dir", type=str, default="./aasaan_khata_adapter")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_pixels", type=int, default=602112)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default="Aasaan-Khata/qwen2.5-vl-7b-lora")
    args = parser.parse_args()

    train(args)
