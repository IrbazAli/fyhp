"""
train_local.py
=============================================================================
Aasaan Khata - Local GPU (RTX 3080 10GB) QLoRA Fine-Tuning Pipeline
=============================================================================
Optimized specifically for 10 GB VRAM:
- 4-bit NF4 base model weights (~5.5 GB).
- Gradient checkpointing + paged_adamw_8bit optimizer.
- LoRA rank r=8, alpha=16 targeting attention & MLP projections.
- Resolution capped to max_pixels: 602112 to eliminate OOM spikes.
- Per-device batch size = 1, gradient accumulation steps = 8.
"""

import torch_patch
import os
import json
import argparse
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


class LocalLedgerDataset(Dataset):
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

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        image = Image.open(image_path).convert("RGB")
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

        # Mask prompt loss
        assistant_marker = "<|im_start|>assistant\n"
        encoded_marker = self.processor.tokenizer.encode(assistant_marker, add_special_tokens=False)
        marker_len = len(encoded_marker)

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
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=151643
    )
    labels_padded = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )

    batch_out = {
        "input_ids": input_ids_padded,
        "labels": labels_padded,
        "attention_mask": input_ids_padded.ne(151643).long()
    }

    if "pixel_values" in batch[0] and batch[0]["pixel_values"] is not None:
        batch_out["pixel_values"] = torch.cat([b["pixel_values"] for b in batch], dim=0)
    if "image_grid_thw" in batch[0] and batch[0]["image_grid_thw"] is not None:
        batch_out["image_grid_thw"] = torch.cat([b["image_grid_thw"] for b in batch], dim=0)

    return batch_out


def run_training():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./qwen2.5-vl-7b")
    parser.add_argument("--train_json", type=str, default="dataset/train.json")
    parser.add_argument("--output_dir", type=str, default="./local_adapter")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    print("=" * 65)
    print("🚀 Starting Local QLoRA Fine-Tuning on RTX 3080 (10 GB VRAM)")
    print(f"Base model: {args.model_path}")
    print(f"Train dataset: {args.train_json}")
    print("=" * 65)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    model = prepare_model_for_kbit_training(model)
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = LocalLedgerDataset(args.train_json, processor)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=1,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        save_total_limit=1,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn
    )

    trainer.train()
    print(f"✅ Local training complete! Adapter saved to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    run_training()
