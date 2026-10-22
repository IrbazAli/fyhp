"""
train_uhwr_qwen.py
=============================================================================
Aasaan Khata - Full Local QLoRA Training Pipeline for Qwen2.5-VL-7B
Trains on the unified NUST UHWR (8,483 lines) + Aasaan Khata Ledger Dataset.
Optimized for NVIDIA GeForce RTX 3080 (10 GB VRAM).
=============================================================================
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch_patch
import json
import argparse
import sys
import time

import torch
from torch.utils.data import Dataset
from PIL import Image

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from qwen_vl_utils import process_vision_info


class UHWRCombinedDataset(Dataset):
    def __init__(self, json_path: str, processor, max_pixels: int = 65536, min_pixels: int = 3136):
        self.processor = processor
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
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
                        "max_pixels": self.max_pixels,
                        "min_pixels": self.min_pixels
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

        # Mask prompt loss (only compute loss on assistant handwritten transcription tokens)
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
            "pixel_values": inputs.get("pixel_values", None),
            "image_grid_thw": inputs.get("image_grid_thw", None)
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
        batch_out["pixel_values"] = torch.cat([b["pixel_values"] for b in batch if b["pixel_values"] is not None], dim=0)
    if "image_grid_thw" in batch[0] and batch[0]["image_grid_thw"] is not None:
        batch_out["image_grid_thw"] = torch.cat([b["image_grid_thw"] for b in batch if b["image_grid_thw"] is not None], dim=0)

    return batch_out


class ProgressLoggerCallback(TrainerCallback):
    def __init__(self):
        self.start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step = state.global_step
            total_steps = state.max_steps
            loss = logs.get("loss", "N/A")
            lr = logs.get("learning_rate", "N/A")
            elapsed = time.time() - self.start_time
            if isinstance(loss, (int, float)):
                print(f"⏱️ Step {step}/{total_steps} | Loss: {loss:.4f} | LR: {lr:.2e} | Elapsed: {elapsed/60:.1f}m", flush=True)


def run_training():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./qwen2.5-vl-7b")
    parser.add_argument("--train_json", type=str, default="dataset/train_combined_uhwr.json")
    parser.add_argument("--output_dir", type=str, default="./local_adapter")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    args = parser.parse_args()

    print("=" * 70)
    print("🚀 FULL HARDWARE TRAINING: Qwen2.5-VL-7B on UHWR + Ledger Dataset")
    print(f"📍 Base Model       : {args.model_path}")
    print(f"📁 Dataset          : {args.train_json}")
    print(f"🎯 Output Directory : {args.output_dir}")
    print(f"⚡ Target Hardware  : NVIDIA GeForce RTX 3080 (10 GB VRAM)")
    print(f"📊 Hyperparameters  : Epochs={args.epochs}, GradAccum={args.grad_accum}, LR={args.lr}")
    print("=" * 70, flush=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    print("Loading base model in 4-bit NF4...")
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    for param in model.parameters():
        param.requires_grad = False
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    print("Configuring LoRA Adapter across Attention & MLP Projection Layers...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"Loading dataset from {args.train_json}...")
    train_dataset = UHWRCombinedDataset(args.train_json, processor, max_pixels=65536, min_pixels=3136)
    print(f"Loaded {len(train_dataset)} training items successfully.")

    total_steps = (len(train_dataset) // args.grad_accum) * args.epochs
    print(f"Total optimization steps: {total_steps} (Effective batch size = {args.grad_accum})")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        save_total_limit=2,
        dataloader_num_workers=2,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        callbacks=[ProgressLoggerCallback()]
    )

    print("\n" + "=" * 70)
    print("🔥 Starting Training Loop now...")
    print("=" * 70, flush=True)

    trainer.train()

    print("\n" + "=" * 70)
    print(f"🎉 Training complete! Saving final LoRA adapter to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"✅ Adapter weights and tokenizer saved successfully in {args.output_dir}")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    run_training()
