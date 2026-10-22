"""Continue the existing Qwen LoRA on handwritten lines, into a separate run."""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import random
import time
from pathlib import Path

import torch_patch
import torch
from PIL import Image
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

from train_local import LocalLedgerDataset, collate_fn


class CheckedDataset(LocalLedgerDataset):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        labels = sample["labels"]
        if labels[0].item() != -100 or not labels.ne(-100).any():
            raise ValueError(f"Assistant-only masking failed for record {index}")
        if len(labels) > 2048:
            raise ValueError(f"Unexpectedly long sequence ({len(labels)}) at record {index}")
        return sample


class Progress(TrainerCallback):
    def __init__(self, output):
        self.output = output

    def on_log(self, args, state, control, logs=None, **kwargs):
        row = {"time": time.time(), "step": state.global_step, "max_steps": state.max_steps,
               "epoch": state.epoch, **(logs or {})}
        with (self.output / "progress.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        (self.output / "status.json").write_text(json.dumps({"status": "running", **row}, indent=2))


def distance(reference, prediction):
    previous = list(range(len(prediction) + 1))
    for i, a in enumerate(reference, 1):
        current = [i]
        for j, b in enumerate(prediction, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j-1] + (a != b)))
        previous = current
    return previous[-1]


def evaluate_text(model, processor, records, max_pixels, output):
    model.eval()
    results = []
    for index, record in enumerate(records):
        with Image.open(record["image"]) as image:
            image = image.convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image, "max_pixels": max_pixels},
            {"type": "text", "text": record["conversations"][0]["content"].replace("<image>\n", "").strip()},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                                       repetition_penalty=1.0, use_cache=True)
        prediction = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)[0]
        reference = record["conversations"][1]["content"]
        results.append({"image": record["image"], "reference": reference, "prediction": prediction,
                        "character_edits": distance(reference, prediction),
                        "word_edits": distance(reference.split(), prediction.split()),
                        "hit_token_limit": generated.shape[1] - inputs.input_ids.shape[1] >= 256})
        print(f"Recognition evaluation {index+1}/{len(records)}", flush=True)
    metrics = {
        "samples": len(results),
        "cer": sum(r["character_edits"] for r in results) / max(1, sum(len(r["reference"]) for r in results)),
        "wer": sum(r["word_edits"] for r in results) / max(1, sum(len(r["reference"].split()) for r in results)),
        "exact_match": sum(r["reference"] == r["prediction"] for r in results) / max(1, len(results)),
        "scope": "Fixed validation subset, not the held-out test set; raw predictions, no character normalization.",
    }
    output.write_text(json.dumps({"metrics": metrics, "predictions": results}, ensure_ascii=False, indent=2))
    print(json.dumps(metrics), flush=True)
    torch.cuda.empty_cache()
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="dataset/nust_uhwr")
    parser.add_argument("--model", default="qwen2.5-vl-7b")
    parser.add_argument("--adapter", default="local_adapter")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-pixels", type=int, default=100352)
    parser.add_argument("--recognition-samples", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output == Path(args.adapter).resolve():
        raise ValueError("Use a separate output directory to preserve the existing adapter")
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError("Output already contains files; use a new directory or --resume")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(vars(args), indent=2))
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; run with access to the RTX 3080")
        set_seed(42)
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model, local_files_only=True, torch_dtype=torch.bfloat16, device_map={"": 0},
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True))
        for parameter in base.parameters():
            parameter.requires_grad = False
        base.enable_input_require_grads()
        base.config.use_cache = False
        model = PeftModel.from_pretrained(base, args.adapter, is_trainable=True, local_files_only=True)
        model.print_trainable_parameters()
        train = CheckedDataset(str(Path(args.data) / "train.json"), processor, args.max_pixels)
        val = CheckedDataset(str(Path(args.data) / "val.json"), processor, args.max_pixels)
        # Exercise the largest line before scheduling a multi-hour run.
        longest = max(range(len(train)), key=lambda i: len(train.data[i]["conversations"][1]["content"]))
        for index in [0, longest]:
            sample = train[index]
            print(f"Data check {index}: {len(sample['labels'])} tokens, "
                  f"{sample['labels'].ne(-100).sum().item()} supervised", flush=True)
        indices = random.Random(42).sample(range(len(val)), min(args.recognition_samples, len(val)))
        recognition_records = [val.data[i] for i in indices]
        if recognition_records and not args.resume:
            evaluate_text(model, processor, recognition_records, args.max_pixels, output / "before.json")
        model.config.use_cache = False
        arguments = TrainingArguments(
            output_dir=str(output), num_train_epochs=args.epochs, max_steps=args.max_steps,
            learning_rate=args.lr, per_device_train_batch_size=1, per_device_eval_batch_size=1,
            gradient_accumulation_steps=8, lr_scheduler_type="cosine", warmup_steps=20,
            bf16=True, gradient_checkpointing=True, optim="paged_adamw_8bit",
            logging_steps=10, logging_first_step=True, disable_tqdm=True,
            eval_strategy="steps", eval_steps=250, prediction_loss_only=True,
            save_strategy="steps", save_steps=250, save_total_limit=2,
            load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
            report_to="none", remove_unused_columns=False, seed=42)
        trainer = Trainer(model=model, args=arguments, train_dataset=train, eval_dataset=val,
                          data_collator=collate_fn, processing_class=processor, callbacks=[Progress(output)])
        result = trainer.train(resume_from_checkpoint=args.resume)
        final_loss = trainer.evaluate()
        model.save_pretrained(output / "adapter")
        processor.save_pretrained(output / "adapter")
        trainer.save_state()
        trainer.save_metrics("train", result.metrics)
        trainer.save_metrics("eval", final_loss)
        after = evaluate_text(model, processor, recognition_records, args.max_pixels, output / "after.json") if recognition_records else {}
        (output / "status.json").write_text(json.dumps({"status": "completed", "train": result.metrics,
            "validation": final_loss, "recognition": after,
            "best_checkpoint": trainer.state.best_model_checkpoint, "adapter": str(output / "adapter")}, indent=2))
    except BaseException as error:
        (output / "status.json").write_text(json.dumps({"status": "failed", "error": str(error)}, indent=2))
        raise


if __name__ == "__main__":
    main()
