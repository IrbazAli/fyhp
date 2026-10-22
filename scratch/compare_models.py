import sys
sys.path.insert(0, "/home/irbazali/Desktop/FYP")
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch_patch
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)
from peft import PeftModel
from qwen_vl_utils import process_vision_info

MODEL_PATH = "/home/irbazali/Desktop/FYP/qwen2.5-vl-7b"
ADAPTER_PATH = "./local_adapter"
IMAGE_PATH = "/home/irbazali/Desktop/FYP/dataset/1.jpeg"

def run_inference(model, processor, image_path, prompt_text=""):
    content = [
        {
            "type": "image",
            "image": image_path,
            "max_pixels": 1920000
        }
    ]
    if prompt_text:
        content.append({
            "type": "text",
            "text": prompt_text
        })
    
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to("cuda")
    
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1536,
            do_sample=False,
            repetition_penalty=1.05
        )
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    return output_text.strip()

def main():
    print("=" * 70)
    print("1. Loading Base Qwen2.5-VL-7B into 4-bit...")
    print("=" * 70)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    print("Base model loaded successfully.")

    results = {}

    # Test 1: Zero instruction (pure image input)
    print("\n[Running Base Model - Zero Instruction / Pure Image]...")
    res_base_no_prompt = run_inference(base_model, processor, IMAGE_PATH, "")
    results["base_no_prompt"] = res_base_no_prompt

    # Test 2: Minimal neutral prompt ("Transcribe this image.")
    print("\n[Running Base Model - Minimal Prompt 'Transcribe this image.']...")
    res_base_minimal = run_inference(base_model, processor, IMAGE_PATH, "Transcribe this image.")
    results["base_minimal"] = res_base_minimal

    print("\n" + "=" * 70)
    print("2. Applying Fine-Tuned LoRA Adapter from ./local_adapter...")
    print("=" * 70)
    trained_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    print("LoRA adapter loaded.")

    # Test 1 with trained model
    print("\n[Running Trained Model - Zero Instruction / Pure Image]...")
    res_trained_no_prompt = run_inference(trained_model, processor, IMAGE_PATH, "")
    results["trained_no_prompt"] = res_trained_no_prompt

    # Test 2 with trained model
    print("\n[Running Trained Model - Minimal Prompt 'Transcribe this image.']...")
    res_trained_minimal = run_inference(trained_model, processor, IMAGE_PATH, "Transcribe this image.")
    results["trained_minimal"] = res_trained_minimal

    print("\n" + "=" * 70)
    print("COMPILATION OF RAW OUTPUTS (WITHOUT ANY IMPROVEMENTS / POST-PROCESSING)")
    print("=" * 70)
    
    print("\n>>> TEST A: Zero Instruction (Input: Image Only)")
    print("--- BASE MODEL OUTPUT ---")
    print(results["base_no_prompt"])
    print("\n--- TRAINED MODEL (LoRA) OUTPUT ---")
    print(results["trained_no_prompt"])
    
    print("\n" + "-" * 70)
    print(">>> TEST B: Minimal Prompt ('Transcribe this image.')")
    print("--- BASE MODEL OUTPUT ---")
    print(results["base_minimal"])
    print("\n--- TRAINED MODEL (LoRA) OUTPUT ---")
    print(results["trained_minimal"])
    print("=" * 70)

    # Save to a json for inspection
    with open("compare_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nResults saved to compare_results.json")

if __name__ == "__main__":
    main()
