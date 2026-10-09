import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch

# --- Patch for PyTorch 2.5 compatibility ---
if not hasattr(torch, "accelerator"):
    class DummyAccelerator:
        @staticmethod
        def current_accelerator():
            return torch.device("cuda")
    torch.accelerator = DummyAccelerator
# -------------------------------------------

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# Point this to the local directory where you downloaded the 16.6 GB model
model_path = "./qwen2.5-vl-7b"

# 1. Configure 4-bit quantization to fit the 10GB VRAM bottleneck
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("Loading Qwen2.5-VL-7B into 4-bit memory...")
# 2. Load the model and processor
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)

# 3. Format the VLM prompt to dynamically detect columns and field values
prompt = (
    "Extract all information from this ledger image into structured JSON format.\n\n"
    "Instructions for dynamic column & field extraction:\n"
    "1. Dynamically identify whatever columns/fields actually exist in the image (e.g., date, description, particulars, quantity, rate, amount, debit, credit, balance, etc.) rather than assuming a fixed template.\n"
    "2. For each entry/row, dynamically adjust the keys and field values to match what is written in that specific line.\n"
    "3. Do not include empty placeholder keys if that field is not present in the entry.\n"
    "4. Return a JSON structure containing:\n"
    "   - 'detected_fields': A list of detected column/field names found in the document.\n"
    "   - 'entries': A list of row objects with dynamic key-value pairs representing each line.\n\n"
    "Output valid JSON only."
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image", 
                "image": "dataset/2.jpeg",
                "max_pixels": 1280 * 28 * 28
            },
            {
                "type": "text", 
                "text": prompt
            }
        ],
    }
]

# 4. Prepare inputs using Qwen's vision utility
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to("cuda")

print("Processing image.jpg...")
# 5. Generate the output text
with torch.inference_mode():
    generated_ids = model.generate(**inputs, max_new_tokens=1024)

# Trim the input prompt tokens from the generated output
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print("\n--- Extracted JSON Output ---")
print(output_text[0])
