"""
inference_engine.py
=============================================================================
Aasaan Khata - Qwen2.5-VL-7B Offline Production Inference Engine
=============================================================================
Manages memory-optimized 4-bit offline inference for handwritten Urdu
wholesale ledgers with resolution capping, JSON parsing, and math validation.
"""

import os
import json
import re
from typing import Dict, Any, Optional, Tuple, Union
from PIL import Image

import torch_patch
import torch

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from math_engine import MathValidator, DocumentValidationResult


SYSTEM_PROMPT = """You are an expert handwritten Urdu (Nastaliq) wholesale ledger transcription system (Aasaan Khata).
Your task is to transcribe raw wholesale transaction ledgers into clean, strictly valid JSON.

Wholesale Rules:
1. Extract the transaction date (تاریخ) and customer/merchant name (نام / کھاتہ دار).
2. For each line item, extract:
   - "item": Commodity name in Urdu (e.g., چاول, چینی, گندم, دال چنا, گھی, تیل).
   - "quantity": Numeric quantity (e.g., 10, 50).
   - "unit": Unit if mentioned (e.g., من, بوری, کلو, پیٹی, نگ, or empty string).
   - "rate": Unit price/rate (نرخ / ریٹ) without currency symbols or slashes.
   - "line_total": Total calculated for that line (رقم / کل).
3. Extract any additional expenses or freight (کرایہ / مزدوری), discounts (چھوٹ / کٹوتی).
4. Extract the grand total (میزان / کل رقم).
5. Output ONLY valid JSON matching this schema:
{
  "date": "YYYY-MM-DD or DD/MM/YYYY",
  "khata_holder": "نام",
  "items": [
    {
      "item": "چاول",
      "quantity": 10,
      "unit": "من",
      "rate": 3500,
      "line_total": 35000
    }
  ],
  "expenses": 0,
  "discount": 0,
  "grand_total": 35000
}
Do not include any conversational filler. Return only valid JSON enclosed in ```json ... ``` code blocks.
"""


def extract_json_from_llm_response(text: str) -> Optional[Dict[str, Any]]:
    """
    Extracts and parses JSON object from LLM response text,
    handling markdown fences, unquoted trailing commas, and formatting noise.
    """
    if not text:
        return None
    
    # 1. Try markdown code block
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    candidate_str = json_match.group(1).strip() if json_match else text.strip()

    # 2. Extract outermost { ... }
    first_brace = candidate_str.find('{')
    last_brace = candidate_str.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate_str = candidate_str[first_brace:last_brace + 1]

    # 3. Direct JSON load attempt
    try:
        return json.loads(candidate_str)
    except Exception:
        pass

    # 4. Cleanup trailing commas before } or ]
    cleaned = re.sub(r',\s*([}\]])', r'\1', candidate_str)
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    return None


class LedgerVLMEngine:
    """
    Offline Qwen2.5-VL-7B Inference Engine with 4-bit BitsAndBytes quantization.
    """
    def __init__(
        self,
        model_path: str = "./qwen2.5-vl-7b",
        max_pixels: int = 602112,  # ~768x784 cap to prevent VRAM spikes
        max_new_tokens: int = 768,
        device: str = "cuda"
    ):
        self.model_path = model_path
        self.max_pixels = max_pixels
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.model = None
        self.processor = None
        self.math_validator = MathValidator()

    def load_model(self):
        """Loads model into 4-bit VRAM if not already loaded."""
        if self.model is not None and self.processor is not None:
            return

        print(f"Loading Qwen2.5-VL-7B into 4-bit memory from {self.model_path}...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        print(f"Model successfully loaded. VRAM allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")

    def process_image(
        self,
        image_input: Union[str, Image.Image],
        custom_prompt: Optional[str] = None
    ) -> Tuple[str, Optional[Dict[str, Any]], DocumentValidationResult]:
        """
        Runs full inference on an image and applies the deterministic math validation layer.
        
        Args:
            image_input: Path to image file or PIL Image object.
            custom_prompt: Optional override for the prompt.
            
        Returns:
            Tuple of:
            - raw_output_text (str)
            - parsed_json (dict or None)
            - math_validation (DocumentValidationResult)
        """
        self.load_model()

        # Handle PIL image vs file path
        if isinstance(image_input, Image.Image):
            # Save temporarily or process directly
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                image_input.convert("RGB").save(tmp.name, "JPEG")
                img_path = tmp.name
                is_tmp = True
        else:
            img_path = str(image_input)
            is_tmp = False

        prompt_text = custom_prompt if custom_prompt else (
            "Transcribe this handwritten Urdu ledger into structured JSON format. "
            "Extract date, customer name, line items (item name, quantity, unit, rate, line total), and grand total."
        )

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img_path,
                        "max_pixels": self.max_pixels
                    },
                    {
                        "type": "text",
                        "text": prompt_text
                    }
                ],
            }
        ]

        try:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False  # Deterministic greedy decoding for maximum OCR fidelity
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            raw_output = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            # Parse JSON
            parsed_json = extract_json_from_llm_response(raw_output)

            # Apply Deterministic Math Validation Layer
            if parsed_json:
                math_result = self.math_validator.validate_document(parsed_json)
            else:
                math_result = DocumentValidationResult(
                    is_valid=False,
                    status="JSON_PARSE_ERROR",
                    requires_hitl=True,
                    audit_summary="Model generated unparseable response; human review required."
                )

            return raw_output, parsed_json, math_result

        finally:
            if is_tmp and os.path.exists(img_path):
                os.remove(img_path)
