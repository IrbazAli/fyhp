"""
qwen_row_pipeline/prompts.py
=============================================================================
Prompt templates and schemas for all pipeline stages:
  - Stage A: Whole-page row proposals
  - Stage B: Two-row context refinement
  - Stage D: Faithful row transcription
  - Stage E: Field assignment
=============================================================================
"""

import json
from typing import Dict, Any, List


STAGE_A_SYSTEM_RULES = (
    "You are locating ledger entries for image cropping.\n"
    "Return only JSON matching the supplied schema.\n"
    "Coordinates refer to the supplied image of width {W} and height {H}.\n"
    "Locate each entry's visible writing, including descending strokes and dots.\n"
    "COORDINATE RULES:\n"
    "- 'top' and 'bottom' MUST be numeric vertical pixel Y-coordinates (integers between 0 and {H}).\n"
    "- NEVER output monetary amounts, numbers written on the page, or trailing dashes (e.g. do NOT write '281-', '50000-', or rupee amounts) into 'top' or 'bottom'.\n"
    "- 'top' is the vertical pixel position where the row begins.\n"
    "- 'bottom' is the vertical pixel position where the row ends.\n"
    "Do NOT start row 1 from the top pixel (0) of the image or empty margins.\n"
    "The rows MUST start where the actual handwritten text begins.\n"
    "Printed lines are layout guides, not mandatory cutting boundaries.\n"
    "Text may cross column borders.\n"
    "Do not invent equal-height rows.\n"
    "Do not transcribe or complete the ledger at this stage.\n"
    "Flag ambiguous boundaries and possible overlaps."
)

STAGE_A_SCHEMA = {
    "coordinate_space": {"width": "{W}", "height": "{H}"},
    "sections": [
        {
            "top": "integer (vertical Y pixel coordinate between 0 and {H})",
            "bottom": "integer (vertical Y pixel coordinate between 0 and {H})",
            "label": "string (visible section or date text)"
        }
    ],
    "columns": [
        {
            "guide_x": "float (approximate vertical line / guide x position)",
            "left": "float",
            "right": "float",
            "label": "string (visible column header label or empty)"
        }
    ],
    "rows": [
        {
            "top": "integer (vertical Y pixel coordinate between 0 and {H})",
            "bottom": "integer (vertical Y pixel coordinate between 0 and {H})",
            "ref_x": "float (horizontal X pixel coordinate inside main writing body)",
            "ref_y": "float (vertical Y pixel coordinate inside main writing body)",
            "row_type": "transaction | section_heading | subtotal | total | note | unknown",
            "review_flags": ["list of strings (e.g., ambiguous_cut, descending_stroke, possible_overlap)"]
        }
    ]
}


def build_stage_a_prompt(input_width: int, input_height: int) -> str:
    """
    Builds the whole-page layout proposal prompt.
    Zero concrete names, dates, or amounts are included.
    """
    rules = STAGE_A_SYSTEM_RULES.format(W=input_width, H=input_height)
    schema_str = json.dumps(STAGE_A_SCHEMA, indent=2).replace('"{W}"', str(input_width)).replace('"{H}"', str(input_height))

    prompt = (
        f"{rules}\n\n"
        f"Output MUST be valid JSON adhering strictly to this schema:\n"
        f"```json\n"
        f"{schema_str}\n"
        f"```\n"
        f"Do not include any text outside the JSON object."
    )
    return prompt


STAGE_B_SYSTEM_RULES = (
    "You are refining vertical boundaries between two neighboring ledger rows.\n"
    "Return only JSON matching the supplied schema.\n"
    "Coordinates refer to the supplied context crop of width {W} and height {H}.\n"
    "Refine the top and bottom bounds of both rows independently.\n"
    "Preserve each row's descending strokes, ascenders, and detached dots where visible.\n"
    "Do not force the bottom of the first row to equal the top of the second row.\n"
    "If writing vertically overlaps between the two rows, keep their true bounds and flag the overlap.\n"
    "Return ambiguity flags if a rectangular separation is unsafe."
)

STAGE_B_SCHEMA = {
    "row_a": {
        "row_id": "{ROW_A_ID}",
        "top": "float (refined upper boundary in context image coordinates)",
        "bottom": "float (refined lower boundary in context image coordinates)",
        "ambiguity_flags": ["list of strings (e.g. descending_flourish_crosses_boundary, touching_ink)"]
    },
    "row_b": {
        "row_id": "{ROW_B_ID}",
        "top": "float (refined upper boundary in context image coordinates)",
        "bottom": "float (refined lower boundary in context image coordinates)",
        "ambiguity_flags": ["list of strings"]
    },
    "separation_notes": "string explaining vertical boundary decisions",
    "conflict_detected": "boolean (true if safe rectangular full-width cut is impossible without cutting ink)"
}


def build_stage_b_prompt(
    context_width: int,
    context_height: int,
    row_a_id: str,
    row_b_id: str,
    row_a_approx: Dict[str, float],
    row_b_approx: Dict[str, float]
) -> str:
    """
    Builds two-row context refinement prompt.
    """
    rules = STAGE_B_SYSTEM_RULES.format(W=context_width, H=context_height)
    schema_str = (
        json.dumps(STAGE_B_SCHEMA, indent=2)
        .replace('"{ROW_A_ID}"', f'"{row_a_id}"')
        .replace('"{ROW_B_ID}"', f'"{row_b_id}"')
    )

    prompt = (
        f"{rules}\n\n"
        f"Context Details:\n"
        f"- Upper row ID: {row_a_id} (approx top: {row_a_approx.get('top', 0):.1f}, approx bottom: {row_a_approx.get('bottom', 0):.1f})\n"
        f"- Lower row ID: {row_b_id} (approx top: {row_b_approx.get('top', 0):.1f}, approx bottom: {row_b_approx.get('bottom', 0):.1f})\n\n"
        f"Output JSON schema:\n"
        f"```json\n"
        f"{schema_str}\n"
        f"```\n"
        f"Do not include any text outside the JSON object."
    )
    return prompt


STAGE_D_PROMPT = (
    "Transcribe the visible content of this ledger row faithfully.\n"
    "Preserve Urdu, English, numbers, punctuation, and ditto marks as written.\n"
    "Do not translate Urdu into English.\n"
    "Do not correct names or complete missing words from familiarity.\n"
    "Do not expand ditto marks.\n"
    "Do not infer missing dates or amounts.\n"
    "Report unreadable or ambiguous spans separately.\n"
    "Do not include content from an adjacent row when its membership is unclear."
)


def build_stage_d_prompt() -> str:
    """
    Returns exact Stage D row transcription prompt.
    """
    return STAGE_D_PROMPT


STAGE_E_SYSTEM_RULES = (
    "You are assigning transcribed text from a single ledger row to specific columns.\n"
    "Return only JSON matching the supplied schema.\n"
    "Rules:\n"
    "1. Combine position, alignment, phrase continuity, and content.\n"
    "2. Treat column guides as evidence, not absolute cutting boundaries. Writing crossing a printed border can still belong to the description.\n"
    "3. Urdu descriptions can contain numbers and English abbreviations.\n"
    "4. Numeric content outside description is not automatically a folio or amount.\n"
    "5. Nonnumeric content in another column is not automatically description overflow; legitimate entries may include Nil, dashes, or ditto marks (//).\n"
    "6. Preserve blank cells as empty strings \"\" and literal // as \"//\".\n"
    "7. Keep section-heading dates separate from explicitly written row dates. Do not silently propagate dates to blank cells.\n"
    "8. Do not rewrite the raw transcription or invent unsupported words.\n"
    "9. Flag uncertain ownership instead of moving content confidently."
)


def build_stage_e_prompt(
    row_id: str,
    row_type: str,
    raw_transcription: str,
    columns: List[Dict[str, Any]],
    crop_width: int,
    crop_height: int
) -> str:
    """
    Builds Stage E field assignment prompt using the crop, transcription, and mapped column guides.
    """
    cols_summary = []
    for col in columns:
        col_id = col.get("col_id", "col_unknown")
        label = col.get("label", "")
        left = col.get("left_canonical", 0)
        right = col.get("right_canonical", crop_width)
        cols_summary.append({
            "col_id": col_id,
            "label": label,
            "horizontal_span_x": f"[{left}, {right}]"
        })

    schema_example = {
        "row_id": row_id,
        "row_type": row_type,
        "cells": {col.get("col_id", f"col_{i+1:02d}"): "exact cell string or empty string" for i, col in enumerate(columns)},
        "unassigned_text": "any remainder text from raw transcription not assigned to columns",
        "ambiguous_spans": ["list of ambiguous phrase strings"],
        "review_flags": ["list of flags, e.g. uncertain_column, overflow, unsupported_text"]
    }

    prompt = (
        f"{STAGE_E_SYSTEM_RULES}\n\n"
        f"Row Metadata:\n"
        f"- Row ID: {row_id}\n"
        f"- Row Type: {row_type}\n"
        f"- Crop Dimensions: width={crop_width}, height={crop_height}\n\n"
        f"Available Page Columns:\n"
        f"{json.dumps(cols_summary, ensure_ascii=False, indent=2)}\n\n"
        f"Raw Transcription to Assign:\n"
        f"\"\"\"\n{raw_transcription}\n\"\"\"\n\n"
        f"Output JSON schema:\n"
        f"```json\n"
        f"{json.dumps(schema_example, ensure_ascii=False, indent=2)}\n"
        f"```\n"
        f"Do not include any text outside the JSON object."
    )
    return prompt
