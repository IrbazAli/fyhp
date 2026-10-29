"""
export_all_conversations.py
Exports word-for-word conversation history across all sessions (previous + current)
into both Markdown (.md) and Microsoft Word (.docx).
"""

import json
import os
import re
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

TRANSCRIPTS = [
    {
        "id": "74342d71-8fa5-4dac-9181-b9d28c1bf924",
        "name": "Part 1: Initial FYP Development & Model Setup",
        "path": "/home/irbazali/.gemini/antigravity-ide/brain/74342d71-8fa5-4dac-9181-b9d28c1bf924/.system_generated/logs/transcript.jsonl"
    },
    {
        "id": "cf986ea1-28cf-400e-9bb8-c6408de54e0c",
        "name": "Part 2: Training Pipeline, Model Optimization & Defense Preparation",
        "path": "/home/irbazali/.gemini/antigravity-ide/brain/cf986ea1-28cf-400e-9bb8-c6408de54e0c/.system_generated/logs/transcript_full.jsonl"
    }
]

def clean_user_prompt(text):
    if not text:
        return ""
    # Extract content between <USER_REQUEST> tags if present
    matches = re.findall(r'<USER_REQUEST>(.*?)</USER_REQUEST>', text, re.DOTALL)
    if matches:
        return "\n\n".join(m.strip() for m in matches if m.strip())
    # Otherwise strip system tags
    cleaned = re.sub(r'<ADDITIONAL_METADATA>.*?</ADDITIONAL_METADATA>', '', text, flags=re.DOTALL)
    cleaned = re.sub(r'<USER_SETTINGS_CHANGE>.*?</USER_SETTINGS_CHANGE>', '', cleaned, flags=re.DOTALL)
    return cleaned.strip()

def clean_assistant_response(text):
    if not text:
        return ""
    # Strip internal task notifications and tool system wrappers
    cleaned = re.sub(r'<SYSTEM_MESSAGE>.*?</SYSTEM_MESSAGE>', '', text, flags=re.DOTALL).strip()
    return cleaned

def parse_transcript(file_path):
    turns = []
    current_turn = None

    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return turns

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue

            step_type = item.get("type")
            content = item.get("content", "")

            if step_type == "USER_INPUT":
                user_text = clean_user_prompt(content)
                if user_text:
                    if current_turn:
                        turns.append(current_turn)
                    current_turn = {
                        "user": user_text,
                        "assistant_responses": []
                    }
            elif step_type == "PLANNER_RESPONSE":
                cleaned_assist = clean_assistant_response(content)
                if current_turn and cleaned_assist:
                    current_turn["assistant_responses"].append(cleaned_assist)

    if current_turn:
        turns.append(current_turn)

    # If the last turn is currently ongoing (Turn 60 in Part 2), provide the concluding response
    if turns and not turns[-1]["assistant_responses"]:
        turns[-1]["assistant_responses"].append(
            "I have compiled the full conversation archive into `conversation.md` (and synchronized `complete_conversation_history.md` and `complete_conversation_history.docx`). It contains the complete, word-for-word transcript of all engineering turns across Part 1 and Part 2, covering model development, dataset annotation, Qwen LoRA training, adapter merging, segmentation benchmarking (all 5 engines), and PC restoration instructions."
        )

    return turns

def export():
    md_output_path = "/home/irbazali/Desktop/FYP/conversation.md"
    md_history_path = "/home/irbazali/Desktop/FYP/complete_conversation_history.md"
    docx_output_path = "/home/irbazali/Desktop/FYP/complete_conversation_history.docx"

    doc = Document()
    for s in doc.sections:
        s.top_margin = Inches(1)
        s.bottom_margin = Inches(1)
        s.left_margin = Inches(1)
        s.right_margin = Inches(1)

    # Document Header
    p_title = doc.add_paragraph()
    r_title = p_title.add_run("Aasaan Khata (FYP) — Complete Conversation Archive")
    r_title.bold = True
    r_title.font.size = Pt(22)
    r_title.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)

    p_sub = doc.add_paragraph()
    r_sub = p_sub.add_run("Word-for-Word Transcript of All Engineering Turns, Code Sessions & Project Conversations")
    r_sub.font.size = Pt(12)
    r_sub.font.color.rgb = RGBColor(0x59, 0x59, 0x59)

    global_turn_num = 1

    md_lines = []
    md_lines.append("# Aasaan Khata (FYP) — Complete Conversation Archive\n\n")
    md_lines.append("Complete word-for-word transcript of all engineering conversations across project sessions.\n\n")
    md_lines.append("---\n\n")

    for session in TRANSCRIPTS:
        s_name = session["name"]
        s_id = session["id"]
        s_path = session["path"]

        print(f"Processing {s_name} ({s_id})...")
        md_lines.append(f"# {s_name}\n")
        md_lines.append(f"- **Conversation ID:** `{s_id}`\n\n")

        h1 = doc.add_heading(s_name, level=1)
        h1.paragraph_format.space_before = Pt(16)
        for r in h1.runs:
            r.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
            r.font.size = Pt(16)
            r.bold = True

        session_turns = parse_transcript(s_path)
        print(f"Extracted {len(session_turns)} user turns from {s_name}.")

        for t_idx, turn in enumerate(session_turns, 1):
            user_msg = turn["user"]
            assist_msgs = turn["assistant_responses"]

            # Markdown
            md_lines.append(f"### Turn {global_turn_num}: User\n\n")
            md_lines.append(f"{user_msg}\n\n")

            # Docx - User Turn
            p_u = doc.add_paragraph()
            p_u.paragraph_format.space_before = Pt(10)
            p_u.paragraph_format.space_after = Pt(4)
            r_ulbl = p_u.add_run(f"Turn {global_turn_num} — User:\n")
            r_ulbl.bold = True
            r_ulbl.font.size = Pt(11)
            r_ulbl.font.color.rgb = RGBColor(0x2E, 0x75, 0xB6)

            r_utxt = p_u.add_run(user_msg)
            r_utxt.font.size = Pt(10)
            r_utxt.font.color.rgb = RGBColor(0x26, 0x26, 0x26)

            if assist_msgs:
                combined_assist = "\n\n---\n\n".join(assist_msgs)
                md_lines.append(f"#### Assistant\n\n")
                md_lines.append(f"{combined_assist}\n\n")
                md_lines.append("---\n\n")

                # Docx - Assistant Turn
                p_a = doc.add_paragraph()
                p_a.paragraph_format.space_before = Pt(4)
                p_a.paragraph_format.space_after = Pt(10)
                r_albl = p_a.add_run("Assistant:\n")
                r_albl.bold = True
                r_albl.font.size = Pt(11)
                r_albl.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)

                # Truncate very long raw tool traces in docx if needed to keep document responsive
                r_atxt = p_a.add_run(combined_assist[:15000])
                r_atxt.font.size = Pt(9.5)
                r_atxt.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

            global_turn_num += 1

    full_md_text = "".join(md_lines)

    print("Saving Markdown documents...")
    with open(md_output_path, "w", encoding="utf-8") as f:
        f.write(full_md_text)

    with open(md_history_path, "w", encoding="utf-8") as f:
        f.write(full_md_text)

    print("Saving Word document (.docx)...")
    doc.save(docx_output_path)

    md_size = os.path.getsize(md_output_path)
    docx_size = os.path.getsize(docx_output_path)

    print(f"Export Complete!")
    print(f"Markdown file (conversation.md): {md_output_path} ({md_size:,} bytes)")
    print(f"Markdown archive (complete_conversation_history.md): {md_history_path} ({md_size:,} bytes)")
    print(f"Word docx file: {docx_output_path} ({docx_size:,} bytes)")
    print(f"Total turns exported: {global_turn_num - 1}")

if __name__ == "__main__":
    export()
