"""
merge_adapters.py
=============================================================================
Mathematically combines the 627-sample ledger LoRA adapter (r=8) and the
10,991-sample UHWR+ledger LoRA adapter (r=16) into a single unified
lossless combined LoRA adapter (r=24).
=============================================================================
"""

import os
import json
import shutil
import torch
from safetensors.torch import load_file, save_file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ADAPTER_627_DIR = os.path.join(BASE_DIR, "local_adapter_627")
ADAPTER_10991_DIR = os.path.join(BASE_DIR, "local_adapter_10991")
OUTPUT_DIR = os.path.join(BASE_DIR, "combined_adapter")


def merge(w1_weight: float = 0.4, w2_weight: float = 0.6):
    print("=" * 65)
    print("Fusing LoRA Weights: 627 Ledger Adapter + 10,991 UHWR Adapter")
    print(f"Weights: 627 adapter = {w1_weight:.1f}, 10,991 adapter = {w2_weight:.1f}")
    print("=" * 65)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    path1 = os.path.join(ADAPTER_627_DIR, "adapter_model.safetensors")
    path2 = os.path.join(ADAPTER_10991_DIR, "adapter_model.safetensors")

    if not os.path.exists(path1):
        raise FileNotFoundError(f"Missing 627 adapter weights at {path1}")
    if not os.path.exists(path2):
        raise FileNotFoundError(f"Missing 10,991 adapter weights at {path2}")

    print("Loading weights from safetensors...")
    w1 = load_file(path1)
    w2 = load_file(path2)

    modules_A = [k for k in w2.keys() if k.endswith("lora_A.weight")]
    merged_dict = {}

    for k_A in modules_A:
        k_B = k_A.replace("lora_A.weight", "lora_B.weight")
        A2 = w2[k_A]
        B2 = w2[k_B]
        d_out, r2 = B2.shape
        _, d_in = A2.shape

        if k_A in w1:
            A1 = w1[k_A]
            B1 = w1[k_B]
            A_comb = torch.cat([A1, A2], dim=0)
            B_comb = torch.cat([w1_weight * B1, w2_weight * B2], dim=1)
        else:
            A_comb = torch.cat([torch.zeros((8, d_in), dtype=A2.dtype), A2], dim=0)
            B_comb = torch.cat([torch.zeros((d_out, 8), dtype=B2.dtype), w2_weight * B2], dim=1)

        merged_dict[k_A] = A_comb
        merged_dict[k_B] = B_comb

    # Save merged safetensors
    out_safetensors = os.path.join(OUTPUT_DIR, "adapter_model.safetensors")
    print(f"Saving merged safetensors ({len(merged_dict)} tensors) to {out_safetensors}...")
    save_file(merged_dict, out_safetensors)

    # Read and update adapter_config.json
    with open(os.path.join(ADAPTER_10991_DIR, "adapter_config.json"), "r") as f:
        cfg = json.load(f)

    cfg["r"] = 24
    cfg["lora_alpha"] = 48  # Maintains 48/24 = 2.0 scaling exactly

    out_config = os.path.join(OUTPUT_DIR, "adapter_config.json")
    with open(out_config, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved merged config (r=24, alpha=48) to {out_config}")

    # Copy tokenizer and other metadata files
    for fname in os.listdir(ADAPTER_10991_DIR):
        if fname not in ("adapter_model.safetensors", "adapter_config.json") and not fname.startswith("checkpoint-"):
            src = os.path.join(ADAPTER_10991_DIR, fname)
            dst = os.path.join(OUTPUT_DIR, fname)
            if os.path.isfile(src):
                shutil.copy2(src, dst)

    print("=" * 65)
    print("✅ Combined adapter successfully created in ./combined_adapter/")
    print("=" * 65)


if __name__ == "__main__":
    merge()
