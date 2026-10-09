"""
torch_patch.py
=============================================================================
Aasaan Khata - PyTorch 2.5.x Compatibility Monkey Patch
=============================================================================
Resolves: AttributeError: module 'torch' has no attribute 'accelerator'
which occurs when transformers 4.49+ / 5.x is used with PyTorch 2.5.x.
Must be imported BEFORE transformers, peft, or diffusers.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch

if not hasattr(torch, "accelerator"):
    class DummyAccelerator:
        @staticmethod
        def current_accelerator():
            return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        @staticmethod
        def device_count():
            return torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        @staticmethod
        def is_available():
            return torch.cuda.is_available()

    torch.accelerator = DummyAccelerator
