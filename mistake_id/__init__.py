"""Mistake identification for AI-tutor responses (BEA 2025 Shared Task, Track 1)."""
import os

# Apple-silicon GPUs share memory with the system. Without a cap, PyTorch's MPS
# allocator caches far more than a 16 GB laptop can spare and the machine starts
# swapping (one training step went from 3.6 s to 14.7 s). Cap the pool at 60% of the
# recommended working set; these must be set before torch initialises MPS.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.6")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")
