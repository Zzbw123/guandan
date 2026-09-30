"""Synchronous CUDA research training; no rule or policy information changes."""

import os

# This must precede importing PyTorch in the CUDA entry point.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

