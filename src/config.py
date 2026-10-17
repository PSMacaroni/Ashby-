"""
Project settings, read from the .env file.

Import this FIRST (before torch / transformers / huggingface_hub), because Hugging Face
only reads HF_HOME (where it saves model downloads) once, when it is first imported.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# An empty "HF_HOME=" line would make Hugging Face save files in the current folder,
# so remove empty values and let it use its normal default instead.
if not os.environ.get("HF_HOME", "").strip():
    os.environ.pop("HF_HOME", None)


def _path(name):
    value = os.getenv(name, "").strip()
    return Path(value).expanduser() if value else None


HF_TOKEN = os.getenv("HF_TOKEN", "").strip() or None
DATA_DIR = _path("DATA_DIR")
MODEL_DIR = _path("MODEL_DIR")
RESULTS_DIR = ROOT / "data" / "results"


def get_device():
    """'cuda' if a GPU is visible (Delta GPU session), otherwise 'cpu' (laptop)."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
