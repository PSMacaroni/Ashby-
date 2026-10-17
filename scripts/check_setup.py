"""
Setup checker for the team. Run from the project folder:

    python scripts/check_setup.py

It prints [OK], [FAIL], or [SKIP] for each check, with a hint for anything that failed.
It never prints your tokens.
"""
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# pip package name -> the name you use in `import ...`
PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "pillow": "PIL",
    "matplotlib": "matplotlib",
    "scikit-learn": "sklearn",
    "scikit-image": "skimage",
    "tqdm": "tqdm",
    "python-dotenv": "dotenv",
    "torch": "torch",
    "torchvision": "torchvision",
    "timm": "timm",
    "open_clip_torch": "open_clip",
    "huggingface_hub": "huggingface_hub",
    "transformers": "transformers",
    "conch (from GitHub)": "conch.open_clip_custom",
    "musk (from GitHub)": "musk.modeling",
    "einops": "einops",
    "fairscale": "fairscale",
    "safetensors": "safetensors",
    "openslide-bin": "openslide_bin",
    "openslide-python": "openslide",
    "openai": "openai",
    "streamlit": "streamlit",
    "plotly": "plotly",
}

GATED_MODELS = ["MahmoodLab/CONCH", "xiangjx/musk", "jamessyx/PathGen-CLIP"]
GATED_DATASETS = ["jamessyx/PathGen"]

problems = 0


def ok(msg):
    print(f"  [OK]   {msg}")


def fail(msg, hint):
    global problems
    problems += 1
    print(f"  [FAIL] {msg}")
    print(f"         hint: {hint}")


def skip(msg):
    print(f"  [SKIP] {msg}")


def section(title):
    print(f"\n== {title} ==")


def short_error(e):
    text = str(e).strip().splitlines()
    return f"{type(e).__name__}: {text[0] if text else ''}"[:200]


def check_python():
    section("Python")
    v = sys.version_info
    if v >= (3, 10):
        ok(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        fail(f"Python {v.major}.{v.minor} is too old (need 3.10+, MUSK requires it)",
             "Install a newer Python, or on Delta run `module load python` first.")


def check_packages():
    section("Packages")
    for pip_name, import_name in PACKAGES.items():
        try:
            importlib.import_module(import_name)
            ok(pip_name)
        except Exception as e:
            fail(f"{pip_name} (import {import_name}) -> {short_error(e)}",
                 "Run: pip install -r requirements.txt  (on Delta add --user)")


def check_gpu():
    section("GPU")
    try:
        import torch
    except Exception:
        skip("torch not installed, can't check GPU")
        return
    if torch.cuda.is_available():
        ok(f"GPU found: {torch.cuda.get_device_name(0)} (x{torch.cuda.device_count()})")
    else:
        fail("No GPU visible to torch",
             "Normal on a laptop - ignore it there. On Delta, start a Jupyter session "
             "on a GPU partition (e.g. gpuA100x4) in Open OnDemand.")


def load_env():
    section(".env file")
    env_path = ROOT / ".env"
    if not env_path.exists():
        fail(".env not found",
             "Copy the template: cp .env.example .env  then fill in your values.")
        return
    try:
        from dotenv import load_dotenv
    except Exception:
        fail("python-dotenv not installed, can't read .env",
             "Run: pip install python-dotenv")
        return
    load_dotenv(env_path)
    ok(f"loaded {env_path}")


def check_huggingface():
    section("Hugging Face")
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        skip("HF_TOKEN not set in .env")
        return
    try:
        from huggingface_hub import HfApi
    except Exception:
        skip("huggingface_hub not installed")
        return

    api = HfApi(token=token)
    try:
        who = api.whoami()
        ok(f"logged in as {who.get('name', '?')}")
    except Exception as e:
        fail(f"token rejected -> {short_error(e)}",
             "Make a new Read token at huggingface.co/settings/tokens and paste it into .env.")
        return

    repos = [(r, "model") for r in GATED_MODELS] + [(r, "dataset") for r in GATED_DATASETS]
    for repo_id, repo_type in repos:
        try:
            files = api.list_repo_files(repo_id, repo_type=repo_type)
            ok(f"{repo_type} {repo_id} ({len(files)} files visible)")
        except Exception as e:
            prefix = "datasets/" if repo_type == "dataset" else ""
            fail(f"{repo_type} {repo_id} -> {short_error(e)}",
                 f"Open https://huggingface.co/{prefix}{repo_id} while logged in and check "
                 "your access request says APPROVED (it can take a while).")


def check_lumen():
    section("NCSA Lumen")
    key = os.getenv("LUMEN_API_KEY", "").strip()
    base_url = os.getenv("LUMEN_BASE_URL", "").strip()
    model = os.getenv("LUMEN_MODEL", "").strip()
    if not key or not base_url:
        skip("LUMEN_API_KEY or LUMEN_BASE_URL not set (optional)")
        return
    try:
        from openai import OpenAI
    except Exception:
        skip("openai package not installed")
        return

    client = OpenAI(api_key=key, base_url=base_url, timeout=30)
    if not model:
        try:
            names = [m.id for m in client.models.list()]
            ok(f"connected; available models: {', '.join(names) or '(none listed)'}")
            print("         -> pick one and put it in LUMEN_MODEL in .env")
        except Exception as e:
            fail(f"could not list models -> {short_error(e)}",
                 "Check LUMEN_BASE_URL (often ends in /v1) and LUMEN_API_KEY.")
        return
    try:
        reply = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with the single word: ready"}],
            max_tokens=5,
        )
        text = (reply.choices[0].message.content or "").strip()
        ok(f"model {model} replied: {text!r}")
    except Exception as e:
        fail(f"test message to {model} failed -> {short_error(e)}",
             "Check the model name (leave LUMEN_MODEL empty to list models), key, and base URL.")


def check_dirs():
    section("Folders")
    for var in ("DATA_DIR", "MODEL_DIR"):
        value = os.getenv(var, "").strip()
        if not value:
            skip(f"{var} not set")
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        if path.is_dir():
            ok(f"{var} = {path}")
        else:
            fail(f"{var} = {path} does not exist",
                 "Fix the path in .env (on Delta, ask organizers for the preloaded path).")


def main():
    print("Checking your setup...")
    check_python()
    check_packages()
    check_gpu()
    load_env()
    check_huggingface()
    check_lumen()
    check_dirs()

    print("\n" + "=" * 40)
    if problems == 0:
        print("All good! 0 problems found.")
    else:
        print(f"{problems} problem(s) found. Read the hints above, fix, and run again.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
