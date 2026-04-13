
import os
import subprocess
import sys
import platform
from huggingface_hub import snapshot_download

MODELS = {
    "whisper-tamil-small": "vasista22/whisper-tamil-small",
    "whisper-tamil-medium": "vasista22/whisper-tamil-medium",
    "whisper-medium": "openai/whisper-medium"
}

MODELS_DIR = "./models"

# =========================
# GPU DETECTION
# =========================
def has_nvidia_gpu():
    try:
        result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    except Exception:
        return False


def install_torch():
    print("🔍 Detecting system...")

    if has_nvidia_gpu():
        print("⚡ NVIDIA GPU detected → Installing CUDA PyTorch")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "torch", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ])
    else:
        print("💻 No GPU detected → Installing CPU PyTorch")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "torch", "torchaudio"
        ])


def install_other_requirements():
    print("📦 Installing other dependencies...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "transformers", "gradio", "huggingface_hub"
    ])


# =========================
# MODEL DOWNLOAD
# =========================
def download_models():
    os.makedirs(MODELS_DIR, exist_ok=True)

    for name, repo in MODELS.items():
        path = os.path.join(MODELS_DIR, name)

        if os.path.exists(path) and os.listdir(path):
            print(f"✅ {name} already exists, skipping...")
    print("🚀 Tamil STT Smart Installer
