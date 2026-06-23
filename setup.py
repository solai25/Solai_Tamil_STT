import os
import subprocess
import sys
import platform

# NOTE: The huggingface_hub import was moved inside download_models() 
# to prevent crashing before dependencies are installed.

MODELS = {
    "whisper-tamil-small": "vasista22/whisper-tamil-small",
    "whisper-medium": "openai/whisper-large-v3"
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
            "--index-url", "https://download.pytorch.org/whl/cu124"
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
        "transformers", "gradio", "huggingface_hub", "PySide6"
    ])


# =========================
# MODEL DOWNLOAD
# =========================
def download_models():
    # Deferring import until pip install has completed successfully
    from huggingface_hub import snapshot_download

    os.makedirs(MODELS_DIR, exist_ok=True)

    for name, repo in MODELS.items():
        path = os.path.join(MODELS_DIR, name)

        if os.path.exists(path) and os.listdir(path):
            print(f"✅ {name} already exists, skipping...")
        else:
            print(f"⬇️ Downloading {name} from Hugging Face...")
            snapshot_download(repo_id=repo, local_dir=path)


# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":
    print("🚀 Tamil STT Smart Installer")
    
    # 1. Install PyTorch (CPU or GPU)
    install_torch()
    
    # 2. Install other pip dependencies
    install_other_requirements()
    
    # 3. Download the Whisper models
    download_models()
    
    print("✅ Setup complete! Launching App...")
