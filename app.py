# =========================
# app.py (Transformer Version - Fixed)
# =========================

import os
import torch
import gradio as gr
import logging
from transformers import pipeline

# CONFIG
MODELS_ROOT = "./models"
OUTPUT_DIR = "./output"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
README_PATH = os.path.join(BASE_DIR, "README.md")

CURRENT_PIPE = None
CURRENT_MODEL_PATH = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

os.makedirs(MODELS_ROOT, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read README.md content safely with UTF-8 encoding if it exists
readme_content = "### 📄 Documentation\n`README.md` file was not found in the project root directory."
if os.path.exists(README_PATH):
    try:
        with open(README_PATH, "r", encoding="utf-8") as f:
            readme_content = f.read()
    except Exception as e:
        readme_content = f"### ⚠️ Error Loading Documentation\nFailed to parse `README.md` correctly: {str(e)}"

# =========================
# MODEL MANAGEMENT
# =========================
def list_local_models():
    return [d for d in os.listdir(MODELS_ROOT) if os.path.isdir(os.path.join(MODELS_ROOT, d))] or ["No models found"]


def load_model(model_name):
    global CURRENT_PIPE, CURRENT_MODEL_PATH

    model_path = os.path.join(MODELS_ROOT, model_name)

    if CURRENT_PIPE and CURRENT_MODEL_PATH == model_path:
        return CURRENT_PIPE

    try:
        logging.info(f"Loading model: {model_name} on {DEVICE}")

        pipe = pipeline(
            task="automatic-speech-recognition",
            model=model_path,
            device=0 if DEVICE == "cuda" else -1,
            dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            chunk_length_s=30,
            stride_length_s=5,
            batch_size=8
        )

        CURRENT_PIPE = pipe
        CURRENT_MODEL_PATH = model_path
        return pipe

    except Exception as e:
        logging.error(f"Model load failed: {e}")
        return None

# =========================
# UTIL
# =========================
def get_next_output_filename():
    files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.startswith("output_")])
    return os.path.join(OUTPUT_DIR, f"output_{len(files):03d}.txt")

# =========================
# TRANSCRIPTION
# =========================
def transcribe(audio, folder_path, model_choice, language_choice, progress=gr.Progress()):
    pipe = load_model(model_choice)
    if not pipe:
        return "❌ Model load failed"

    lang_code = "ta" if language_choice == "Tamil" else "en"

    pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(
        language=lang_code,
        task="transcribe"
    )

    files = []
    if audio:
        files = [audio]
    elif folder_path and os.path.exists(folder_path):
        files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith((".wav", ".mp3", ".m4a"))
        ]

    if not files:
        return "⚠️ No audio files found"

    results = []

    for i, file in enumerate(files):
        progress(i / len(files), desc=f"Processing {os.path.basename(file)}")

        try:
            result = pipe(file)
            text = result.get("text", "")

            out_file = get_next_output_filename()
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(text)

            preview = f"""
### 📄 {os.path.basename(file)}
**🌐 Language:** {language_choice}

{text}
"""

            results.append(preview)

        except Exception as e:
            results.append(f"❌ {file}: {str(e)}")

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    return "\n\n---\n\n".join(results)

# =========================
# UI
# =========================
def build_ui():
    with gr.Blocks(title="Solai's Tamil STT Pro") as app:

        gr.Markdown("""
        # 🎙️ Solai's Tamil Speech-to-Text Studio
        Local • Fast • Privacy Friendly
        """)

        with gr.Row():
            model_dropdown = gr.Dropdown(label="🤖 Model", choices=list_local_models())
            lang_dropdown = gr.Radio(["Tamil", "English"], value="Tamil", label="🌐 Language")

        with gr.Tab("🎧 Single File"):
            audio_input = gr.Audio(type="filepath", label="Upload Audio")

        with gr.Tab("📂 Batch Folder"):
            folder_input = gr.Textbox(label="Folder Path")

        gr.Markdown("## 📝 Transcription Preview")
        output_text = gr.Markdown()
            
        with gr.Row():
            submit_btn = gr.Button("🚀 Start", variant="primary")
            clear_btn = gr.Button("🧹 Clear")
            
        submit_btn.click(
            fn=transcribe,
            inputs=[audio_input, folder_input, model_dropdown, lang_dropdown],
            outputs=output_text
        )
        
        # DOCUMENTATION
        with gr.Tab("📄 Studio Guide & Docs"):
            gr.Markdown(readme_content)

        clear_btn.click(lambda: "", None, output_text)

    return app


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="127.0.0.1", server_port=7861, inbrowser=True, theme=gr.themes.Soft())
