import sys
import os
import torch
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QProgressBar, QListWidget, QComboBox
)
from PySide6.QtCore import Qt, QThread, Signal
from transformers import pipeline

# CONFIG
MODELS_ROOT = "./models"
OUTPUT_DIR = "./output"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# WORKER THREAD (IMPORTANT)
# =========================
class TranscribeWorker(QThread):
    progress = Signal(int)
    finished = Signal(str)

    def __init__(self, files, model_path, lang):
        super().__init__()
        self.files = files
        self.model_path = model_path
        self.lang = lang

    def run(self):
        try:
            pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model_path,
                device=0 if DEVICE == "cuda" else -1
            )

            lang_code = "ta" if self.lang == "Tamil" else "en"
            pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(
                language=lang_code,
                task="transcribe"
            )

            results = []

            for i, file in enumerate(self.files):
                result = pipe(file)
                text = result.get("text", "")

                out_file = os.path.join(
                    OUTPUT_DIR, f"{os.path.basename(file)}.txt"
                )
                with open(out_file, "w", encoding="utf-8") as f:
                    f.write(text)

                results.append(f"{file} → DONE")

                percent = int((i + 1) / len(self.files) * 100)
                self.progress.emit(percent)

            self.finished.emit("\n".join(results))

        except Exception as e:
            self.finished.emit(f"ERROR: {str(e)}")


# =========================
# MAIN UI
# =========================
class App(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("🔥 Solai Tamil STT Pro")
        self.setMinimumSize(500, 500)

        layout = QVBoxLayout()

        # MODEL SELECT
        self.model_box = QComboBox()
        self.model_box.addItems(self.get_models())
        layout.addWidget(self.model_box)

        # LANGUAGE
        self.lang_box = QComboBox()
        self.lang_box.addItems(["Tamil", "English"])
        layout.addWidget(self.lang_box)

        # FILE LIST (DRAG & DROP)
        self.file_list = QListWidget()
        self.file_list.setAcceptDrops(True)
        layout.addWidget(self.file_list)

        # BUTTONS
        self.btn_file = QPushButton("📂 Add Audio Files")
        self.btn_file.clicked.connect(self.load_files)
        layout.addWidget(self.btn_file)

        self.btn_folder = QPushButton("📁 Add Folder")
        self.btn_folder.clicked.connect(self.load_folder)
        layout.addWidget(self.btn_folder)

        # ADD THIS NEW BUTTON:
        self.btn_remove = QPushButton("🗑️ Remove Selected")
        self.btn_remove.clicked.connect(self.remove_selected)
        layout.addWidget(self.btn_remove)

        self.btn_start = QPushButton("🚀 Start Transcription")
        self.btn_start.clicked.connect(self.start_transcription)
        layout.addWidget(self.btn_start)

        # PROGRESS
        self.progress = QProgressBar()
        layout.addWidget(self.progress)

        # OUTPUT
        self.output = QLabel("")
        self.output.setWordWrap(True)
        layout.addWidget(self.output)

        self.setLayout(layout)

        self.files = []

        self.set_dark_theme()

    # =========================
    # FUNCTIONS
    # =========================
    def get_models(self):
        if not os.path.exists(MODELS_ROOT):
            return ["No models"]
        return [
            d for d in os.listdir(MODELS_ROOT)
            if os.path.isdir(os.path.join(MODELS_ROOT, d))
        ] or ["No models"]

    def load_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Audio", "", "Audio Files (*.wav *.mp3 *.m4a)"
        )
        self.add_files(files)

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            files = [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.endswith((".wav", ".mp3", ".m4a"))
            ]
            self.add_files(files)

    def add_files(self, files):
        for f in files:
            if f not in self.files:
                self.files.append(f)
                self.file_list.addItem(f)

    # ADD THIS NEW FUNCTION:
    def remove_selected(self):
        # Get a list of all currently selected items in the UI
        selected_items = self.file_list.selectedItems()
        
        if not selected_items:
            return # Nothing is selected, do nothing
            
        for item in selected_items:
            file_path = item.text()
            
            # 1. Remove from our internal Python list
            if file_path in self.files:
                self.files.remove(file_path)
                
            # 2. Remove from the visual QListWidget
            row = self.file_list.row(item)
            self.file_list.takeItem(row)

    def start_transcription(self):
        if not self.files:
            self.output.setText("⚠️ No files selected")
            return

        model = os.path.join(MODELS_ROOT, self.model_box.currentText())
        lang = self.lang_box.currentText()

        self.worker = TranscribeWorker(self.files, model, lang)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def on_finished(self, msg):
        self.output.setText(msg)

    # =========================
    # DARK THEME
    # =========================
    def set_dark_theme(self):
        self.setStyleSheet("""
        QWidget {
            background-color: #121212;
            color: white;
            font-size: 14px;
        }
        QPushButton {
            background-color: #1f1f1f;
            padding: 10px;
            border-radius: 6px;
        }
        QPushButton:hover {
            background-color: #333;
        }
        QProgressBar {
            border: 1px solid #444;
            border-radius: 5px;
        }
        QProgressBar::chunk {
            background-color: #00ffcc;
        }
        """)

    # =========================
    # DRAG & DROP
    # =========================
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        self.add_files(files)


# =========================
# RUN
# =========================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec())
