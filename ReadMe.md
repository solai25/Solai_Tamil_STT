# 🎙️ Solai's Tamil Speech-to-Text (Offline)

A simple and powerful local Speech-to-Text application for Tamil & English using Whisper models.

---

## 🚀 Features

* 🎧 Audio to Text (Tamil & English)
* 📂 Batch folder transcription
* ⚡ GPU acceleration (Auto detected)
* 🤖 Automatic model download
* 🔒 Fully offline (privacy safe)
* 📝 Clean preview UI

---

## ⚡ One-Click Setup (Recommended)

👉 **For beginners (Windows users):**

Just double-click:

```
One-Click-Install.bat
```

This will automatically:

* Detect your GPU 🧠
* Install correct PyTorch (CUDA / CPU)
* Install all required dependencies 📦
* Download models 🤖
* Launch the app 🚀

---

## 🛠️ Manual Installation (Step-by-Step)

### 1. Clone Repository

```
git clone https://github.com/solai25/Solai_Tamil_STT.git
cd Solai_Tamil_STT
```

---

### 2. Create Virtual Environment

```
python -m venv venv
venv\Scripts\activate
```

---

### 3. Run Smart Installer

```
python setup.py
```

✔️ This will:

* Auto detect GPU
* Install correct Torch version
* Install dependencies
* Download models

---

### 4. Run the App

```
python app.py
```

Open in browser:

```
http://127.0.0.1:7860
```

---

## 📥 Models Used

### 🔹 Tamil Whisper Models

* https://huggingface.co/vasista22/whisper-tamil-small
* https://huggingface.co/vasista22/whisper-tamil-medium

### 🔹 OpenAI Whisper

* https://huggingface.co/openai/whisper-medium

---

## 📁 Project Structure

```
Tamil-STT-App/
│
├── app.py
├── setup.py
├── run.bat
├── README.md
├── models/
└── output/
```

---

## 📌 Usage

1. Select model
2. Upload audio OR folder
3. Click **Start**
4. View transcription

---

## ⚠️ Notes

* First run may take time (model download ~1–3GB)
* GPU recommended for faster performance
* Works fully offline after setup

---

## 🙏 Credits

* OpenAI Whisper
  https://huggingface.co/openai/whisper-medium

* Tamil Whisper Models by Vasista
  https://huggingface.co/vasista22

* HuggingFace Transformers
  https://github.com/huggingface/transformers

---

Made with ❤️ for Tamil AI community
