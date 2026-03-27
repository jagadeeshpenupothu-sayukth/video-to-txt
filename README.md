# 🎥 Video to Text UI

A web-based application to convert video/audio into text using AI models.
Supports transcription, translation, and optional text-to-speech (TTS).

---

# ✨ Features

* 🎬 Upload video/audio files
* 🧠 Speech-to-text using Faster-Whisper
* 🌐 Translate text (Deep Translator)
* 🔊 Optional Text-to-Speech (XTTS)
* 🖥️ Simple UI (FastAPI + Jinja2)

---

# 🛠️ Tech Stack

* FastAPI
* Faster-Whisper
* Torch (AI backend)
* Transformers
* Jinja2 Templates

---

# 📦 Requirements

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

# 🚀 Setup Instructions

## 1. Clone Repo

```bash
git clone https://github.com/jagadeeshpenupothu-sayukth/video-to-txt.git
cd video-to-txt
```

---

## 2. Create Virtual Environment

```bash
python3.10 -m venv venv
```

---

## 3. Activate

```bash
source venv/bin/activate
```

---

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the App

```bash
uvicorn app:app --reload
```

Open:

```
http://localhost:8000
```

---

# ⚠️ Important Notes

* Use **Python 3.10**
* Versions are locked to avoid AI library conflicts
* First run may download models (takes time)

---

# 🔊 TTS (Text-to-Speech)

TTS is included but depends on specific versions:

```text
torch==2.1.2
transformers==4.38.2
```

If you face issues:

```bash
pip install torch==2.1.2
pip install transformers==4.38.2
```

---

# 🧪 Testing Workflow

```bash
git checkout <branch>
rm -rf venv
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

---

# 🔄 Branch Strategy

```text
main            → stable version
feature/copy    → feature version
feature/base    → initial version
```

Switch:

```bash
git checkout feature/copy
```

---

# 📁 Project Structure

```text
video-to-txt/
│
├── app.py
├── xtts_service.py
├── requirements.txt
├── start_server.sh
├── templates/
│   ├── index.html
│   └── index copy.html
```

---

# 🚀 Future Improvements

* Docker support
* GPU acceleration
* UI improvements
* API endpoints

---

# 🙌 Acknowledgements

* OpenAI Whisper
* Faster-Whisper
* Coqui TTS
* FastAPI
