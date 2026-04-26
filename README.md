# 🎙️ AudioMind AI — Real-Time Video/Audio Summarization Pipeline

> **Project 9** — End-to-end AI pipeline: Upload audio/video → Get transcript, insights & action items in real time.

---

## 🏗️ Architecture (10-Stage Pipeline)

```
User Upload / URL
      │
      ▼
[1] Streamlit UI          ← File upload, URL input, real-time progress
      │
      ▼
[2] Audio Extraction      ← FFmpeg / MoviePy extracts audio from video
      │
      ▼
[3] WhisperX Transcription ← Speech-to-Text with timestamps & diarization
      │
      ▼
[4] Text Cleaning & Chunking ← Remove fillers, split via LangChain
      │
      ▼
[5] LLM Map Step          ← Summarize each chunk (GPT-4o / Gemini)
      │
      ▼
[6] LLM Reduce Step       ← Consolidate summaries via LangChain chain
      │
      ▼
[7] Structured Output     ← Pydantic schema enforcement (Summary/Keys/Actions)
      │
      ▼
[8] Storage Layer         ← MongoDB (with local JSON fallback)
      │
      ▼
[9] Format & Export       ← TXT / JSON / Markdown download
      │
      ▼
[10] Streamlit Display    ← Split-screen: transcript + AI analysis
```

---

## 🚀 Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/yourname/audiomind-ai
cd audiomind-ai
pip install -r requirements.txt
```

> **FFmpeg required** for audio extraction:
> - macOS: `brew install ffmpeg`
> - Ubuntu: `sudo apt-get install ffmpeg`
> - Windows: Download from https://ffmpeg.org/download.html

### 2. Configure API Keys

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=sk-...       # Required for GPT-4o / GPT-4o-mini
GOOGLE_API_KEY=AIza...      # Required for Gemini 1.5 Pro
MONGO_URI=mongodb+srv://... # Optional (local JSON fallback if empty)
```

### 3. Run Locally

```bash
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Cloud (Free)

1. Push this folder to a **GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → Select your repo → Set **Main file**: `app.py`
4. Go to **Settings → Secrets** and add:
   ```toml
   OPENAI_API_KEY = "sk-..."
   GOOGLE_API_KEY = "AIza..."
   MONGO_URI = "mongodb+srv://..."
   ```
5. Click **Deploy** ✅

> ⚠️ Whisper requires ~1-2 GB RAM. Use `whisper-base` or `whisper-small` on free tier.
> For large files, upgrade to Streamlit Community Cloud Pro or deploy on AWS EC2.

---

## 📦 Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| Audio Extraction | FFmpeg, MoviePy |
| Speech-to-Text | OpenAI Whisper / WhisperX |
| Orchestration | LangChain (Map-Reduce chain) |
| LLM | GPT-4o, GPT-4o-mini, Gemini 1.5 Pro |
| Output Schema | Pydantic v2 |
| Storage | MongoDB Atlas (+ local JSON fallback) |
| URL Downloads | yt-dlp (optional) |
| Deployment | Streamlit Cloud / AWS |

---

## 🗂️ Project Structure

```
audiomind/
├── app.py                    # Main Streamlit application
├── style.css                 # Custom UI styles
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
├── .streamlit/
│   ├── config.toml           # Streamlit theme & server config
│   └── secrets.toml          # API keys (never commit real keys!)
├── pipeline/
│   ├── audio_extractor.py    # Stage 2: FFmpeg/MoviePy audio extraction
│   ├── transcriber.py        # Stage 3: Whisper/WhisperX transcription
│   ├── text_processor.py     # Stage 4: LangChain text cleaning & chunking
│   ├── summarizer.py         # Stages 5-7: LangChain LLM pipeline + Pydantic
│   └── storage.py            # Stage 8: MongoDB / JSON storage
├── utils/
│   └── helpers.py            # Timestamp formatting, export utilities
└── tmp/                      # Temporary file storage (auto-created)
```

---

## 🎯 Features

- ✅ Upload MP3, MP4, WAV, M4A, OGG, WebM, AVI, MKV, MOV
- ✅ YouTube / direct URL support (with yt-dlp)
- ✅ Real-time pipeline progress (Streamlit status widgets)
- ✅ WhisperX word-level timestamps & speaker diarization
- ✅ LangChain Map-Reduce summarization (handles long content)
- ✅ Structured output: Summary / Key Points / Action Items / Timestamps
- ✅ Pydantic schema validation
- ✅ MongoDB storage with search
- ✅ Export: TXT / JSON / Markdown
- ✅ Demo Mode (no API key required)
- ✅ Multi-model: GPT-4o, GPT-4o-mini, Gemini 1.5 Pro

---

## 🔧 Optional Enhancements

### Enable WhisperX (better timestamps + diarization)
```bash
pip install whisperx torch torchaudio
# Requires HuggingFace token for speaker diarization
```

### Enable YouTube URL downloads
```bash
pip install yt-dlp
```

### AWS Deployment (Stage 10)
- Deploy FastAPI backend on EC2
- Use Prometheus for monitoring (API uptime, performance metrics)
- Set up CloudWatch alerts

---

## 📄 License

MIT License — Free to use and modify.
