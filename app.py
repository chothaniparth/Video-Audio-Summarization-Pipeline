import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(override=True)

st.set_page_config(
    page_title="AudioMind AI | Real-Time Summarization",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",   # collapsed — all config is inline now
)

css_path = Path(__file__).parent / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

from pipeline.audio_extractor import extract_audio
from pipeline.transcriber import transcribe_audio
from pipeline.text_processor import clean_and_chunk_text
from pipeline.summarizer import summarize_chunks
from pipeline.storage import save_to_db, search_past_sessions
from utils.helpers import format_timestamp, export_results

# ─── Session State ─────────────────────────────────────────────────────────────
for key, default in {
    "transcript": None,
    "summary_result": None,
    "processing": False,
    "stage": 0,
    "chunks": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:1.5rem 0 0.5rem;'>
    <div style='display:inline-block;background:linear-gradient(135deg,#7C3AED,#4F46E5);
                color:white;padding:4px 14px;border-radius:20px;font-size:0.8rem;
                font-weight:700;letter-spacing:1px;margin-bottom:0.6rem;'>
        PROJECT 9
    </div>
    <h1 style='font-size:2.4rem;font-weight:900;letter-spacing:-1px;margin:0;line-height:1.1;'>
        Real-Time Video/Audio
        <span style='background:linear-gradient(135deg,#7C3AED,#06B6D4);
                     -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
            Summarization Pipeline
        </span>
    </h1>
    <p style='font-size:1rem;color:#6B7280;margin-top:0.4rem;'>
        Upload any video or audio → get transcript, insights &amp; action items in real time
    </p>
</div>
""", unsafe_allow_html=True)

# ─── Pipeline Progress Bar ─────────────────────────────────────────────────────
stage_labels = [("📁","Upload"),("🎵","Extract"),("🎤","Transcribe"),("🧹","Clean"),("🤖","Summarize"),("✅","Done")]
pcols = st.columns(len(stage_labels))
for i,(col,(icon,label)) in enumerate(zip(pcols, stage_labels)):
    done   = i < st.session_state.stage
    active = i == st.session_state.stage and st.session_state.processing
    bg = ("background:linear-gradient(135deg,#7C3AED,#4F46E5);color:white;" if done else
          "background:#FEF3C7;color:#92400E;border:2px solid #F59E0B;" if active else
          "background:#F3F4F6;color:#9CA3AF;")
    col.markdown(f"<div style='text-align:center;padding:0.5rem 0.2rem;border-radius:10px;{bg}font-size:0.78rem;font-weight:600;'>{icon} {label}</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Config Row (inline, always visible) ───────────────────────────────────────
st.markdown("### ⚙️ Configuration")
cfg1, cfg2, cfg3, cfg4 = st.columns([1.2, 1, 1, 1])

with cfg1:
    # Read env keys
    _env_openai = os.getenv("OPENAI_API_KEY","").strip()
    _env_google  = os.getenv("GOOGLE_API_KEY","").strip()

    openai_key = st.text_input(
        "🔑 OpenAI API Key",
        type="password",
        value=_env_openai,
        placeholder="sk-proj-...",
    )
    google_key = st.text_input(
        "🔑 Google API Key",
        type="password",
        value=_env_google,
        placeholder="AIza...",
    )

with cfg2:
    _has_openai = bool((openai_key or _env_openai).strip())
    _has_google  = bool((google_key or _env_google).strip())
    _default_model = "gpt-4o" if _has_openai else ("gemini-2.5-pro" if _has_google else "gpt-4o")
    _model_options = ["gpt-4o", "gpt-4o-mini", "gemini-2.5-pro"]
    model_choice = st.selectbox(
        "🤖 LLM Model",
        _model_options,
        index=_model_options.index(_default_model),
        help="Auto-selected based on available API keys",
    )
    whisper_model = st.selectbox(
        "🎙️ Whisper Model",
        ["tiny", "base", "small", "medium", "large"],
        index=2,
        help="Larger = more accurate but slower",
    )

with cfg3:
    language = st.selectbox("🌐 Language", ["auto","en","hi","es","fr","de","zh","ja"])
    chunk_size = st.slider("📦 Chunk Size", 500, 2000, 1000, 100)

with cfg4:
    mongo_uri = st.text_input(
        "🗄️ MongoDB URI (optional)",
        type="password",
        value=os.getenv("MONGO_URI",""),
        placeholder="mongodb+srv://...",
    )

    # Key status indicators
    st.markdown("<div style='margin-top:0.5rem;'>", unsafe_allow_html=True)
    if (openai_key or _env_openai).strip():
        st.success("✅ OpenAI key loaded")
    else:
        st.warning("⚠️ No OpenAI key")
    if (google_key or _env_google).strip():
        st.success("✅ Google key loaded")
    else:
        st.info("ℹ️ No Google key")
    st.markdown("</div>", unsafe_allow_html=True)

# Commit keys to env + session_state
_final_openai = (openai_key or _env_openai).strip()
_final_google  = (google_key  or _env_google).strip()
if _final_openai:
    os.environ["OPENAI_API_KEY"] = _final_openai
    st.session_state["_openai_key"] = _final_openai
if _final_google:
    os.environ["GOOGLE_API_KEY"] = _final_google
    st.session_state["_google_key"] = _final_google
if mongo_uri:
    os.environ["MONGO_URI"] = mongo_uri

st.divider()

# ─── Input Section ─────────────────────────────────────────────────────────────
st.markdown("### 📥 Input")
col_upload, col_url = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown("**📁 Upload File**")
    uploaded_file = st.file_uploader(
        "Upload audio or video",
        type=["mp3","mp4","wav","m4a","ogg","webm","avi","mkv","mov"],
        label_visibility="collapsed",
    )
    if uploaded_file:
        st.success(f"📎 **{uploaded_file.name}** ({uploaded_file.size/1024/1024:.1f} MB)")

with col_url:
    st.markdown("**🔗 Or Paste URL**")
    url_input = st.text_input(
        "URL",
        placeholder="https://youtube.com/watch?v=... or direct audio URL",
        label_visibility="collapsed",
    )
    if url_input:
        st.info(f"🔗 `{url_input[:70]}{'...' if len(url_input)>70 else ''}`")

st.markdown("<br>", unsafe_allow_html=True)

# ─── Action Buttons ────────────────────────────────────────────────────────────
btn1, btn2, btn3 = st.columns([2,1,1])
with btn1:
    process_btn = st.button("🚀 Start Full Pipeline", use_container_width=True, type="primary")
with btn2:
    if st.button("🗑️ Clear Results", use_container_width=True):
        st.session_state.transcript     = None
        st.session_state.summary_result = None
        st.session_state.stage          = 0
        st.rerun()
with btn3:
    demo_btn = st.button("🎭 Demo Mode", use_container_width=True, help="No API key needed")

# ─── Demo Mode ─────────────────────────────────────────────────────────────────
if demo_btn:
    st.session_state.transcript = {
        "text": "Welcome to this podcast about artificial intelligence. Today we discuss the future of AI in healthcare. Key topics include machine learning diagnostics, drug discovery, and personalized medicine. Our expert panel believes AI will reduce diagnostic errors by 30% over the next decade. Action items: invest in AI training for medical staff, partner with biotech startups, and run a pilot program in Q3.",
        "segments": [
            {"start":0.0,  "end":5.2,  "text":"Welcome to this podcast about artificial intelligence."},
            {"start":5.2,  "end":11.0, "text":"Today we discuss the future of AI in healthcare."},
            {"start":11.0, "end":20.0, "text":"Key topics include machine learning diagnostics, drug discovery, and personalized medicine."},
            {"start":20.0, "end":32.0, "text":"Our expert panel believes AI will reduce diagnostic errors by 30% over the next decade."},
            {"start":32.0, "end":45.0, "text":"Action items: invest in AI training, partner with biotech startups, and run a Q3 pilot."},
        ],
        "engine": "demo",
    }
    st.session_state.summary_result = {
        "summary": "This podcast explores the transformative role of AI in healthcare, covering diagnostic ML models, AI-driven drug discovery, and personalized treatment plans.",
        "key_points": [
            "AI diagnostics can reduce errors by up to 30% in the next decade",
            "Drug discovery pipelines are being accelerated using generative AI",
            "Personalized medicine enabled through patient data AI analysis",
            "Medical staff training on AI tools is critical for adoption",
            "Biotech partnerships are the fastest path to implementation",
        ],
        "action_items": [
            "Invest in AI training programs for medical staff",
            "Partner with biotech startups for faster AI integration",
            "Run a pilot program for AI diagnostics in Q3",
            "Evaluate AI tools for patient record analysis",
        ],
        "timestamps": [
            {"time":"0:00","topic":"Introduction to AI in Healthcare"},
            {"time":"0:11","topic":"ML Diagnostics & Drug Discovery"},
            {"time":"0:32","topic":"Action Items & Next Steps"},
        ],
    }
    st.session_state.chunks = ["demo_chunk"]
    st.session_state.stage  = 5
    st.success("✅ Demo mode loaded! Scroll down to see results.")

# ─── Pipeline Execution ────────────────────────────────────────────────────────
if process_btn and (uploaded_file or url_input):

    # Key resolution (session_state always wins over env)
    openai_key_val = st.session_state.get("_openai_key") or os.getenv("OPENAI_API_KEY","")
    google_key_val  = st.session_state.get("_google_key")  or os.getenv("GOOGLE_API_KEY","")

    # Validate at least one key exists for the chosen model
    if "gpt" in model_choice and not openai_key_val.strip():
        if google_key_val.strip():
            st.warning("⚠️ No OpenAI key — switching to Gemini 1.5 Pro automatically.")
            model_choice = "gemini-2.5-pro"
        else:
            st.error("❌ No API key found. Enter your OpenAI or Google key above.")
            st.stop()
    if "gemini" in model_choice and not google_key_val.strip():
        if openai_key_val.strip():
            st.warning("⚠️ No Google key — switching to GPT-4o-mini automatically.")
            model_choice = "gpt-4o-mini"
        else:
            st.error("❌ No API key found. Enter your OpenAI or Google key above.")
            st.stop()

    st.session_state.processing     = True
    st.session_state.transcript     = None
    st.session_state.summary_result = None
    st.session_state.stage          = 0

    tmp_dir = Path(__file__).parent / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    if uploaded_file:
        input_path  = tmp_dir / uploaded_file.name
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        source_name = uploaded_file.name
    else:
        input_path  = url_input
        source_name = url_input

    # Stage 1 — Extract Audio
    st.session_state.stage = 1
    with st.status("🎵 Extracting audio...", expanded=True) as status:
        st.write("Running FFmpeg / MoviePy / yt-dlp...")
        audio_path = extract_audio(input_path, tmp_dir)
        if audio_path:
            st.write(f"✅ Audio ready: `{Path(str(audio_path)).name}`")
            status.update(label="✅ Audio extracted!", state="complete")
        else:
            audio_path = str(input_path)
            status.update(label="⚠️ Using original file", state="complete")

    # Stage 2 — Transcribe
    st.session_state.stage = 2
    with st.status("🎤 Transcribing with Whisper...", expanded=True) as status:
        st.write(f"Model: **{whisper_model}** | Language: **{language}**")
        lang_opt = None if language == "auto" else language
        transcript_data = transcribe_audio(str(audio_path), model_size=whisper_model, language=lang_opt)
        if transcript_data:
            st.session_state.transcript = transcript_data
            wc  = len(transcript_data["text"].split())
            sc  = len(transcript_data.get("segments",[]))
            eng = transcript_data.get("engine","unknown")
            st.write(f"✅ {wc:,} words | {sc} segments | engine: **{eng}**")
            status.update(label=f"✅ Transcription complete! ({eng})", state="complete")
        else:
            status.update(label="❌ Transcription failed", state="error")
            st.error("No STT engine found. Install one: `pip install openai-whisper` + `brew install ffmpeg`")
            st.session_state.processing = False
            st.stop()

    # Stage 3 — Clean & Chunk
    st.session_state.stage = 3
    with st.status("🧹 Cleaning & chunking text...", expanded=True) as status:
        chunks = clean_and_chunk_text(transcript_data["text"], chunk_size=chunk_size)
        st.session_state.chunks = chunks
        st.write(f"✅ {len(chunks)} chunks ready for LLM")
        status.update(label=f"✅ {len(chunks)} chunks created!", state="complete")

    # Stage 4 — Summarize
    st.session_state.stage = 4
    with st.status("🤖 Running LLM Summarization...", expanded=True) as status:
        st.write(f"Model: **{model_choice}** | LangChain map→reduce pipeline")

        api_key = openai_key_val if "gpt" in model_choice else google_key_val
        result  = summarize_chunks(chunks, model_name=model_choice, api_key=api_key)

        if result:
            st.session_state.summary_result = result
            st.write(f"✅ Summary generated")
            st.write(f"✅ {len(result.get('key_points',[]))} key insights | {len(result.get('action_items',[]))} action items")
            status.update(label="✅ Summarization complete!", state="complete")
        else:
            status.update(label="❌ Summarization failed", state="error")
            st.error("LLM failed. Check your API key and selected model.")
            st.session_state.processing = False
            st.stop()

    # Save
    with st.status("💾 Saving...", expanded=False) as status:
        save_to_db({"filename":source_name,"transcript":transcript_data["text"],
                    "summary":result,"segments":transcript_data.get("segments",[])})
        status.update(label="✅ Saved!", state="complete")

    st.session_state.stage      = 5
    st.session_state.processing = False
    st.balloons()
    st.success("🎉 Pipeline complete! Scroll down for results.")

elif process_btn:
    st.warning("⚠️ Please upload a file OR enter a URL first.")

# ─── Results ───────────────────────────────────────────────────────────────────
if st.session_state.transcript and st.session_state.summary_result:
    st.divider()
    st.markdown("## 📊 Results")

    result          = st.session_state.summary_result
    transcript_data = st.session_state.transcript

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("📝 Words",      f"{len(transcript_data['text'].split()):,}")
    m2.metric("🔢 Chunks",     len(st.session_state.chunks))
    m3.metric("🎯 Action Items", len(result.get("action_items",[])))
    m4.metric("💡 Key Insights", len(result.get("key_points",[])))

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1,1], gap="large")

    with left:
        st.markdown("### 📜 Transcript")
        segs = transcript_data.get("segments",[])
        if segs:
            html = ""
            for seg in segs[:60]:
                ts   = format_timestamp(seg["start"])
                text = seg["text"].strip()
                html += f"""<div style='padding:6px 0;border-bottom:1px solid #F3F4F6;display:flex;gap:10px;align-items:flex-start;'>
                    <span style='background:#EDE9FE;color:#7C3AED;padding:2px 8px;border-radius:12px;
                                 font-size:0.72rem;font-weight:700;white-space:nowrap;'>{ts}</span>
                    <span style='font-size:0.9rem;color:#374151;line-height:1.5;'>{text}</span></div>"""
            st.markdown(f"<div style='height:480px;overflow-y:auto;padding:10px;background:#FAFAFA;border-radius:12px;border:1px solid #E5E7EB;'>{html}</div>",
                        unsafe_allow_html=True)
            if len(segs) > 60:
                st.caption(f"Showing 60 of {len(segs)} segments.")
        else:
            st.text_area("Full Transcript", transcript_data["text"], height=450)

    with right:
        st.markdown("### 🧠 AI Analysis")

        with st.expander("📋 Summary", expanded=True):
            st.markdown(result.get("summary","—"))

        with st.expander("💡 Key Insights", expanded=True):
            for i, pt in enumerate(result.get("key_points",[]), 1):
                st.markdown(f"<div style='padding:8px 12px;margin:4px 0;background:#F5F3FF;border-left:3px solid #7C3AED;border-radius:6px;font-size:0.9rem;'><strong>{i}.</strong> {pt}</div>",
                            unsafe_allow_html=True)

        with st.expander("✅ Action Items", expanded=True):
            for item in result.get("action_items",[]):
                c1, c2 = st.columns([0.08, 0.92])
                with c1: st.checkbox("", key=f"act_{hash(item)}")
                with c2: st.markdown(f"<span style='font-size:0.9rem;'>{item}</span>", unsafe_allow_html=True)

        with st.expander("🕐 Topic Timestamps"):
            for t in result.get("timestamps",[]):
                st.markdown(f"**`{t.get('time','?')}`** — {t.get('topic','')}")

    st.divider()
    st.markdown("### 📤 Export")
    e1,e2,e3 = st.columns(3)
    with e1:
        st.download_button("📄 Transcript (.txt)", export_results(transcript_data,result,"text"),
                           "transcript.txt","text/plain", use_container_width=True)
    with e2:
        st.download_button("📊 Full JSON", export_results(transcript_data,result,"json"),
                           "summary.json","application/json", use_container_width=True)
    with e3:
        st.download_button("📝 Report (.md)", export_results(transcript_data,result,"markdown"),
                           "report.md","text/markdown", use_container_width=True)

# ─── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""<div style='text-align:center;color:#9CA3AF;font-size:0.8rem;padding:0.5rem 0;'>
    Built with ❤️ · <strong>WhisperX · LangChain · GPT-4o / Gemini · MongoDB · Streamlit</strong>
</div>""", unsafe_allow_html=True)