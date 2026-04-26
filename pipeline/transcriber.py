"""
Stage 3: Speech-to-Text Transcription
Tries engines in this order (stops at first success):
  1. WhisperX        – word-level timestamps, speaker diarization
  2. openai-whisper  – standard, GPU/CPU
  3. faster-whisper  – lightweight CTranslate2 backend
  4. SpeechRecognition + Google STT  – no model download, online API
  5. Returns None + prints a clear install message
"""

from typing import Optional, Dict, Any, List
import os


def transcribe_audio(
    audio_path: str,
    model_size: str = "small",
    language: Optional[str] = None,
    enable_diarization: bool = False,
    hf_token: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Transcribe an audio file. Tries multiple engines automatically.

    Args:
        audio_path:          Path to audio file (wav/mp3/m4a/ogg ...)
        model_size:          Whisper model size - tiny/base/small/medium/large
        language:            ISO-639-1 code ('en', 'hi' ...) or None for auto
        enable_diarization:  Speaker diarization (WhisperX + HF token only)
        hf_token:            HuggingFace token for pyannote diarization

    Returns:
        {'text', 'segments', 'language', 'engine'} or None
    """
    engines = [
        ("WhisperX",          lambda: _try_whisperx(audio_path, model_size, language, enable_diarization, hf_token)),
        ("openai-whisper",    lambda: _try_openai_whisper(audio_path, model_size, language)),
        ("faster-whisper",    lambda: _try_faster_whisper(audio_path, model_size, language)),
        ("SpeechRecognition", lambda: _try_speech_recognition(audio_path, language)),
    ]

    for name, fn in engines:
        try:
            result = fn()
            if result:
                print(f"[Transcriber] Success with {name}")
                return result
        except ImportError:
            print(f"[Transcriber] {name} not installed — skipping.")
        except Exception as e:
            print(f"[Transcriber] {name} failed: {e}")

    _print_install_help()
    return None


# ─── Engine 1: WhisperX ────────────────────────────────────────────────────────

def _try_whisperx(audio_path, model_size, language, enable_diarization, hf_token):
    import whisperx
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"[Transcriber] WhisperX '{model_size}' on {device} ...")
    model = whisperx.load_model(model_size, device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_path)

    # whisperx 3.x: batch_size is a direct param, language optional
    kwargs = {}
    if language:
        kwargs["language"] = language
    raw = model.transcribe(audio, batch_size=16, **kwargs)
    detected_lang = raw.get("language", language or "en")

    try:
        am, meta = whisperx.load_align_model(language_code=detected_lang, device=device)
        aligned = whisperx.align(raw["segments"], am, meta, audio, device, return_char_alignments=False)
        segments = aligned["segments"]
    except Exception as e:
        print(f"[Transcriber] WhisperX alignment skipped: {e}")
        segments = raw["segments"]

    if enable_diarization and hf_token:
        try:
            dm = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
            dsegs = dm(audio)
            segments = whisperx.assign_word_speakers(dsegs, {"segments": segments})["segments"]
        except Exception as e:
            print(f"[Transcriber] Diarization skipped: {e}")

    return _build_result(
        text=" ".join(s.get("text", "").strip() for s in segments),
        segments=_norm(segments),
        language=detected_lang,
        engine="whisperx",
    )


# ─── Engine 2: openai-whisper ──────────────────────────────────────────────────

def _try_openai_whisper(audio_path, model_size, language):
    import whisper
    import ssl
    import urllib.request

    print(f"[Transcriber] openai-whisper '{model_size}' ...")

    # Fix: corporate/VPN networks often have self-signed certs that block model downloads.
    # Patch urllib to skip SSL verification just for the model download, then restore it.
    original_https = urllib.request.HTTPSHandler
    try:
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_ctx))
        urllib.request.install_opener(opener)
        model = whisper.load_model(model_size)
    finally:
        # Always restore default opener after download attempt
        urllib.request.install_opener(urllib.request.build_opener(original_https()))

    try:
        opts = {"verbose": False}
        if language:
            opts["language"] = language
        result = model.transcribe(audio_path, **opts)
    except TypeError:
        opts.pop("verbose", None)
        result = model.transcribe(audio_path, **opts)

    segments = [
        {"start": s["start"], "end": s["end"], "text": s["text"].strip(), "speaker": None}
        for s in result.get("segments", [])
    ]
    return _build_result(
        text=result["text"].strip(),
        segments=segments,
        language=result.get("language", "unknown"),
        engine="openai-whisper",
    )


# ─── Engine 3: faster-whisper ──────────────────────────────────────────────────

def _try_faster_whisper(audio_path, model_size, language):
    from faster_whisper import WhisperModel

    print(f"[Transcriber] faster-whisper '{model_size}' ...")
    # device="auto" works in faster-whisper>=1.0 (falls back to cpu automatically)
    try:
        model = WhisperModel(model_size, device="auto", compute_type="default")
    except Exception:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")

    # faster-whisper 1.x: pass language=None for auto-detect, not as kwarg if unsupported
    transcribe_kwargs = {"beam_size": 5}
    if language:
        transcribe_kwargs["language"] = language

    segs_gen, info = model.transcribe(audio_path, **transcribe_kwargs)

    segments = []
    full_text_parts = []
    for seg in segs_gen:
        segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
            "speaker": None,
        })
        full_text_parts.append(seg.text.strip())

    if not full_text_parts:
        return None

    return _build_result(
        text=" ".join(full_text_parts),
        segments=segments,
        language=getattr(info, "language", language or "unknown"),
        engine="faster-whisper",
    )


# ─── Engine 4: SpeechRecognition (Google STT, no model download) ───────────────

def _try_speech_recognition(audio_path, language):
    import speech_recognition as sr

    print("[Transcriber] SpeechRecognition + Google STT ...")
    recognizer = sr.Recognizer()

    wav_path = _ensure_wav(audio_path)
    duration = _wav_duration(wav_path)
    chunk_secs = 55  # Google STT limit

    lang_code = _lang_to_bcp47(language)
    all_text = []
    segments = []
    current_start = 0.0

    with sr.AudioFile(wav_path) as source:
        while current_start < duration:
            length = min(chunk_secs, duration - current_start)
            audio_chunk = recognizer.record(source, duration=length)
            try:
                text = recognizer.recognize_google(audio_chunk, language=lang_code)
                all_text.append(text)
                segments.append({
                    "start": current_start,
                    "end": current_start + length,
                    "text": text,
                    "speaker": None,
                })
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                print(f"[Transcriber] Google STT request error: {e}")
                break
            current_start += chunk_secs

    if not all_text:
        return None

    return _build_result(
        text=" ".join(all_text),
        segments=segments,
        language=language or "en",
        engine="SpeechRecognition+Google",
    )


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _build_result(text, segments, language, engine):
    if not text or not text.strip():
        return None
    return {"text": text, "segments": segments, "language": language, "engine": engine}


def _norm(raw_segs: List[Dict]) -> List[Dict]:
    return [
        {
            "start": s.get("start", 0.0),
            "end": s.get("end", 0.0),
            "text": s.get("text", "").strip(),
            "speaker": s.get("speaker"),
        }
        for s in raw_segs
    ]


def _lang_to_bcp47(language: Optional[str]) -> str:
    mapping = {
        "en": "en-US", "hi": "hi-IN", "es": "es-ES",
        "fr": "fr-FR", "de": "de-DE", "zh": "zh-CN",
        "ja": "ja-JP", "ar": "ar-SA", "pt": "pt-BR",
    }
    return mapping.get(language or "en", "en-US")


def _ensure_wav(audio_path: str) -> str:
    """Convert audio to 16kHz mono WAV for SpeechRecognition."""
    if audio_path.lower().endswith(".wav"):
        return audio_path
    out = audio_path.rsplit(".", 1)[0] + "_sr.wav"
    try:
        import subprocess
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", out],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        )
        return out
    except Exception:
        try:
            from pydub import AudioSegment
            seg = AudioSegment.from_file(audio_path).set_frame_rate(16000).set_channels(1)
            seg.export(out, format="wav")
            return out
        except Exception:
            return audio_path


def _wav_duration(wav_path: str) -> float:
    try:
        import wave, contextlib
        with contextlib.closing(wave.open(wav_path, "r")) as f:
            return f.getnframes() / float(f.getframerate())
    except Exception:
        return 60.0


def _print_install_help():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  [Transcriber] No STT engine found!                             ║
║                                                                  ║
║  Install ONE of the following:                                   ║
║                                                                  ║
║  RECOMMENDED — openai-whisper (offline, no GPU required):       ║
║    pip install openai-whisper                                    ║
║    + FFmpeg: brew install ffmpeg  /  sudo apt install ffmpeg     ║
║                                                                  ║
║  LIGHTWEIGHT — faster-whisper (fastest on CPU):                 ║
║    pip install faster-whisper                                    ║
║                                                                  ║
║  ONLINE (no download) — SpeechRecognition + Google:             ║
║    pip install SpeechRecognition pydub                           ║
║                                                                  ║
║  BEST QUALITY — WhisperX (GPU recommended):                     ║
║    pip install whisperx torch torchaudio                         ║
╚══════════════════════════════════════════════════════════════════╝
""")