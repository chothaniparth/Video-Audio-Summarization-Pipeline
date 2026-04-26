"""
Stage 2: Audio Extraction
Uses FFmpeg (via moviepy/pydub) to extract audio from video files or download from URLs.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional, Union


def extract_audio(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    output_format: str = "wav"
) -> Optional[str]:
    """
    Extract audio from a video file or download from a URL.

    Args:
        input_path: Path to video file OR a URL string
        output_dir: Directory to save the extracted audio
        output_format: Output audio format ('wav', 'mp3', 'ogg')

    Returns:
        Path to extracted audio file, or None on failure
    """
    input_path = str(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # If input is already an audio file, return as-is
    audio_extensions = {".mp3", ".wav", ".ogg", ".m4a", ".flac", ".aac"}
    if not input_path.startswith("http") and Path(input_path).suffix.lower() in audio_extensions:
        print(f"[AudioExtractor] Input is already audio: {input_path}")
        return input_path

    # Handle URL input
    if input_path.startswith("http"):
        return _download_audio_from_url(input_path, output_dir, output_format)

    # Handle video file
    return _extract_from_video_file(input_path, output_dir, output_format)


def _extract_from_video_file(
    video_path: str,
    output_dir: Path,
    output_format: str
) -> Optional[str]:
    """Extract audio track from a video file using FFmpeg."""
    stem = Path(video_path).stem
    out_path = output_dir / f"{stem}_audio.{output_format}"

    # Try FFmpeg directly first (fastest)
    if _ffmpeg_available():
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vn",                        # no video
                "-acodec", "pcm_s16le",       # WAV codec
                "-ar", "16000",               # 16kHz sample rate (Whisper prefers this)
                "-ac", "1",                   # mono channel
                str(out_path)
            ]
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=300
            )
            if result.returncode == 0 and out_path.exists():
                print(f"[AudioExtractor] FFmpeg extracted: {out_path}")
                return str(out_path)
            else:
                print(f"[AudioExtractor] FFmpeg error: {result.stderr.decode()[:200]}")
        except Exception as e:
            print(f"[AudioExtractor] FFmpeg failed: {e}")

    # Fallback: Try moviepy
    try:
        from moviepy.editor import VideoFileClip
        clip = VideoFileClip(video_path)
        if clip.audio:
            clip.audio.write_audiofile(str(out_path), fps=16000, nbytes=2, codec="pcm_s16le",
                                       verbose=False, logger=None)
            clip.close()
            print(f"[AudioExtractor] MoviePy extracted: {out_path}")
            return str(out_path)
        else:
            print("[AudioExtractor] No audio track found in video file.")
            clip.close()
            return None
    except ImportError:
        print("[AudioExtractor] MoviePy not installed. Install: pip install moviepy")
    except Exception as e:
        print(f"[AudioExtractor] MoviePy failed: {e}")

    return None


def _download_audio_from_url(url: str, output_dir: Path, output_format: str) -> Optional[str]:
    """Download audio from a YouTube or direct URL using yt-dlp."""
    try:
        import yt_dlp

        out_template = str(output_dir / "%(title)s.%(ext)s")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": out_template,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": output_format,
                "preferredquality": "192",
            }],
            "quiet": True,
            "no_warnings": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get("title", "audio").replace("/", "-")
            expected = output_dir / f"{title}.{output_format}"
            if expected.exists():
                print(f"[AudioExtractor] Downloaded: {expected}")
                return str(expected)

            # Search for any file just created
            files = sorted(output_dir.glob(f"*.{output_format}"), key=lambda p: p.stat().st_mtime, reverse=True)
            if files:
                return str(files[0])

    except ImportError:
        print("[AudioExtractor] yt-dlp not installed. Install: pip install yt-dlp")
    except Exception as e:
        print(f"[AudioExtractor] URL download failed: {e}")

    return None


def _ffmpeg_available() -> bool:
    """Check if FFmpeg is installed and accessible."""
    try:
        result = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    except FileNotFoundError:
        return False
