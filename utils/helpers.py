"""
Utility helpers for AudioMind AI pipeline.
"""

import json
from typing import Dict, Any, Optional


def format_timestamp(seconds: float) -> str:
    """Convert float seconds to MM:SS or HH:MM:SS string."""
    if seconds is None:
        return "0:00"
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def export_results(
    transcript_data: Dict[str, Any],
    summary_result: Dict[str, Any],
    format: str = "text"
) -> str:
    """
    Export transcript and summary in various formats.

    Args:
        transcript_data: Dict with 'text' and 'segments'
        summary_result: Dict with 'summary', 'key_points', 'action_items', 'timestamps'
        format: 'text', 'json', or 'markdown'

    Returns:
        Formatted string ready for download
    """
    if format == "json":
        return _export_json(transcript_data, summary_result)
    elif format == "markdown":
        return _export_markdown(transcript_data, summary_result)
    else:
        return _export_text(transcript_data, summary_result)


def _export_text(transcript_data: Dict, summary: Dict) -> str:
    lines = ["=" * 60, "AUDIOMIND AI — TRANSCRIPT & SUMMARY REPORT", "=" * 60, ""]

    lines += ["SUMMARY", "-------", summary.get("summary", ""), ""]

    kp = summary.get("key_points", [])
    if kp:
        lines += ["KEY INSIGHTS", "-----------"]
        for i, p in enumerate(kp, 1):
            lines.append(f"{i}. {p}")
        lines.append("")

    ai = summary.get("action_items", [])
    if ai:
        lines += ["ACTION ITEMS", "------------"]
        for item in ai:
            lines.append(f"[ ] {item}")
        lines.append("")

    ts = summary.get("timestamps", [])
    if ts:
        lines += ["TOPIC TIMESTAMPS", "----------------"]
        for t in ts:
            lines.append(f"{t.get('time','?')}  {t.get('topic','')}")
        lines.append("")

    lines += ["FULL TRANSCRIPT", "---------------", transcript_data.get("text", "")]

    return "\n".join(lines)


def _export_markdown(transcript_data: Dict, summary: Dict) -> str:
    lines = ["# AudioMind AI — Report", ""]
    lines += ["## 📋 Summary", "", summary.get("summary", ""), ""]

    kp = summary.get("key_points", [])
    if kp:
        lines += ["## 💡 Key Insights", ""]
        for p in kp:
            lines.append(f"- {p}")
        lines.append("")

    ai = summary.get("action_items", [])
    if ai:
        lines += ["## ✅ Action Items", ""]
        for item in ai:
            lines.append(f"- [ ] {item}")
        lines.append("")

    ts = summary.get("timestamps", [])
    if ts:
        lines += ["## 🕐 Topic Timestamps", "", "| Time | Topic |", "|------|-------|"]
        for t in ts:
            lines.append(f"| {t.get('time','?')} | {t.get('topic','')} |")
        lines.append("")

    segs = transcript_data.get("segments", [])
    lines += ["## 📜 Full Transcript", ""]
    if segs:
        for seg in segs:
            ts_str = format_timestamp(seg.get("start", 0))
            speaker = f"**{seg.get('speaker', '')}** " if seg.get("speaker") else ""
            lines.append(f"`{ts_str}` {speaker}{seg.get('text','').strip()}")
            lines.append("")
    else:
        lines.append(transcript_data.get("text", ""))

    return "\n".join(lines)


def _export_json(transcript_data: Dict, summary: Dict) -> str:
    output = {
        "summary": summary,
        "transcript": {
            "full_text": transcript_data.get("text", ""),
            "language": transcript_data.get("language", "unknown"),
            "engine": transcript_data.get("engine", "whisper"),
            "segments": transcript_data.get("segments", []),
        }
    }
    return json.dumps(output, indent=2, ensure_ascii=False)
