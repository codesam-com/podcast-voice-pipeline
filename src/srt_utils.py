from __future__ import annotations

from typing import List, Dict, Any


def seconds_to_srt_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    ms = int(round((seconds - int(seconds)) * 1000))
    total = int(seconds)
    s = total % 60
    m = (total // 60) % 60
    h = total // 3600
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def build_srt(entries: List[Dict[str, Any]]) -> str:
    lines = []
    for idx, item in enumerate(entries, start=1):
        start = seconds_to_srt_time(float(item["start"]))
        end = seconds_to_srt_time(float(item["end"]))
        text = item["text"].strip()
        lines.append(str(idx))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)
