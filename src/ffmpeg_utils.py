from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable


def run_cmd(cmd: Iterable[str]) -> None:
    process = subprocess.run(list(cmd), capture_output=True, text=True)
    if process.returncode != 0:
      raise RuntimeError(
          f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}"
      )


def convert_to_wav_mono_16k(input_audio: Path, output_wav: Path) -> None:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    run_cmd([
        "ffmpeg", "-y",
        "-i", str(input_audio),
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        str(output_wav),
    ])


def extract_segment(input_wav: Path, output_wav: Path, start: float, end: float) -> None:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.0, end - start)
    run_cmd([
        "ffmpeg", "-y",
        "-i", str(input_wav),
        "-ss", f"{start:.3f}",
        "-t", f"{duration:.3f}",
        "-acodec", "pcm_s16le",
        str(output_wav),
    ])


def probe_duration(input_audio: Path) -> float:
    process = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(input_audio)
        ],
        capture_output=True,
        text=True
    )
    if process.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {process.stderr}")
    return float(process.stdout.strip())
