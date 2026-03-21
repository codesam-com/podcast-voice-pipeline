from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

from .diarization_utils import diarization_to_segments, assign_speakers_to_transcript


def run_transcription(audio_wav: Path, language: str = "es") -> Dict[str, Any]:
    model = WhisperModel(
        "small",
        device="cpu",
        compute_type="int8",
    )

    segments, info = model.transcribe(
        str(audio_wav),
        language=language,
        vad_filter=True,
        word_timestamps=True,
        beam_size=5,
    )

    out_segments: List[Dict[str, Any]] = []

    for seg in segments:
        words = []
        if seg.words:
            for w in seg.words:
                if w.start is None or w.end is None:
                    continue
                words.append({
                    "start": float(w.start),
                    "end": float(w.end),
                    "word": (w.word or "").strip(),
                    "probability": float(w.probability) if w.probability is not None else None,
                })

        out_segments.append({
            "id": len(out_segments),
            "start": float(seg.start),
            "end": float(seg.end),
            "text": (seg.text or "").strip(),
            "words": words,
        })

    return {
        "language": language if language else getattr(info, "language", None),
        "segments": out_segments,
    }


def run_pyannote_diarization(audio_wav: Path) -> Any:
    pipeline_dir = os.environ.get("PYANNOTE_PIPELINE_DIR")
    if not pipeline_dir:
        raise RuntimeError(
            "Falta PYANNOTE_PIPELINE_DIR. El workflow debe descargar "
            "pyannote/speaker-diarization-community-1 en local."
        )

    pipeline_path = Path(pipeline_dir)
    if not pipeline_path.exists():
        raise RuntimeError(f"No existe el pipeline local de pyannote en: {pipeline_path}")

    pipeline = Pipeline.from_pretrained(str(pipeline_path))
    return pipeline(str(audio_wav))


def merge_transcript_and_diarization(transcript: Dict[str, Any], diarization_result: Any) -> Dict[str, Any]:
    speaker_segments = diarization_to_segments(diarization_result)
    transcript_segments = transcript.get("segments", [])
    merged = assign_speakers_to_transcript(transcript_segments, speaker_segments)

    return {
        "language": transcript.get("language"),
        "speaker_segments": speaker_segments,
        "segments": merged,
    }
