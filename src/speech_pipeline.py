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
    hf_token = os.environ.get("PYANNOTE_HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "Falta PYANNOTE_HF_TOKEN. Añade el secret HF_TOKEN en GitHub Actions."
        )

    pipeline = None
    load_errors = []

    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=hf_token,
        )
    except TypeError as exc:
        load_errors.append(f"token=: {exc}")
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1",
                use_auth_token=hf_token,
            )
        except Exception as exc2:
            load_errors.append(f"use_auth_token=: {exc2}")
    except Exception as exc:
        load_errors.append(f"token=: {exc}")

    if pipeline is None:
        details = "\n".join(load_errors) if load_errors else "sin detalle adicional"
        raise RuntimeError(
            "No se pudo cargar pyannote/speaker-diarization-community-1.\n"
            "Comprueba estas 3 cosas:\n"
            "1) el token de Hugging Face no está expirado,\n"
            "2) el usuario del token ha aceptado las condiciones del modelo gated,\n"
            "3) el secret HF_TOKEN del repo contiene ese token correcto.\n\n"
            f"Detalles capturados:\n{details}"
        )

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
