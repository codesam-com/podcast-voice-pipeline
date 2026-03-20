from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import torchaudio

if not hasattr(torchaudio, "AudioMetaData"):
    raise RuntimeError(
        "La versión instalada de torchaudio no es compatible con pyannote.audio. "
        "Fija torch/torchaudio a 2.3.1 en el workflow."
    )

try:
    import transformers  # noqa: F401
except ImportError as exc:
    raise RuntimeError(
        "Falta la dependencia 'transformers'."
    ) from exc

try:
    import nltk  # noqa: F401
except ImportError as exc:
    raise RuntimeError(
        "Falta la dependencia 'nltk'. Añádela al workflow."
    ) from exc

import whisperx
from pyannote.audio import Pipeline

from .diarization_utils import diarization_to_segments, assign_speakers_to_transcript


def run_whisperx_transcription(audio_wav: Path, language: str = "es") -> Dict[str, Any]:
    device = "cpu"
    compute_type = "int8"

    model = whisperx.load_model(
        "small",
        device=device,
        compute_type=compute_type,
        language=language,
    )

    audio = whisperx.load_audio(str(audio_wav))
    result = model.transcribe(audio, batch_size=4)

    align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
    aligned = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )
    return aligned


def run_pyannote_diarization(audio_wav: Path) -> Any:
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("Missing HF_TOKEN environment variable.")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=hf_token,
    )
    diarization = pipeline(str(audio_wav))
    return diarization


def merge_transcript_and_diarization(aligned_transcript: Dict[str, Any], diarization_result: Any) -> Dict[str, Any]:
    speaker_segments = diarization_to_segments(diarization_result)
    transcript_segments = aligned_transcript.get("segments", [])

    merged = assign_speakers_to_transcript(transcript_segments, speaker_segments)

    return {
        "language": aligned_transcript.get("language"),
        "speaker_segments": speaker_segments,
        "segments": merged,
    }
