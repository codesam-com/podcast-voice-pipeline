from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any

import librosa
import numpy as np
import soundfile as sf


def safe_mean(values: List[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def safe_std(values: List[float]) -> float | None:
    if not values:
        return None
    return float(np.std(values))


def analyze_audio_file(path: Path) -> Dict[str, Any]:
    y, sr = librosa.load(path, sr=16000, mono=True)

    duration_sec = float(len(y) / sr) if sr else 0.0
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = float(np.mean(rms)) if len(rms) else None

    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr
    )

    voiced_f0 = [float(v) for v in f0 if v is not None and not np.isnan(v)]
    speaking_rate_hint = None

    return {
        "duration_sec": round(duration_sec, 3),
        "sample_rate": sr,
        "rms_mean": round(rms_mean, 6) if rms_mean is not None else None,
        "f0_mean_hz": round(safe_mean(voiced_f0), 2) if voiced_f0 else None,
        "f0_std_hz": round(safe_std(voiced_f0), 2) if voiced_f0 else None,
        "voiced_ratio": round(float(np.mean(voiced_flag)) if voiced_flag is not None else 0.0, 4),
        "speaking_rate_hint": speaking_rate_hint,
    }


def build_voice_profile(
    speaker: str,
    segments: List[Dict[str, Any]],
    extracted_audio_files: List[Path],
) -> Dict[str, Any]:
    texts = [seg["text"].strip() for seg in segments if seg["text"].strip()]
    full_text = " ".join(texts)

    word_count = len(full_text.split())
    segment_count = len(segments)
    total_speech_sec = round(sum(float(seg["end"]) - float(seg["start"]) for seg in segments), 3)

    audio_stats = []
    for wav_path in extracted_audio_files:
        try:
            audio_stats.append(analyze_audio_file(wav_path))
        except Exception:
            continue

    f0_means = [x["f0_mean_hz"] for x in audio_stats if x.get("f0_mean_hz") is not None]
    rms_means = [x["rms_mean"] for x in audio_stats if x.get("rms_mean") is not None]
    voiced_ratios = [x["voiced_ratio"] for x in audio_stats if x.get("voiced_ratio") is not None]

    words_per_second = round(word_count / total_speech_sec, 3) if total_speech_sec > 0 else None

    return {
        "speaker_id": speaker,
        "summary": {
            "segment_count": segment_count,
            "word_count": word_count,
            "total_speech_sec": total_speech_sec,
            "words_per_second": words_per_second,
        },
        "acoustic_estimates": {
            "f0_mean_hz_avg": round(float(np.mean(f0_means)), 2) if f0_means else None,
            "f0_mean_hz_std_between_segments": round(float(np.std(f0_means)), 2) if f0_means else None,
            "energy_rms_avg": round(float(np.mean(rms_means)), 6) if rms_means else None,
            "voiced_ratio_avg": round(float(np.mean(voiced_ratios)), 4) if voiced_ratios else None,
        },
        "perceived_traits": {
            "sexo_vocal_percibido": "pendiente de revisión humana",
            "registro_aproximado": "estimación automática incompleta",
            "timbre": "pendiente de revisión humana",
            "energía": "inferida parcialmente por RMS",
            "ritmo": "inferido parcialmente por words_per_second",
            "articulación": "pendiente de revisión humana",
            "acento_variedad": "español de España (supuesto del proyecto; confirmar manualmente)",
            "nasalidad": "pendiente de revisión humana",
            "aspereza_aire": "pendiente de revisión humana",
            "estabilidad": "inferida parcialmente por variación de f0",
            "emotividad": "pendiente de revisión humana",
        },
        "cloning_notes": {
            "recommended_for_training": "usar segmentos limpios, sin solape, sin ruido y mayores de 2 segundos",
            "avoid_for_training": "segmentos con solapamiento, ruido, risas, música, respiraciones marcadas o recorte",
            "consent_verified": True
        },
        "text_sample_preview": full_text[:1000],
        "source_audio_files": [str(p) for p in extracted_audio_files],
    }


def profile_to_markdown(profile: Dict[str, Any]) -> str:
    summary = profile["summary"]
    acoustic = profile["acoustic_estimates"]
    traits = profile["perceived_traits"]
    notes = profile["cloning_notes"]

    md = []
    md.append(f"# {profile['speaker_id']}")
    md.append("")
    md.append("## Resumen")
    md.append(f"- Segmentos: {summary['segment_count']}")
    md.append(f"- Palabras: {summary['word_count']}")
    md.append(f"- Tiempo total hablado: {summary['total_speech_sec']} s")
    md.append(f"- Velocidad aproximada: {summary['words_per_second']} palabras/seg")
    md.append("")
    md.append("## Estimaciones acústicas")
    md.append(f"- F0 media: {acoustic['f0_mean_hz_avg']} Hz")
    md.append(f"- Variabilidad F0 entre segmentos: {acoustic['f0_mean_hz_std_between_segments']} Hz")
    md.append(f"- Energía RMS media: {acoustic['energy_rms_avg']}")
    md.append(f"- Ratio de voz: {acoustic['voiced_ratio_avg']}")
    md.append("")
    md.append("## Rasgos percibidos")
    for k, v in traits.items():
        md.append(f"- {k}: {v}")
    md.append("")
    md.append("## Notas para clonación")
    for k, v in notes.items():
        md.append(f"- {k}: {v}")
    md.append("")
    md.append("## Muestra textual")
    md.append(profile["text_sample_preview"] or "(vacío)")
    md.append("")
    return "\n".join(md)
