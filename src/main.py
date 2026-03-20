from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from .ffmpeg_utils import convert_to_wav_mono_16k, extract_segment
from .io_utils import ensure_dir, write_json, write_text
from .speech_pipeline import run_transcription, run_pyannote_diarization, merge_transcript_and_diarization
from .srt_utils import build_srt
from .voice_profile import build_voice_profile, profile_to_markdown


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-id", required=True, help="Ejemplo: test001")
    return parser.parse_args()


def build_paths(audio_id: str) -> Dict[str, Path]:
    base = Path("inputs") / audio_id
    input_mp3 = base / f"{audio_id}.mp3"

    source_dir = base / "source"
    outputs_dir = base / "outputs"
    full_dir = outputs_dir / "full"
    by_speaker_dir = outputs_dir / "by_speaker"
    voices_dir = outputs_dir / "voices"
    audio_by_speaker_dir = outputs_dir / "audio_by_speaker"
    manifests_dir = outputs_dir / "manifests"

    wav_path = source_dir / f"{audio_id}.wav"

    return {
        "base": base,
        "input_mp3": input_mp3,
        "source_dir": source_dir,
        "outputs_dir": outputs_dir,
        "full_dir": full_dir,
        "by_speaker_dir": by_speaker_dir,
        "voices_dir": voices_dir,
        "audio_by_speaker_dir": audio_by_speaker_dir,
        "manifests_dir": manifests_dir,
        "wav_path": wav_path,
    }


def main():
    args = parse_args()
    paths = build_paths(args.audio_id)

    ensure_dir(paths["source_dir"])
    ensure_dir(paths["full_dir"])
    ensure_dir(paths["by_speaker_dir"])
    ensure_dir(paths["voices_dir"])
    ensure_dir(paths["audio_by_speaker_dir"])
    ensure_dir(paths["manifests_dir"])

    if not paths["input_mp3"].exists():
        raise FileNotFoundError(f"No existe el audio de entrada: {paths['input_mp3']}")

    print(f"[1/6] Convirtiendo audio a WAV: {paths['input_mp3']}")
    convert_to_wav_mono_16k(paths["input_mp3"], paths["wav_path"])

    print("[2/6] Ejecutando transcripción con faster-whisper")
    transcript = run_transcription(paths["wav_path"], language="es")
    write_json(paths["full_dir"] / "transcript_raw.json", transcript)

    print("[3/6] Ejecutando diarización con pyannote")
    diarization = run_pyannote_diarization(paths["wav_path"])

    print("[4/6] Uniendo transcripción + diarización")
    merged = merge_transcript_and_diarization(transcript, diarization)
    write_json(paths["full_dir"] / "transcript.json", merged)

    full_entries = [
        {
            "start": seg["start"],
            "end": seg["end"],
            "text": f"[{seg['speaker']}] {seg['text']}",
        }
        for seg in merged["segments"]
    ]
    write_text(paths["full_dir"] / "full.srt", build_srt(full_entries))

    print("[5/6] Generando SRT por hablante")
    by_speaker: Dict[str, List[dict]] = {}
    for seg in merged["segments"]:
        by_speaker.setdefault(seg["speaker"], []).append(seg)

    extracted_files_by_speaker: Dict[str, List[Path]] = {}
    segment_manifest = {}

    for speaker, segments in by_speaker.items():
        srt_entries = [
            {"start": s["start"], "end": s["end"], "text": s["text"]}
            for s in segments
        ]
        write_text(paths["by_speaker_dir"] / f"{speaker}.srt", build_srt(srt_entries))

        speaker_audio_dir = paths["audio_by_speaker_dir"] / speaker
        ensure_dir(speaker_audio_dir)

        extracted_files_by_speaker[speaker] = []
        segment_manifest[speaker] = []

        for idx, seg in enumerate(segments, start=1):
            out_wav = speaker_audio_dir / f"{speaker}_{idx:04d}.wav"
            extract_segment(paths["wav_path"], out_wav, float(seg["start"]), float(seg["end"]))
            extracted_files_by_speaker[speaker].append(out_wav)
            segment_manifest[speaker].append({
                "index": idx,
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "audio_file": str(out_wav),
            })

    write_json(paths["manifests_dir"] / "segments_by_speaker.json", segment_manifest)

    print("[6/6] Generando fichas de voz")
    for speaker, segments in by_speaker.items():
        profile = build_voice_profile(
            speaker=speaker,
            segments=segments,
            extracted_audio_files=extracted_files_by_speaker.get(speaker, []),
        )
        write_json(paths["voices_dir"] / f"{speaker}.json", profile)
        write_text(paths["voices_dir"] / f"{speaker}.md", profile_to_markdown(profile))

    print("Proceso completado.")
    print(f"Salidas en: {paths['outputs_dir']}")


if __name__ == "__main__":
    main()
