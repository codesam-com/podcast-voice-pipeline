from __future__ import annotations

from typing import Any, Dict, List


def diarization_to_segments(diarization_result: Any) -> List[Dict]:
    segments = []
    for segment, _, speaker in diarization_result.itertracks(yield_label=True):
        segments.append({
            "start": float(segment.start),
            "end": float(segment.end),
            "speaker": str(speaker),
        })
    segments.sort(key=lambda x: x["start"])
    return segments


def find_speaker_for_interval(start: float, end: float, speaker_segments: List[Dict]) -> str:
    best_speaker = "UNKNOWN"
    best_overlap = 0.0

    for seg in speaker_segments:
        overlap = min(end, seg["end"]) - max(start, seg["start"])
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = seg["speaker"]

    return best_speaker


def assign_speakers_to_transcript(aligned_segments: List[Dict], speaker_segments: List[Dict]) -> List[Dict]:
    output = []
    for seg in aligned_segments:
        start = float(seg["start"])
        end = float(seg["end"])
        speaker = find_speaker_for_interval(start, end, speaker_segments)

        item = {
            "start": start,
            "end": end,
            "text": seg["text"].strip(),
            "speaker": speaker,
            "words": seg.get("words", []),
        }
        output.append(item)

    return output


def split_by_speaker(transcript_segments: List[Dict]) -> Dict[str, List[Dict]]:
    by_speaker: Dict[str, List[Dict]] = {}
    for seg in transcript_segments:
        speaker = seg["speaker"]
        by_speaker.setdefault(speaker, []).append(seg)
    return by_speaker
