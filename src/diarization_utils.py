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


def assign_speakers_to_transcript(transcript_segments: List[Dict], speaker_segments: List[Dict]) -> List[Dict]:
    output = []

    for seg in transcript_segments:
        start = float(seg["start"])
        end = float(seg["end"])
        words = seg.get("words", [])

        speaker = find_speaker_for_interval(start, end, speaker_segments)

        if words:
            speaker_buckets: Dict[str, List[Dict]] = {}
            for word in words:
                ws = float(word["start"])
                we = float(word["end"])
                spk = find_speaker_for_interval(ws, we, speaker_segments)
                speaker_buckets.setdefault(spk, []).append(word)

            if len(speaker_buckets) == 1:
                only_speaker = next(iter(speaker_buckets.keys()))
                output.append({
                    "start": start,
                    "end": end,
                    "text": seg["text"].strip(),
                    "speaker": only_speaker,
                    "words": words,
                })
            else:
                for spk, spk_words in speaker_buckets.items():
                    spk_words = sorted(spk_words, key=lambda w: w["start"])
                    text = " ".join(w["word"] for w in spk_words).strip()
                    if not text:
                        continue
                    output.append({
                        "start": float(spk_words[0]["start"]),
                        "end": float(spk_words[-1]["end"]),
                        "text": text,
                        "speaker": spk,
                        "words": spk_words,
                    })
        else:
            output.append({
                "start": start,
                "end": end,
                "text": seg["text"].strip(),
                "speaker": speaker,
                "words": [],
            })

    output.sort(key=lambda x: x["start"])
    return output
