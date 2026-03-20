from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(slots=True)
class PipelineConfig:
    input_dir: Path
    hf_token: str | None = None
    language: str = "es"
    model_size: str = "small"
    compute_type: str = "int8"
    device: str = "cpu"
    sample_rate: int = 16000

    @property
    def audio_file(self) -> Path:
        matches = list(self.input_dir.glob("*.mp3")) + list(self.input_dir.glob("*.wav")) + list(self.input_dir.glob("*.m4a"))
        if not matches:
            raise FileNotFoundError(f"No se encontró audio en {self.input_dir}")
        return matches[0]

    @property
    def output_dir(self) -> Path:
        return self.input_dir / "outputs"

    @property
    def full_dir(self) -> Path:
        return self.output_dir / "full"

    @property
    def by_speaker_dir(self) -> Path:
        return self.output_dir / "by_speaker"

    @property
    def voices_dir(self) -> Path:
        return self.output_dir / "voices"

    @property
    def audio_by_speaker_dir(self) -> Path:
        return self.output_dir / "audio_by_speaker"

    @property
    def manifests_dir(self) -> Path:
        return self.output_dir / "manifests"


def load_config(input_dir: str | Path) -> PipelineConfig:
    return PipelineConfig(
        input_dir=Path(input_dir),
        hf_token=os.getenv("HF_TOKEN"),
        language=os.getenv("PIPELINE_LANGUAGE", "es"),
        model_size=os.getenv("WHISPER_MODEL_SIZE", "small"),
        compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
        device=os.getenv("PIPELINE_DEVICE", "cpu"),
    )
