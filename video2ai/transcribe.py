"""Audio transcription using local Whisper model."""

import os
import subprocess
import tempfile
from dataclasses import dataclass


def _ffmpeg_bin() -> str:
    from .probe import FFMPEG
    return FFMPEG


@dataclass
class Segment:
    start: float
    end: float
    text: str


def transcribe(
    video_path: str,
    model_name: str = "base",
) -> list[Segment]:
    """Extract audio from video and transcribe with Whisper."""
    # Extract audio to temp WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = tmp.name

    try:
        cmd = [
            _ffmpeg_bin(),
            "-i", video_path,
            "-vn",  # no video
            "-acodec", "pcm_s16le",
            "-ar", "16000",  # 16kHz for Whisper
            "-ac", "1",  # mono
            "-y",
            audio_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Warning: audio extraction failed: {result.stderr[:200]}")
            return []

        return _run_whisper(audio_path, model_name)
    finally:
        if os.path.exists(audio_path):
            os.unlink(audio_path)


def _run_whisper(audio_path: str, model_name: str) -> list[Segment]:
    """Run Whisper transcription on an audio file."""
    try:
        import whisper
    except ImportError:
        print(
            "Warning: openai-whisper not installed. "
            "Install with: pip install openai-whisper"
        )
        return []

    print(f"Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name)

    print("Transcribing audio...")
    result = model.transcribe(audio_path, verbose=False)

    segments = []
    for seg in result.get("segments", []):
        segments.append(
            Segment(
                start=round(seg["start"], 2),
                end=round(seg["end"], 2),
                text=seg["text"].strip(),
            )
        )

    return segments
