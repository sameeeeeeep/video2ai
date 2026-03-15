"""Extract video metadata using ffprobe."""

import json
import shutil
import subprocess
from dataclasses import dataclass


@dataclass
class VideoInfo:
    path: str
    duration: float
    width: int
    height: int
    fps: float
    codec: str
    has_audio: bool
    file_size_mb: float

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"

    @property
    def duration_fmt(self) -> str:
        m, s = divmod(int(self.duration), 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"


def check_ffmpeg() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError(
            "ffmpeg/ffprobe not found. Install with: brew install ffmpeg"
        )


def probe(video_path: str) -> VideoInfo:
    check_ffmpeg()

    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    video_stream = next(
        (s for s in data["streams"] if s["codec_type"] == "video"), None
    )
    if not video_stream:
        raise ValueError(f"No video stream found in {video_path}")

    has_audio = any(s["codec_type"] == "audio" for s in data["streams"])

    # Parse FPS from r_frame_rate (e.g., "30/1" or "30000/1001")
    fps_parts = video_stream.get("r_frame_rate", "30/1").split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 30.0

    duration = float(data["format"].get("duration", 0))
    file_size = float(data["format"].get("size", 0)) / (1024 * 1024)

    return VideoInfo(
        path=video_path,
        duration=duration,
        width=int(video_stream["width"]),
        height=int(video_stream["height"]),
        fps=fps,
        codec=video_stream.get("codec_name", "unknown"),
        has_audio=has_audio,
        file_size_mb=round(file_size, 2),
    )
