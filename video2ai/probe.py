"""Extract video metadata using ffprobe."""

import json
import os
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


def _find_bin(name: str) -> str | None:
    """Find a binary by name, checking PATH and common homebrew locations."""
    found = shutil.which(name)
    if found:
        return found
    for prefix in ("/opt/homebrew/bin", "/usr/local/bin"):
        path = os.path.join(prefix, name)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"


def check_ffmpeg() -> None:
    global FFMPEG, FFPROBE
    ff = _find_bin("ffmpeg")
    fp = _find_bin("ffprobe")
    if not ff or not fp:
        raise RuntimeError(
            "ffmpeg/ffprobe not found. Install with: brew install ffmpeg"
        )
    FFMPEG = ff
    FFPROBE = fp


def probe(video_path: str) -> VideoInfo:
    check_ffmpeg()

    cmd = [
        FFPROBE,
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

    # Fallback: webm from MediaRecorder often has no duration in format.
    # Try stream-level duration, then fall back to counting packets.
    if duration <= 0:
        duration = float(video_stream.get("duration", 0))
    if duration <= 0:
        # Last resort: ask ffprobe to read all packets and report duration
        dur_cmd = [
            FFPROBE, "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            "-read_intervals", "%+999999",
            video_path,
        ]
        dur_result = subprocess.run(dur_cmd, capture_output=True, text=True)
        try:
            duration = float(dur_result.stdout.strip())
        except (ValueError, AttributeError):
            pass
    if duration <= 0:
        # Final fallback: use ffmpeg to demux and get duration
        dur_cmd2 = [
            FFMPEG, "-i", video_path,
            "-f", "null", "-",
        ]
        dur_result2 = subprocess.run(dur_cmd2, capture_output=True, text=True)
        # Parse "Duration: HH:MM:SS.ss" or "time=HH:MM:SS.ss" from stderr
        import re
        time_match = re.search(r"time=(\d+):(\d+):(\d[\d.]*)", dur_result2.stderr)
        if time_match:
            h, m, s = time_match.groups()
            duration = int(h) * 3600 + int(m) * 60 + float(s)

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
