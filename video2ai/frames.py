"""Extract frames from video at fixed intervals."""

import os
import subprocess
from dataclasses import dataclass


def _ffmpeg_bin() -> str:
    from .probe import FFMPEG
    return FFMPEG


@dataclass
class ExtractedFrame:
    index: int
    timestamp: float
    path: str

    @property
    def timestamp_fmt(self) -> str:
        m, s = divmod(int(self.timestamp), 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h:02d}h{m:02d}m{s:02d}s"
        return f"{m:02d}m{s:02d}s"


def extract_frames(
    video_path: str,
    output_dir: str,
    duration: float,
    interval: float = 1.0,
    max_width: int = 1280,
    quality: int = 85,
    on_frame: "callable | None" = None,
) -> list[ExtractedFrame]:
    """
    Extract frames from video at fixed intervals.

    Default: 1 frame per second. Every frame is kept — key frame selection
    happens later via LLM analysis.
    """
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    frames = []
    idx = 0
    t = 0.0

    while t < duration:
        idx += 1
        frame = _grab_frame(video_path, t, frames_dir, idx, max_width, quality)
        if frame:
            frames.append(frame)
            if on_frame:
                on_frame(frame, t / duration if duration > 0 else 1.0)
        t += interval

    return frames


def _grab_frame(
    video_path: str,
    timestamp: float,
    frames_dir: str,
    index: int,
    max_width: int,
    quality: int,
) -> ExtractedFrame | None:
    """Grab a single frame at the given timestamp."""
    frame = ExtractedFrame(index=index, timestamp=timestamp, path="")
    path = os.path.join(frames_dir, f"frame_{index:03d}_{frame.timestamp_fmt}.jpg")

    if _ffmpeg_grab(video_path, timestamp, path, max_width, quality):
        return ExtractedFrame(index=index, timestamp=timestamp, path=path)
    return None


def _ffmpeg_grab(
    video_path: str, timestamp: float, output_path: str, max_width: int, quality: int
) -> bool:
    """Use ffmpeg to extract a single frame."""
    scale = f"scale='min({max_width},iw):-2'"
    cmd = [
        _ffmpeg_bin(),
        "-ss", str(timestamp),
        "-i", video_path,
        "-vframes", "1",
        "-vf", scale,
        "-q:v", str(max(1, min(31, (100 - quality) * 31 // 100 + 1))),
        "-y",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and os.path.exists(output_path)
