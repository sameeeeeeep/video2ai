"""Embed analysis metadata into the video file itself.

Creates a copy of the original video with:
- MP4 chapter markers for each section (universally supported)
- Subtitle track with structured analysis data (time-aligned, extractable)
- MP4 metadata tags for summary and high-level info
"""

import json
import os


def _ffmpeg_bin() -> str:
    from .probe import FFMPEG
    return FFMPEG
import subprocess
import tempfile

from .frames import ExtractedFrame
from .llm import LLMAnalysis, VideoSection
from .probe import VideoInfo
from .transcribe import Segment
from .vision import FrameAnalysis


def embed_metadata(
    video_path: str,
    video_info: VideoInfo,
    frames: list[ExtractedFrame],
    segments: list[Segment],
    analyses: list[FrameAnalysis],
    llm_analysis: LLMAnalysis | None,
    output_dir: str,
) -> str | None:
    """Embed analysis metadata into the video and return path to enriched video.

    The enriched video plays identically but carries chapters, a subtitle
    track with structured data, and MP4-level metadata tags so any AI can
    extract the analysis by inspecting the file.
    """
    if not llm_analysis or not llm_analysis.sections:
        print("  No LLM analysis to embed, skipping")
        return None

    analysis_map = {a.frame_index: a for a in analyses}
    key_set = set(llm_analysis.key_frame_indices)

    basename = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{basename}_smart.mp4")

    with tempfile.TemporaryDirectory() as tmp:
        chapters_path = _write_chapters(llm_analysis.sections, tmp)
        subs_path = _write_subtitles(
            llm_analysis, frames, segments, analysis_map, key_set, tmp
        )
        meta_args = _build_metadata_args(video_info, llm_analysis)

        success = _mux_video(
            video_path, output_path, chapters_path, subs_path, meta_args
        )

    if success:
        size_mb = round(os.path.getsize(output_path) / (1024 * 1024), 2)
        print(f"  Smart video: {output_path} ({size_mb}MB)")
        return output_path
    else:
        print("  Warning: Failed to create smart video")
        return None


# ── Chapter markers ──────────────────────────────────────────────────────────


def _write_chapters(sections: list[VideoSection], tmp_dir: str) -> str:
    """Write FFMETADATA1 chapter file."""
    lines = [";FFMETADATA1"]

    for section in sections:
        start_ms = int(section.start_time * 1000)
        end_ms = int(section.end_time * 1000)
        # Escape special chars in title
        title = section.title.replace("=", "\\=").replace(";", "\\;").replace("#", "\\#")

        lines.append("[CHAPTER]")
        lines.append("TIMEBASE=1/1000")
        lines.append(f"START={start_ms}")
        lines.append(f"END={end_ms}")
        lines.append(f"title={title}")

    path = os.path.join(tmp_dir, "chapters.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


# ── Subtitle track with structured data ──────────────────────────────────────


def _write_subtitles(
    llm_analysis: LLMAnalysis,
    frames: list[ExtractedFrame],
    segments: list[Segment],
    analysis_map: dict[int, FrameAnalysis],
    key_set: set[int],
    tmp_dir: str,
) -> str:
    """Write WebVTT subtitle track with structured analysis data.

    Each cue contains JSON with the section info, transcript, and key frames
    for that time range. An AI extracting this subtitle track gets the
    complete analysis time-aligned to the video.
    """
    lines = ["WEBVTT", "Kind: metadata", ""]

    for i, section in enumerate(llm_analysis.sections):
        start_vtt = _fmt_vtt(section.start_time)
        end_vtt = _fmt_vtt(section.end_time)

        # Collect key frames in this section
        section_key_frames = []
        for f in frames:
            if f.index in key_set and section.start_time <= f.timestamp <= section.end_time:
                entry = {
                    "frame_index": f.index,
                    "timestamp": round(f.timestamp, 2),
                }
                if f.index in llm_analysis.frame_reasoning:
                    entry["significance"] = llm_analysis.frame_reasoning[f.index]
                a = analysis_map.get(f.index)
                if a and a.ocr_text:
                    entry["on_screen_text"] = a.ocr_text[:200]
                if a and a.labels:
                    entry["labels"] = a.labels
                section_key_frames.append(entry)

        # Collect transcript in this section
        section_transcript = []
        for seg in segments:
            if seg.start < section.end_time and seg.end > section.start_time:
                section_transcript.append({
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "text": seg.text,
                })

        cue_data = {
            "section": i + 1,
            "title": section.title,
            "topic": section.topic,
            "key_points": section.key_points,
            "key_frames": section_key_frames,
            "transcript": section_transcript,
        }

        # Write cue — JSON payload as subtitle text
        lines.append(f"section-{i + 1}")
        lines.append(f"{start_vtt} --> {end_vtt}")
        lines.append(json.dumps(cue_data, ensure_ascii=False))
        lines.append("")

    path = os.path.join(tmp_dir, "analysis.vtt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


# ── MP4 metadata tags ────────────────────────────────────────────────────────


def _build_metadata_args(
    video_info: VideoInfo,
    llm_analysis: LLMAnalysis,
) -> list[str]:
    """Build ffmpeg metadata arguments for MP4 tags."""
    args = []

    if llm_analysis.summary:
        args.extend(["-metadata", f"description={llm_analysis.summary}"])
        args.extend(["-metadata", f"synopsis={llm_analysis.summary}"])

    args.extend(["-metadata", f"comment=Analyzed by video2ai using {llm_analysis.model}"])
    args.extend(["-metadata", f"encoder=video2ai"])

    # Store section count and key frame count as custom tags
    args.extend(["-metadata", f"video2ai_sections={len(llm_analysis.sections)}"])
    args.extend(["-metadata", f"video2ai_key_frames={len(llm_analysis.key_frame_indices)}"])
    args.extend(["-metadata", f"video2ai_model={llm_analysis.model}"])

    # Store section titles as a compact list
    section_titles = " | ".join(
        f"{s.title} ({_fmt_ts(s.start_time)}-{_fmt_ts(s.end_time)})"
        for s in llm_analysis.sections
    )
    args.extend(["-metadata", f"video2ai_structure={section_titles}"])

    return args


# ── Mux everything together ─────────────────────────────────────────────────


def _mux_video(
    input_path: str,
    output_path: str,
    chapters_path: str,
    subs_path: str,
    meta_args: list[str],
) -> bool:
    """Mux original video + chapters + subtitle track + metadata into new file."""
    cmd = [
        _ffmpeg_bin(),
        "-i", input_path,
        "-i", chapters_path,
        "-i", subs_path,
        "-map", "0",              # All streams from original video
        "-map_chapters", "1",     # Chapters from metadata file
        "-map", "2",              # Subtitle track
        "-c", "copy",             # No re-encoding of video/audio
        "-c:s", "mov_text",       # Subtitle codec for MP4
        "-metadata:s:s:0", "language=eng",
        "-metadata:s:s:0", "title=video2ai analysis",
        *meta_args,
        "-y",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ffmpeg error: {result.stderr[-300:]}")
        return False
    return os.path.exists(output_path)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _fmt_vtt(seconds: float) -> str:
    """Format seconds to WebVTT timestamp HH:MM:SS.mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"
