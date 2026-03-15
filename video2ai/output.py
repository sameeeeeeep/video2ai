"""Assemble final JSON and Markdown outputs."""

import json
import os

from .contact_sheet import ContactSheet
from .frames import ExtractedFrame
from .llm import LLMAnalysis
from .probe import VideoInfo
from .transcribe import Segment
from .vision import FrameAnalysis


def write_json(
    video_info: VideoInfo,
    frames: list[ExtractedFrame],
    sheets: list[ContactSheet],
    segments: list[Segment],
    analyses: list[FrameAnalysis],
    llm_analysis: LLMAnalysis | None,
    output_dir: str,
) -> str:
    """Write structured metadata.json."""
    analysis_map = {a.frame_index: a for a in analyses}
    key_set = set(llm_analysis.key_frame_indices) if llm_analysis else set()

    frame_data = []
    for f in frames:
        entry = {
            "index": f.index,
            "timestamp": round(f.timestamp, 2),
            "path": os.path.relpath(f.path, output_dir),
            "is_key_frame": f.index in key_set,
        }
        a = analysis_map.get(f.index)
        if a:
            if a.ocr_text:
                entry["ocr_text"] = a.ocr_text
            if a.labels:
                entry["labels"] = a.labels
        if f.index in (llm_analysis.frame_reasoning if llm_analysis else {}):
            entry["llm_reasoning"] = llm_analysis.frame_reasoning[f.index]
        frame_data.append(entry)

    data = {
        "source": os.path.basename(video_info.path),
        "duration_seconds": round(video_info.duration, 2),
        "duration_formatted": video_info.duration_fmt,
        "resolution": video_info.resolution,
        "fps": round(video_info.fps, 2),
        "codec": video_info.codec,
        "file_size_mb": video_info.file_size_mb,
        "total_frames_extracted": len(frames),
        "key_frames_count": len(key_set),
        "contact_sheets_count": len(sheets),
        "transcript": [
            {"start": s.start, "end": s.end, "text": s.text} for s in segments
        ],
        "frames": frame_data,
        "key_frame_indices": sorted(key_set),
        "contact_sheets": [
            {
                "index": s.index,
                "time_range": s.time_range,
                "path": os.path.relpath(s.path, output_dir),
                "frame_indices": s.frame_indices,
            }
            for s in sheets
        ],
    }

    if llm_analysis and llm_analysis.summary:
        data["llm_analysis"] = {
            "model": llm_analysis.model,
            "summary": llm_analysis.summary,
            "key_frame_count": len(llm_analysis.key_frame_indices),
            "sections": [
                {
                    "title": s.title,
                    "start_time": s.start_time,
                    "end_time": s.end_time,
                    "topic": s.topic,
                    "key_points": s.key_points,
                }
                for s in llm_analysis.sections
            ],
        }

    path = os.path.join(output_dir, "metadata.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def write_markdown(
    video_info: VideoInfo,
    frames: list[ExtractedFrame],
    sheets: list[ContactSheet],
    segments: list[Segment],
    analyses: list[FrameAnalysis],
    llm_analysis: LLMAnalysis | None,
    output_dir: str,
) -> str:
    """Write human/AI-readable summary.md with interleaved timeline."""
    analysis_map = {a.frame_index: a for a in analyses}
    key_set = set(llm_analysis.key_frame_indices) if llm_analysis else set()
    key_frames = [f for f in frames if f.index in key_set] if key_set else frames
    lines = []

    # Header
    lines.append(f"# Video Summary: {os.path.basename(video_info.path)}")
    lines.append("")
    lines.append(
        f"**Duration:** {video_info.duration_fmt} | "
        f"**Resolution:** {video_info.resolution} | "
        f"**Total frames:** {len(frames)} | "
        f"**Key frames:** {len(key_frames)} | "
        f"**Contact sheets:** {len(sheets)}"
    )
    lines.append("")

    # LLM Summary
    if llm_analysis and llm_analysis.summary:
        lines.append("## AI Summary")
        lines.append("")
        lines.append(llm_analysis.summary)
        lines.append("")
        if llm_analysis.model:
            lines.append(f"*Analysis by {llm_analysis.model}*")
            lines.append("")

    # Sections breakdown
    if llm_analysis and llm_analysis.sections:
        lines.append("## Video Structure")
        lines.append("")
        for s in llm_analysis.sections:
            lines.append(f"### {s.title} ({_fmt_ts(s.start_time)} - {_fmt_ts(s.end_time)})")
            lines.append(f"{s.topic}")
            if s.key_points:
                for pt in s.key_points:
                    lines.append(f"- {pt}")
            # Show key frames for this section
            section_keys = [
                f for f in key_frames
                if s.start_time <= f.timestamp <= s.end_time
            ]
            if section_keys:
                lines.append("")
                for f in section_keys:
                    _append_frame_block(lines, f, analysis_map, llm_analysis, output_dir)
            lines.append("")

    # Contact sheets overview
    if sheets:
        lines.append("## Contact Sheets (Key Frames)")
        lines.append("")
        for s in sheets:
            rel = os.path.relpath(s.path, output_dir)
            lines.append(f"### Sheet {s.index} ({s.time_range})")
            lines.append(f"![Contact Sheet {s.index}]({rel})")
            lines.append("")

    # Interleaved timeline (key frames + transcript)
    if segments:
        lines.append("## Timeline")
        lines.append("")

        frame_iter = iter(key_frames)
        next_frame = next(frame_iter, None)

        for seg in segments:
            while next_frame and next_frame.timestamp <= seg.end:
                _append_frame_block(lines, next_frame, analysis_map, llm_analysis, output_dir)
                next_frame = next(frame_iter, None)

            lines.append(
                f"**[{_fmt_ts(seg.start)} - {_fmt_ts(seg.end)}]** {seg.text}"
            )
            lines.append("")

        while next_frame:
            _append_frame_block(lines, next_frame, analysis_map, llm_analysis, output_dir)
            next_frame = next(frame_iter, None)

    # Full transcript section
    if segments:
        lines.append("## Full Transcript")
        lines.append("")
        for seg in segments:
            lines.append(f"[{_fmt_ts(seg.start)}] {seg.text}")
        lines.append("")

    # OCR text summary
    ocr_frames = [a for a in analyses if a.ocr_text]
    if ocr_frames:
        lines.append("## On-Screen Text (OCR)")
        lines.append("")
        for a in ocr_frames:
            lines.append(f"**[{_fmt_ts(a.timestamp)}]**")
            lines.append(f"```\n{a.ocr_text}\n```")
            lines.append("")

    path = os.path.join(output_dir, "summary.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def _append_frame_block(
    lines: list[str],
    frame: ExtractedFrame,
    analysis_map: dict[int, FrameAnalysis],
    llm_analysis: LLMAnalysis | None,
    output_dir: str,
) -> None:
    """Append a frame entry with optional OCR/labels/reasoning to the markdown."""
    rel = os.path.relpath(frame.path, output_dir)
    lines.append(
        f"**[{_fmt_ts(frame.timestamp)}]** "
        f"![Frame {frame.index}]({rel})"
    )

    # LLM reasoning
    if llm_analysis and frame.index in llm_analysis.frame_reasoning:
        lines.append(f"  > {llm_analysis.frame_reasoning[frame.index]}")

    a = analysis_map.get(frame.index)
    if a:
        if a.labels:
            lines.append(f"  Labels: {', '.join(a.labels)}")
        if a.ocr_text:
            preview = a.ocr_text[:120].replace("\n", " ")
            if len(a.ocr_text) > 120:
                preview += "..."
            lines.append(f"  Text on screen: {preview}")

    lines.append("")


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"
