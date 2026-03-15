"""CLI entrypoint for video2ai."""

import argparse
import os
import sys
import time

from . import __version__
from .contact_sheet import build_contact_sheets
from .embed import embed_metadata
from .frames import extract_frames
from .llm import analyze_video, check_ollama
from .output import write_json, write_markdown
from .probe import check_ffmpeg, probe
from .transcribe import transcribe
from .vision import analyze_frames, is_available as vision_available


def main():
    parser = argparse.ArgumentParser(
        prog="video2ai",
        description="Convert video to AI-ingestable format (frames + transcript + LLM analysis)",
    )
    parser.add_argument("input", nargs="?", help="Path to input video file")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output directory (default: <input_name>_v2ai/)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Extract one frame every N seconds (default: 1.0)",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1280,
        help="Max frame width in pixels (default: 1280)",
    )
    parser.add_argument(
        "--grid",
        default="3x3",
        help="Contact sheet grid size as COLSxROWS (default: 3x3)",
    )
    parser.add_argument(
        "--no-transcribe",
        action="store_true",
        help="Skip audio transcription",
    )
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="Skip Apple Vision OCR/classification (macOS only)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM key frame analysis",
    )
    parser.add_argument(
        "--model",
        default="llama3.2",
        help="Ollama model for key frame analysis (default: llama3.2)",
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=85,
        help="JPEG quality 1-100 (default: 85)",
    )
    parser.add_argument(
        "--format",
        choices=["both", "json", "markdown"],
        default="both",
        help="Output format (default: both)",
    )
    parser.add_argument(
        "--no-embed",
        action="store_true",
        help="Skip embedding metadata into the video file",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Launch web UI instead of processing a file",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8910,
        help="Port for web UI (default: 8910)",
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"video2ai {__version__}",
    )

    args = parser.parse_args()

    # Web UI mode
    if args.web:
        from .web import run_web
        run_web(port=args.port)
        return

    # Validate input
    if not args.input or not os.path.isfile(args.input):
        print(f"Error: file not found: {args.input}")
        sys.exit(1)

    # Check ffmpeg
    try:
        check_ffmpeg()
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Parse grid
    try:
        cols, rows = map(int, args.grid.lower().split("x"))
    except ValueError:
        print(f"Error: invalid grid format '{args.grid}', use COLSxROWS e.g. 3x3")
        sys.exit(1)

    # Output dir
    if args.output:
        output_dir = args.output
    else:
        base = os.path.splitext(os.path.basename(args.input))[0]
        output_dir = f"{base}_v2ai"
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()

    # 1. Probe video
    print(f"Probing {args.input}...")
    info = probe(args.input)
    print(
        f"  {info.resolution}, {info.duration_fmt}, "
        f"{info.fps:.1f}fps, {info.codec}, "
        f"{'has' if info.has_audio else 'no'} audio, "
        f"{info.file_size_mb}MB"
    )

    # 2. Extract ALL frames at interval (default 1fps)
    print(f"Extracting frames (1 every {args.interval}s)...")
    frames = extract_frames(
        video_path=args.input,
        output_dir=output_dir,
        duration=info.duration,
        interval=args.interval,
        max_width=args.max_width,
        quality=args.quality,
    )
    print(f"  Extracted {len(frames)} frames")

    # 3. Transcribe audio
    segments = []
    if not args.no_transcribe and info.has_audio:
        print("Transcribing audio...")
        segments = transcribe(args.input, model_name=args.whisper_model)
        print(f"  Transcribed {len(segments)} segments")
    elif args.no_transcribe:
        print("Skipping transcription (--no-transcribe)")
    else:
        print("No audio track found, skipping transcription")

    # 4. Apple Vision analysis on ALL frames (macOS)
    analyses = []
    if not args.no_vision and vision_available():
        print("Running Apple Vision analysis (OCR + classification)...")
        analyses = analyze_frames(frames)
    elif not args.no_vision:
        print("Apple Vision not available, skipping OCR/classification")

    # 5. LLM key frame analysis
    llm_result = None
    if not args.no_llm:
        if check_ollama():
            print(f"Running LLM analysis ({args.model})...")
            llm_result = analyze_video(
                frames=frames,
                analyses=analyses,
                segments=segments,
                model=args.model,
            )
            print(
                f"  Selected {len(llm_result.key_frame_indices)} key frames"
                f" from {len(frames)} total"
            )
            if llm_result.summary:
                print(f"  Summary: {llm_result.summary[:120]}...")
        else:
            print("Ollama not running, skipping LLM analysis")
    else:
        print("Skipping LLM analysis (--no-llm)")

    # 6. Build contact sheets from KEY frames only
    key_set = set(llm_result.key_frame_indices) if llm_result else set()
    key_frames = [f for f in frames if f.index in key_set] if key_set else frames
    print(f"Building contact sheets from {len(key_frames)} frames...")
    sheets = build_contact_sheets(
        frames=key_frames,
        output_dir=output_dir,
        cols=cols,
        rows=rows,
        quality=args.quality,
    )
    print(f"  Created {len(sheets)} contact sheets")

    # 7. Write outputs
    if args.format in ("both", "json"):
        json_path = write_json(info, frames, sheets, segments, analyses, llm_result, output_dir)
        print(f"  JSON: {json_path}")

    if args.format in ("both", "markdown"):
        md_path = write_markdown(info, frames, sheets, segments, analyses, llm_result, output_dir)
        print(f"  Markdown: {md_path}")

    # 8. Embed metadata into video (smart video)
    if not args.no_embed and llm_result:
        print("Embedding metadata into video...")
        smart_path = embed_metadata(
            video_path=args.input,
            video_info=info,
            frames=frames,
            segments=segments,
            analyses=analyses,
            llm_analysis=llm_result,
            output_dir=output_dir,
        )
        if smart_path:
            print(f"  Smart video ready — upload to any AI for instant analysis")
    elif args.no_embed:
        print("Skipping metadata embedding (--no-embed)")

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s. Output: {output_dir}/")
    if llm_result and llm_result.key_frame_indices:
        print(f"  {len(llm_result.key_frame_indices)} key frames selected by {args.model}")


if __name__ == "__main__":
    main()
