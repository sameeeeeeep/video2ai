"""video2ai web interface — simple tool for extracting frames + transcript.

Upload a video → extract frames + transcribe audio → review frames alongside
transcript → select key frames → download PDF with transcript + key frames.

No LLM, no vision framework, no scoring algorithms. Just a clean manual workflow.
"""

import json
import os
import queue
import threading
import uuid

from flask import Flask, Response, jsonify, render_template_string, request, send_file

from .clip_match import is_available as clip_available, suggest_key_frames, cluster_frames, filter_by_clusters
from .frames import ExtractedFrame, extract_frames
from .probe import VideoInfo, check_ffmpeg, probe
from .transcribe import Segment, transcribe

app = Flask(__name__)

JOBS_DIR = os.path.expanduser("~/.video2ai_jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

_progress_queues: dict[str, queue.Queue] = {}


def _job_dir(job_id: str) -> str:
    return os.path.join(JOBS_DIR, job_id)


def _load_job_state(job_id: str) -> dict:
    path = os.path.join(_job_dir(job_id), "state.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_job_state(job_id: str, state: dict):
    path = os.path.join(_job_dir(job_id), "state.json")
    with open(path, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def _emit(job_id: str, **data):
    q = _progress_queues.get(job_id)
    if q:
        q.put(data)


# ─── Routes ───────────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return render_template_string(UPLOAD_PAGE)


@app.route("/process", methods=["POST"])
def process_video():
    file = request.files.get("video")
    if not file or not file.filename:
        return jsonify(error="No video file provided"), 400

    job_id = uuid.uuid4().hex[:12]
    job_path = _job_dir(job_id)
    os.makedirs(job_path, exist_ok=True)

    video_path = os.path.join(job_path, file.filename)
    file.save(video_path)

    config = {
        "interval": float(request.form.get("interval", "1.0")),
        "max_width": int(request.form.get("max_width", "1280")),
        "whisper_model": request.form.get("whisper_model", "base"),
        "quality": int(request.form.get("quality", "85")),
    }

    _progress_queues[job_id] = queue.Queue()

    thread = threading.Thread(
        target=_run_pipeline, args=(job_id, video_path, config), daemon=True
    )
    thread.start()

    return jsonify(job_id=job_id)


@app.route("/process-url", methods=["POST"])
def process_url():
    data = request.get_json()
    url = (data or {}).get("url", "").strip()
    if not url:
        return jsonify(error="No URL provided"), 400

    job_id = uuid.uuid4().hex[:12]
    job_path = _job_dir(job_id)
    os.makedirs(job_path, exist_ok=True)

    config = {
        "interval": float(data.get("interval", "1.0")),
        "max_width": int(data.get("max_width", "1280")),
        "whisper_model": data.get("whisper_model", "base"),
        "quality": int(data.get("quality", "85")),
    }

    _progress_queues[job_id] = queue.Queue()

    thread = threading.Thread(
        target=_run_url_pipeline, args=(job_id, url, job_path, config), daemon=True
    )
    thread.start()

    return jsonify(job_id=job_id)


@app.route("/processing/<job_id>")
def processing(job_id):
    return render_template_string(PROCESSING_PAGE, job_id=job_id)


@app.route("/progress/<job_id>")
def progress(job_id):
    def stream():
        q = _progress_queues.get(job_id)
        if not q:
            yield f"data: {json.dumps({'type': 'error', 'detail': 'Job not found'})}\n\n"
            return
        while True:
            try:
                msg = q.get(timeout=30)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg.get("type") in ("done", "error"):
                    break
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"

    return Response(stream(), mimetype="text/event-stream")


@app.route("/result/<job_id>")
def result(job_id):
    state = _load_job_state(job_id)
    if not state:
        return "Job not found", 404
    return render_template_string(RESULT_PAGE, job_id=job_id, state=state)


@app.route("/result/<job_id>/toggle-key-frame", methods=["POST"])
def toggle_key_frame(job_id):
    """Toggle a frame's key frame status."""
    state = _load_job_state(job_id)
    if not state:
        return jsonify(error="Job not found"), 404

    frame_index = request.json.get("index")
    if frame_index is None:
        return jsonify(error="No index"), 400

    key_set = set(state.get("key_frame_indices", []))
    if frame_index in key_set:
        key_set.discard(frame_index)
        is_key = False
    else:
        key_set.add(frame_index)
        is_key = True

    state["key_frame_indices"] = sorted(key_set)
    for f in state["frames"]:
        f["is_key_frame"] = f["index"] in key_set

    _save_job_state(job_id, state)
    return jsonify(ok=True, is_key=is_key, total_key=len(key_set))


@app.route("/result/<job_id>/apply-filter", methods=["POST"])
def apply_cluster_filter(job_id):
    """Toggle a cluster on/off to suppress/restore its frames from key selection."""
    state = _load_job_state(job_id)
    if not state:
        return jsonify(error="Job not found"), 404

    data = request.get_json()
    cluster_id = data.get("cluster_id")
    action = data.get("action", "suppress")  # "suppress" or "restore"

    if cluster_id is None:
        return jsonify(error="No cluster_id"), 400

    suppressed = set(state.get("suppressed_clusters", []))
    cluster_frame_indices = set()
    for c in state.get("clusters", []):
        if c["id"] == cluster_id:
            cluster_frame_indices = set(c["frame_indices"])
            break

    if action == "suppress":
        suppressed.add(cluster_id)
        # Remove cluster's frames from key frames
        key_set = set(state.get("key_frame_indices", []))
        key_set -= cluster_frame_indices
        state["key_frame_indices"] = sorted(key_set)
    else:
        suppressed.discard(cluster_id)
        # Don't auto-restore — user can manually add back

    state["suppressed_clusters"] = sorted(suppressed)
    for f in state["frames"]:
        f["is_key_frame"] = f["index"] in set(state["key_frame_indices"])

    _save_job_state(job_id, state)
    return jsonify(
        ok=True,
        suppressed=sorted(suppressed),
        total_key=len(state["key_frame_indices"]),
        removed=len(cluster_frame_indices) if action == "suppress" else 0,
    )


@app.route("/result/<job_id>/update-transcript", methods=["POST"])
def update_transcript(job_id):
    state = _load_job_state(job_id)
    if not state:
        return jsonify(error="Job not found"), 404

    state["transcript"] = request.json.get("segments", [])
    _save_job_state(job_id, state)
    return jsonify(ok=True)


@app.route("/result/<job_id>/download")
def download_export(job_id):
    """Generate and download a self-contained HTML with key frames + transcript."""
    state = _load_job_state(job_id)
    if not state:
        return "Job not found", 404

    job_path = _job_dir(job_id)
    key_set = set(state.get("key_frame_indices", []))
    key_frames = [f for f in state["frames"] if f["index"] in key_set]

    if not key_frames:
        return "No key frames selected. Go back and select some frames.", 400

    html_path = _build_export_html(
        job_path=job_path,
        source_name=state["source"],
        duration=state.get("duration_formatted", ""),
        resolution=state.get("resolution", ""),
        key_frames=key_frames,
        transcript=state.get("transcript", []),
    )

    return send_file(
        html_path, as_attachment=True,
        download_name=f"{os.path.splitext(state['source'])[0]}_export.html",
    )


@app.route("/files/<job_id>/<path:filepath>")
def serve_file(job_id, filepath):
    fpath = os.path.join(_job_dir(job_id), filepath)
    if not os.path.isfile(fpath):
        return "Not found", 404
    return send_file(fpath)


# ─── Pipeline ─────────────────────────────────────────────────────────────────


def _run_url_pipeline(job_id: str, url: str, job_path: str, config: dict):
    """Download video from URL via yt-dlp, then run the normal pipeline."""
    import subprocess

    _emit(job_id, type="step", step="download", status="active")
    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "--no-playlist",
                "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
                "--merge-output-format", "mp4",
                "-o", os.path.join(job_path, "%(title).80s.%(ext)s"),
                "--print", "after_move:filepath",
                url,
            ],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            _emit(job_id, type="error", detail=f"Download failed: {result.stderr[:300]}")
            return

        video_path = result.stdout.strip().split("\n")[-1]
        if not os.path.isfile(video_path):
            _emit(job_id, type="error", detail="Download completed but file not found")
            return

        _emit(job_id, type="step", step="download", status="done",
              filename=os.path.basename(video_path))
    except subprocess.TimeoutExpired:
        _emit(job_id, type="error", detail="Download timed out (5 min limit)")
        return
    except FileNotFoundError:
        _emit(job_id, type="error", detail="yt-dlp not found. Install with: pip install yt-dlp")
        return

    _run_pipeline(job_id, video_path, config)


def _run_pipeline(job_id: str, video_path: str, config: dict):
    """Simple pipeline: probe → extract frames → transcribe. That's it."""
    job_path = _job_dir(job_id)
    try:
        check_ffmpeg()
    except RuntimeError as e:
        _emit(job_id, type="error", detail=str(e))
        return

    try:
        # 1. Probe
        _emit(job_id, type="step", step="probe", status="active")
        info = probe(video_path)
        _emit(job_id, type="probe_done", source=os.path.basename(video_path),
              duration=info.duration_fmt, resolution=info.resolution,
              fps=round(info.fps, 1), codec=info.codec,
              has_audio=info.has_audio, size_mb=info.file_size_mb)
        _emit(job_id, type="step", step="probe", status="done")

        # 2. Extract frames
        _emit(job_id, type="step", step="frames", status="active")

        def on_frame(frame, progress):
            _emit(job_id, type="frame", index=frame.index,
                  timestamp=round(frame.timestamp, 2),
                  path=os.path.relpath(frame.path, job_path),
                  progress=round(progress, 3))

        frames = extract_frames(
            video_path=video_path, output_dir=job_path, duration=info.duration,
            interval=config["interval"], max_width=config["max_width"],
            quality=config["quality"], on_frame=on_frame,
        )
        _emit(job_id, type="step", step="frames", status="done", count=len(frames))

        # 3. Transcribe audio
        segments = []
        if info.has_audio:
            _emit(job_id, type="step", step="transcribe", status="active")
            segments = transcribe(video_path, model_name=config["whisper_model"])
            for seg in segments:
                _emit(job_id, type="segment", start=seg.start, end=seg.end, text=seg.text)
            _emit(job_id, type="step", step="transcribe", status="done", count=len(segments))
        else:
            _emit(job_id, type="step", step="transcribe", status="skipped")

        # 4. Vision-based frame suggestions (works with or without transcript)
        suggested_indices = set()
        if clip_available():
            _emit(job_id, type="step", step="suggest", status="active")

            def on_clip_progress(stage, pct):
                _emit(job_id, type="clip_progress", stage=stage, progress=round(pct, 3))

            suggestions = suggest_key_frames(
                frames=frames, segments=segments or None,
                top_k=3, on_progress=on_clip_progress,
            )
            suggested_indices = {s.frame_index for s in suggestions}
            _emit(job_id, type="step", step="suggest", status="done",
                  count=len(suggested_indices))
        else:
            _emit(job_id, type="step", step="suggest", status="skipped")

        # 5. Cluster frames into visual themes for filtering
        clusters_data = {}
        if clip_available():
            clusters_data = cluster_frames(frames)

        frame_cluster_map = clusters_data.get("frame_cluster_map", {})

        # 6. Save state — suggestions pre-selected, clusters for filtering
        state = {
            "source": os.path.basename(video_path),
            "duration_seconds": round(info.duration, 2),
            "duration_formatted": info.duration_fmt,
            "resolution": info.resolution,
            "fps": round(info.fps, 2),
            "codec": info.codec,
            "file_size_mb": info.file_size_mb,
            "total_frames": len(frames),
            "key_frame_indices": sorted(suggested_indices),
            "frames": [
                {
                    "index": f.index,
                    "timestamp": round(f.timestamp, 2),
                    "path": os.path.relpath(f.path, job_path),
                    "is_key_frame": f.index in suggested_indices,
                    "cluster": frame_cluster_map.get(f.index, -1),
                }
                for f in frames
            ],
            "transcript": [
                {"start": s.start, "end": s.end, "text": s.text}
                for s in segments
            ],
            "clusters": [
                {
                    "id": c["id"],
                    "representative_index": c["representative_index"],
                    "size": c["size"],
                    "frame_indices": c["frame_indices"],
                }
                for c in clusters_data.get("clusters", [])
            ],
        }
        _save_job_state(job_id, state)
        _emit(job_id, type="done")

    except Exception as e:
        import traceback
        traceback.print_exc()
        _emit(job_id, type="error", detail=str(e))


# ─── Export: Self-contained HTML with embedded images ─────────────────────────


def _build_export_html(
    job_path: str,
    source_name: str,
    duration: str,
    resolution: str,
    key_frames: list[dict],
    transcript: list[dict],
) -> str:
    """Build a single self-contained HTML file with base64-embedded key frame
    images alongside their transcript segments.

    This is the ideal format for AI consumption: one file, structured HTML,
    images inline next to the text that was spoken at that moment. Any AI
    can parse it. Any browser can render it for human review.
    """
    import base64
    from html import escape

    def _ts(seconds):
        m, s = divmod(int(seconds), 60)
        return f"{m}:{s:02d}"

    def _img_to_base64(path):
        try:
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode("ascii")
            return f"data:image/jpeg;base64,{data}"
        except Exception:
            return ""

    # Group key frames by which transcript segment they belong to
    # Each segment gets the key frames that fall within its time range
    seg_frames = {}  # seg_index -> [frame, ...]
    orphan_frames = []  # frames with no matching segment

    for frame in key_frames:
        ts = frame["timestamp"]
        matched = False
        for i, seg in enumerate(transcript):
            if seg["start"] <= ts + 0.5 and seg["end"] >= ts - 0.5:
                seg_frames.setdefault(i, []).append(frame)
                matched = True
                break
        if not matched:
            orphan_frames.append(frame)

    # Build HTML
    parts = [f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{escape(source_name)} - Key Frames &amp; Transcript</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 900px; margin: 0 auto; padding: 24px; color: #1a1a1a; background: #fafafa; }}
  h1 {{ font-size: 22px; border-bottom: 3px solid #222; padding-bottom: 8px; }}
  .meta {{ color: #666; font-size: 13px; margin-bottom: 24px; }}
  .segment {{ margin-bottom: 32px; border: 2px solid #ddd; background: #fff; padding: 16px; }}
  .segment.has-frames {{ border-color: #ff6b35; }}
  .seg-time {{ font-size: 12px; color: #4361ee; font-weight: 700; font-family: monospace; margin-bottom: 6px; }}
  .seg-text {{ font-size: 15px; line-height: 1.6; margin-bottom: 12px; }}
  .frame-row {{ display: flex; gap: 8px; flex-wrap: wrap; }}
  .frame-item {{ flex: 1; min-width: 250px; max-width: 100%; }}
  .frame-item img {{ width: 100%; border: 2px solid #222; display: block; }}
  .frame-ts {{ font-size: 11px; font-weight: 700; font-family: monospace; color: #ff6b35; margin-top: 4px; }}
  .transcript-section {{ margin-top: 40px; border-top: 3px solid #222; padding-top: 20px; }}
  .transcript-section h2 {{ font-size: 18px; margin-bottom: 16px; }}
  .transcript-line {{ display: flex; gap: 12px; padding: 4px 0; border-bottom: 1px solid #eee; }}
  .transcript-line .time {{ font-size: 12px; color: #4361ee; font-weight: 700; font-family: monospace; min-width: 50px; padding-top: 2px; }}
  .transcript-line .text {{ font-size: 14px; line-height: 1.5; }}
</style>
</head>
<body>
<h1>{escape(source_name)}</h1>
<div class="meta">{escape(duration)} &bull; {escape(resolution)} &bull; {len(key_frames)} key frames &bull; {len(transcript)} transcript segments</div>
"""]

    # Segments with key frames
    for i, seg in enumerate(transcript):
        frames = seg_frames.get(i, [])
        has_frames = "has-frames" if frames else ""
        parts.append(f'<div class="segment {has_frames}">')
        parts.append(f'<div class="seg-time">[{_ts(seg["start"])} - {_ts(seg["end"])}]</div>')
        parts.append(f'<div class="seg-text">{escape(seg["text"].strip())}</div>')

        if frames:
            parts.append('<div class="frame-row">')
            for frame in frames:
                img_path = os.path.join(job_path, frame["path"])
                b64 = _img_to_base64(img_path)
                if b64:
                    parts.append(f'<div class="frame-item">')
                    parts.append(f'<img src="{b64}" alt="Frame at {_ts(frame["timestamp"])}">')
                    parts.append(f'<div class="frame-ts">{_ts(frame["timestamp"])}</div>')
                    parts.append('</div>')
            parts.append('</div>')

        parts.append('</div>')

    # Orphan frames (not matching any segment)
    if orphan_frames:
        parts.append('<div class="segment has-frames">')
        parts.append('<div class="seg-time">Other key frames</div>')
        parts.append('<div class="frame-row">')
        for frame in orphan_frames:
            img_path = os.path.join(job_path, frame["path"])
            b64 = _img_to_base64(img_path)
            if b64:
                parts.append(f'<div class="frame-item">')
                parts.append(f'<img src="{b64}" alt="Frame at {_ts(frame["timestamp"])}">')
                parts.append(f'<div class="frame-ts">{_ts(frame["timestamp"])}</div>')
                parts.append('</div>')
        parts.append('</div></div>')

    # Full transcript
    if transcript:
        parts.append('<div class="transcript-section">')
        parts.append('<h2>Full Transcript</h2>')
        for seg in transcript:
            parts.append(f'<div class="transcript-line">')
            parts.append(f'<span class="time">{_ts(seg["start"])}</span>')
            parts.append(f'<span class="text">{escape(seg["text"].strip())}</span>')
            parts.append('</div>')
        parts.append('</div>')

    parts.append('</body></html>')

    out_path = os.path.join(job_path, "export.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))

    return out_path


# ─── Templates ────────────────────────────────────────────────────────────────

CSS_VARS = """
:root {
  --bg: #e8e4da; --surface: #fff; --surface2: #f0ece2;
  --border: #222; --text: #1a1a1a; --text2: #555;
  --accent: #4361ee; --accent2: #ff6b35; --accent-bg: #c3f73a;
  --danger: #ff3333; --success: #2ec4b6; --warn: #ff9f1c;
  --key: #ff6b35; --key-bg: #fff0e6;
  --shadow: 4px 4px 0 #222;
  --bw: 3px;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: var(--bg); color: var(--text); font-family: 'Space Grotesk', 'DM Sans', -apple-system, system-ui, sans-serif; }
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Space+Mono:wght@400;700&display=swap');
"""

UPLOAD_PAGE = (
    r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>video2ai</title>
<style>"""
    + CSS_VARS
    + r"""
  body { min-height: 100vh; display: flex; align-items: center; justify-content: center; }
  .container { max-width: 640px; width: 100%; padding: 24px; }
  .logo {
    display: inline-block; font-size: 14px; font-weight: 700; letter-spacing: 2px;
    text-transform: uppercase; color: var(--text); background: var(--accent-bg);
    padding: 6px 14px; border: var(--bw) solid var(--border); box-shadow: var(--shadow);
    margin-bottom: 36px; font-family: 'Space Mono', monospace;
  }
  h1 { font-size: 42px; font-weight: 700; margin-bottom: 8px; letter-spacing: -1px; line-height: 1.1; }
  h1 span { background: var(--accent); color: #fff; padding: 0 8px; border: var(--bw) solid var(--border); display: inline-block; transform: rotate(-1deg); }
  .subtitle { color: var(--text2); margin-bottom: 36px; font-size: 16px; line-height: 1.6; }

  .dropzone {
    border: var(--bw) dashed var(--border); padding: 48px 32px; text-align: center;
    cursor: pointer; transition: all .15s; background: var(--surface);
    margin-bottom: 28px; box-shadow: var(--shadow);
  }
  .dropzone:hover, .dropzone.dragover {
    border-style: solid; background: var(--accent-bg);
    transform: translate(-2px, -2px); box-shadow: 6px 6px 0 var(--border);
  }
  .dropzone .icon { font-size: 48px; margin-bottom: 12px; display: block; }
  .dropzone p { color: var(--text2); font-size: 15px; font-weight: 500; }
  .dropzone .filename { color: var(--accent2); font-weight: 700; margin-top: 10px; font-size: 15px; }
  .dropzone input { display: none; }

  .settings { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 28px; }
  .field { display: flex; flex-direction: column; gap: 5px; }
  .field label {
    font-size: 11px; color: var(--text); text-transform: uppercase; letter-spacing: 1px;
    font-weight: 700; font-family: 'Space Mono', monospace;
  }
  .field input, .field select {
    background: var(--surface); border: var(--bw) solid var(--border);
    color: var(--text); padding: 10px 12px; font-size: 14px; outline: none;
    font-family: inherit; box-shadow: 2px 2px 0 var(--border); transition: all .1s;
  }
  .field input:focus, .field select:focus {
    box-shadow: 4px 4px 0 var(--accent); border-color: var(--accent);
    transform: translate(-1px, -1px);
  }

  .btn {
    display: block; width: 100%; padding: 16px; border: var(--bw) solid var(--border);
    background: var(--accent); color: #fff; font-size: 16px; font-weight: 700;
    cursor: pointer; transition: all .1s; letter-spacing: .5px;
    text-transform: uppercase; box-shadow: var(--shadow); font-family: 'Space Mono', monospace;
  }
  .btn:hover { transform: translate(-2px, -2px); box-shadow: 6px 6px 0 var(--border); }
  .btn:active { transform: translate(2px, 2px); box-shadow: 1px 1px 0 var(--border); }
  .btn:disabled { opacity: .5; cursor: not-allowed; transform: none; box-shadow: var(--shadow); }

  .tab-bar { display: flex; gap: 0; margin-bottom: 20px; }
  .tab {
    flex: 1; padding: 12px; border: var(--bw) solid var(--border); background: var(--surface);
    font-size: 13px; font-weight: 700; cursor: pointer; text-transform: uppercase;
    letter-spacing: 1px; font-family: 'Space Mono', monospace; color: var(--text2);
    transition: all .1s;
  }
  .tab:first-child { border-right: none; }
  .tab.active { background: var(--accent); color: #fff; }
  .tab:hover:not(.active) { background: var(--accent-bg); }

  .url-input { margin-bottom: 28px; }
  .url-input input {
    width: 100%; box-sizing: border-box; padding: 16px; border: var(--bw) solid var(--border);
    background: var(--surface); color: var(--text); font-size: 15px; font-family: inherit;
    box-shadow: var(--shadow); outline: none; transition: all .1s;
  }
  .url-input input:focus { box-shadow: 4px 4px 0 var(--accent); border-color: var(--accent); transform: translate(-1px,-1px); }
  .url-input input::placeholder { color: var(--text2); }
  .url-hint { color: var(--text2); font-size: 12px; margin-top: 8px; }
</style>
</head>
<body>
<div class="container">
  <div class="logo">video2ai</div>
  <h1>Extract frames & <span>transcript</span></h1>
  <p class="subtitle">Upload a video or paste a URL. Get frames + transcript. Pick key frames. Download.</p>

  <div class="tab-bar">
    <button class="tab active" data-tab="upload" onclick="switchTab('upload')">Upload File</button>
    <button class="tab" data-tab="url" onclick="switchTab('url')">Paste URL</button>
  </div>

  <form id="form" enctype="multipart/form-data">
    <div id="tab-upload">
      <div class="dropzone" id="dropzone">
        <span class="icon">&#127916;</span>
        <p>Drop a video here or click to browse</p>
        <div class="filename" id="filename"></div>
        <input type="file" name="video" id="videoInput" accept="video/*">
      </div>
    </div>

    <div id="tab-url" style="display:none">
      <div class="url-input">
        <input type="text" name="video_url" id="urlInput" placeholder="https://youtube.com/watch?v=... or any video URL" autocomplete="off">
        <p class="url-hint">YouTube, Vimeo, Twitter/X, or any yt-dlp supported URL</p>
      </div>
    </div>

    <div class="settings">
      <div class="field"><label>Frame interval (sec)</label><input type="number" name="interval" value="1.0" step="0.5" min="0.5"></div>
      <div class="field"><label>Max width (px)</label><input type="number" name="max_width" value="1280"></div>
      <div class="field"><label>Whisper model</label>
        <select name="whisper_model">
          <option value="tiny">tiny</option><option value="base" selected>base</option>
          <option value="small">small</option><option value="medium">medium</option>
          <option value="large">large</option>
        </select>
      </div>
      <div class="field"><label>JPEG quality</label><input type="number" name="quality" value="85" min="1" max="100"></div>
    </div>

    <button type="submit" class="btn" id="processBtn">Process Video</button>
  </form>
</div>

<script>
var dz=document.getElementById('dropzone'),inp=document.getElementById('videoInput'),fn=document.getElementById('filename');
var activeTab='upload';

function switchTab(tab){
  activeTab=tab;
  document.querySelectorAll('.tab').forEach(function(t){t.classList.toggle('active',t.dataset.tab===tab)});
  document.getElementById('tab-upload').style.display=tab==='upload'?'':'none';
  document.getElementById('tab-url').style.display=tab==='url'?'':'none';
}

dz.onclick=function(){inp.click()};
dz.ondragover=function(e){e.preventDefault();dz.classList.add('dragover')};
dz.ondragleave=function(){dz.classList.remove('dragover')};
dz.ondrop=function(e){e.preventDefault();dz.classList.remove('dragover');if(e.dataTransfer.files.length){inp.files=e.dataTransfer.files;fn.textContent=inp.files[0].name}};
inp.onchange=function(){if(inp.files.length)fn.textContent=inp.files[0].name};

document.getElementById('form').onsubmit=async function(e){
  e.preventDefault();
  var btn=document.getElementById('processBtn');

  if(activeTab==='url'){
    var url=document.getElementById('urlInput').value.trim();
    if(!url)return alert('Paste a video URL first');
    btn.disabled=true;btn.textContent='DOWNLOADING...';
    var fd=new FormData(e.target);
    fd.delete('video');
    var res=await fetch('/process-url',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url:url,interval:fd.get('interval'),max_width:fd.get('max_width'),whisper_model:fd.get('whisper_model'),quality:fd.get('quality')})});
    var data=await res.json();
    if(data.error){alert(data.error);btn.disabled=false;btn.textContent='Process Video';return}
    window.location.href='/processing/'+data.job_id;
  } else {
    if(!inp.files.length)return alert('Select a video first');
    btn.disabled=true;btn.textContent='UPLOADING...';
    var fd=new FormData(e.target);
    var res=await fetch('/process',{method:'POST',body:fd});
    var data=await res.json();
    if(data.error){alert(data.error);btn.disabled=false;btn.textContent='Process Video';return}
    window.location.href='/processing/'+data.job_id;
  }
};
</script>
</body></html>"""
)


PROCESSING_PAGE = (
    r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>video2ai - Processing</title>
<style>"""
    + CSS_VARS
    + r"""
  .top-bar { position: fixed; top: 0; left: 0; right: 0; height: 6px; background: var(--surface2); z-index: 100; border-bottom: 2px solid var(--border); }
  .top-bar-fill { height: 100%; width: 0%; transition: width .4s ease; background: var(--accent-bg); }

  .page { max-width: 960px; margin: 0 auto; padding: 48px 24px 100px; }
  .logo {
    display: inline-block; font-size: 13px; font-weight: 700; letter-spacing: 2px;
    text-transform: uppercase; color: var(--text); background: var(--accent-bg);
    padding: 5px 12px; border: var(--bw) solid var(--border); box-shadow: var(--shadow);
    margin-bottom: 28px; font-family: 'Space Mono', monospace;
  }

  .info-card {
    background: var(--surface); border: var(--bw) solid var(--border);
    padding: 18px 22px; margin-bottom: 28px; display: none; box-shadow: var(--shadow);
  }
  .info-card .name { font-size: 18px; font-weight: 700; margin-bottom: 8px; }
  .info-card .tags { display: flex; gap: 6px; flex-wrap: wrap; }
  .info-card .tag {
    font-size: 12px; color: var(--text); background: var(--surface2);
    padding: 3px 10px; border: 2px solid var(--border); font-weight: 700;
    font-variant-numeric: tabular-nums; font-family: 'Space Mono', monospace;
  }

  .pipeline { display: flex; gap: 4px; margin-bottom: 32px; align-items: center; flex-wrap: wrap; }
  .pipe-step {
    display: flex; align-items: center; gap: 6px; padding: 7px 12px;
    font-size: 13px; font-weight: 700; color: var(--text2);
    transition: all .2s; background: var(--surface); border: 2px solid var(--border);
    text-transform: uppercase; font-family: 'Space Mono', monospace; font-size: 11px;
  }
  .pipe-step .dot { width: 10px; height: 10px; background: var(--border); transition: all .2s; }
  .pipe-step.active { color: var(--text); background: var(--accent-bg); border-color: var(--border); }
  .pipe-step.active .dot { background: var(--accent); animation: blink 1s step-end infinite; }
  .pipe-step.done { color: var(--success); background: #d4f5ef; }
  .pipe-step.done .dot { background: var(--success); animation: none; }
  .pipe-step.skipped { opacity: .35; }
  .pipe-connector { width: 12px; height: 2px; background: var(--border); flex-shrink: 0; }

  @keyframes blink { 50% { opacity: .3; } }
  @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

  .section { margin-bottom: 28px; }
  .section-header {
    font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px;
    color: var(--text); margin-bottom: 12px; display: flex; align-items: center; gap: 8px;
    font-family: 'Space Mono', monospace;
  }
  .section-header .count {
    background: var(--accent); color: #fff; font-size: 11px; padding: 2px 8px;
    font-weight: 700; min-width: 24px; text-align: center; border: 2px solid var(--border);
  }

  .frame-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 6px;
    min-height: 60px;
  }
  .live-frame {
    overflow: hidden; position: relative; background: var(--surface2); aspect-ratio: 16/9;
    border: 2px solid var(--border); animation: fadeIn .3s ease both;
  }
  .live-frame img { width: 100%; height: 100%; object-fit: cover; display: block; }
  .live-frame .ts {
    position: absolute; bottom: 0; left: 0;
    background: var(--text); color: var(--bg); padding: 2px 6px;
    font-size: 11px; font-weight: 700; font-variant-numeric: tabular-nums;
    font-family: 'Space Mono', monospace;
  }

  .transcript-feed { max-height: 280px; overflow-y: auto; display: flex; flex-direction: column; gap: 3px; }
  .live-seg {
    display: flex; gap: 10px; padding: 6px 10px;
    background: var(--surface); border: 2px solid var(--border);
  }
  .live-seg .time {
    font-size: 12px; color: var(--accent); font-weight: 700; white-space: nowrap;
    font-variant-numeric: tabular-nums; min-width: 70px; font-family: 'Space Mono', monospace;
  }
  .live-seg .txt { font-size: 14px; color: var(--text); line-height: 1.4; }

  .stats { display: flex; gap: 10px; margin-bottom: 28px; }
  .stat {
    background: var(--surface); border: var(--bw) solid var(--border);
    padding: 14px 18px; flex: 1; text-align: center; box-shadow: var(--shadow);
  }
  .stat .num {
    font-size: 36px; font-weight: 700; font-variant-numeric: tabular-nums;
    transition: all .15s; font-family: 'Space Mono', monospace;
  }
  .stat .num.bump { transform: scale(1.1); color: var(--accent2); }
  .stat .label {
    font-size: 10px; color: var(--text2); text-transform: uppercase; letter-spacing: 1.5px;
    margin-top: 4px; font-weight: 700; font-family: 'Space Mono', monospace;
  }

  .done-bar {
    position: fixed; bottom: 0; left: 0; right: 0;
    background: var(--accent-bg); border-top: var(--bw) solid var(--border);
    padding: 14px 24px; display: none; z-index: 50;
  }
  .done-bar .inner { max-width: 960px; margin: 0 auto; display: flex; justify-content: space-between; align-items: center; }
  .done-bar .msg { font-weight: 700; font-size: 15px; }
  .done-bar .msg span { color: var(--success); }
  .done-bar a {
    background: var(--accent); color: #fff; padding: 10px 28px;
    text-decoration: none; font-weight: 700; font-size: 14px; border: var(--bw) solid var(--border);
    box-shadow: var(--shadow); transition: all .1s; text-transform: uppercase;
    font-family: 'Space Mono', monospace; letter-spacing: .5px;
  }
  .done-bar a:hover { transform: translate(-2px, -2px); box-shadow: 6px 6px 0 var(--border); }
</style>
</head>
<body>
<div class="top-bar"><div class="top-bar-fill" id="topBar"></div></div>

<div class="page">
  <div class="logo">video2ai</div>

  <div class="info-card" id="infoCard">
    <div class="name" id="videoName">Processing...</div>
    <div class="tags" id="videoTags"></div>
  </div>

  <div class="pipeline" id="pipeline">
    <div class="pipe-step" data-step="download"><div class="dot"></div>Download</div>
    <div class="pipe-step active" data-step="probe"><div class="dot"></div>Probe</div>
    <div class="pipe-connector"></div>
    <div class="pipe-step" data-step="frames"><div class="dot"></div>Frames</div>
    <div class="pipe-connector"></div>
    <div class="pipe-step" data-step="transcribe"><div class="dot"></div>Transcribe</div>
    <div class="pipe-connector"></div>
    <div class="pipe-step" data-step="suggest"><div class="dot"></div>Key Frames</div>
  </div>

  <div class="stats">
    <div class="stat"><div class="num" id="statFrames">0</div><div class="label">Frames</div></div>
    <div class="stat"><div class="num" id="statSegments">0</div><div class="label">Segments</div></div>
  </div>

  <div class="section" id="framesSection" style="display:none">
    <div class="section-header">Extracting Frames <span class="count" id="frameCountBadge">0</span></div>
    <div class="frame-grid" id="liveFrameGrid"></div>
  </div>

  <div class="section" id="transcriptSection" style="display:none">
    <div class="section-header">Transcript <span class="count" id="segCountBadge">0</span></div>
    <div class="transcript-feed" id="transcriptFeed"></div>
  </div>
</div>

<div class="done-bar" id="doneBar">
  <div class="inner">
    <div class="msg"><span>DONE</span> &mdash; Review suggested key frames</div>
    <a href="/result/{{ job_id }}">Review & Select</a>
  </div>
</div>

<script>
var JOB = "{{ job_id }}";
var frameCount = 0, segCount = 0;
var progressPct = 0;

var es = new EventSource('/progress/' + JOB);
es.onmessage = function(e) {
  var d = JSON.parse(e.data);
  if (d.type === 'heartbeat') return;
  handle(d);
};

function handle(d) {
  if (d.type === 'probe_done') {
    document.getElementById('infoCard').style.display = 'block';
    document.getElementById('videoName').textContent = d.source;
    var tags = document.getElementById('videoTags');
    var items = [d.resolution, d.duration, d.fps+'fps', d.codec, d.has_audio?'audio':'no audio', d.size_mb+'MB'];
    for (var i = 0; i < items.length; i++) {
      var s = document.createElement('span');
      s.className = 'tag';
      s.textContent = items[i];
      tags.appendChild(s);
    }
    setProgress(5);
  }
  else if (d.type === 'step') {
    setStep(d.step, d.status);
    if (d.step === 'frames' && d.status === 'active') {
      document.getElementById('framesSection').style.display = 'block';
      setProgress(8);
    }
    if (d.step === 'frames' && d.status === 'done') setProgress(40);
    if (d.step === 'transcribe' && d.status === 'active') {
      document.getElementById('transcriptSection').style.display = 'block';
      setProgress(45);
    }
    if (d.step === 'transcribe' && d.status === 'done') setProgress(60);
    if (d.step === 'transcribe' && d.status === 'skipped') setProgress(60);
    if (d.step === 'suggest' && d.status === 'active') setProgress(65);
    if (d.step === 'suggest' && d.status === 'done') setProgress(100);
    if (d.step === 'suggest' && d.status === 'skipped') setProgress(100);
  }
  else if (d.type === 'frame') {
    frameCount++;
    addFrame(d);
    bumpStat('statFrames', frameCount);
    document.getElementById('frameCountBadge').textContent = frameCount;
    setProgress(8 + Math.min(32, d.progress * 32));
  }
  else if (d.type === 'segment') {
    segCount++;
    addSegment(d);
    bumpStat('statSegments', segCount);
    document.getElementById('segCountBadge').textContent = segCount;
  }
  else if (d.type === 'done') {
    es.close();
    document.getElementById('doneBar').style.display = 'block';
  }
  else if (d.type === 'error') {
    es.close();
    alert('Error: ' + d.detail);
  }
}

function setStep(name, status) {
  var el = document.querySelector('.pipe-step[data-step="' + name + '"]');
  if (!el) return;
  el.classList.remove('active','done','skipped');
  if (status) el.classList.add(status);
}

function setProgress(pct) {
  pct = Math.max(progressPct, pct);
  progressPct = pct;
  document.getElementById('topBar').style.width = pct + '%';
}

function addFrame(d) {
  var grid = document.getElementById('liveFrameGrid');
  var el = document.createElement('div');
  el.className = 'live-frame';
  var img = document.createElement('img');
  img.src = '/files/' + JOB + '/' + d.path;
  img.loading = 'lazy';
  var ts = document.createElement('div');
  ts.className = 'ts';
  ts.textContent = fmtTs(d.timestamp);
  el.appendChild(img);
  el.appendChild(ts);
  grid.appendChild(el);
  el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function addSegment(d) {
  var feed = document.getElementById('transcriptFeed');
  var el = document.createElement('div');
  el.className = 'live-seg';
  var timeSpan = document.createElement('span');
  timeSpan.className = 'time';
  timeSpan.textContent = fmtTs(d.start);
  var txtSpan = document.createElement('span');
  txtSpan.className = 'txt';
  txtSpan.textContent = d.text;
  el.appendChild(timeSpan);
  el.appendChild(txtSpan);
  feed.appendChild(el);
  feed.scrollTop = feed.scrollHeight;
}

function bumpStat(id, val) {
  var el = document.getElementById(id);
  el.textContent = val;
  el.classList.add('bump');
  setTimeout(function() { el.classList.remove('bump'); }, 150);
}

function fmtTs(s) {
  var m = Math.floor(s/60), sec = Math.floor(s%60);
  return m + ':' + String(sec).padStart(2,'0');
}
</script>
</body></html>"""
)


RESULT_PAGE = (
    r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>video2ai - Select Key Frames</title>
<style>"""
    + CSS_VARS
    + r"""
  html, body { height: 100%; overflow: hidden; }

  .top-header {
    background: var(--surface); border-bottom: var(--bw) solid var(--border);
    padding: 12px 24px; display: flex; justify-content: space-between; align-items: center;
    flex-shrink: 0;
  }
  .top-header .left { display: flex; align-items: center; gap: 14px; }
  .top-header .logo {
    font-size: 12px; font-weight: 700; letter-spacing: 2px;
    text-transform: uppercase; background: var(--accent-bg);
    padding: 4px 10px; border: 2px solid var(--border);
    font-family: 'Space Mono', monospace;
  }
  .top-header h1 { font-size: 18px; font-weight: 700; }
  .top-header .meta { display: flex; gap: 6px; }
  .top-header .meta span {
    font-size: 11px; color: var(--text); background: var(--surface2);
    padding: 3px 8px; border: 2px solid var(--border); font-weight: 700;
    font-family: 'Space Mono', monospace;
  }

  .main-layout { display: flex; height: calc(100vh - 60px - 64px); }

  .sidebar {
    width: 360px; min-width: 360px; background: var(--surface);
    border-right: var(--bw) solid var(--border);
    display: flex; flex-direction: column; overflow: hidden;
  }
  .sidebar-header {
    padding: 14px 16px; border-bottom: 2px solid var(--border);
    font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px;
    font-family: 'Space Mono', monospace; flex-shrink: 0;
    display: flex; justify-content: space-between; align-items: center;
  }
  .sidebar-header .seg-count {
    background: var(--accent); color: #fff; padding: 2px 8px;
    border: 2px solid var(--border); font-size: 11px;
  }
  .seg-list { flex: 1; overflow-y: auto; }
  .seg-item {
    padding: 12px 16px; border-bottom: 1px solid var(--surface2);
    cursor: pointer; transition: all .1s; position: relative;
  }
  .seg-item:hover { background: var(--surface2); }
  .seg-item.active {
    background: var(--accent-bg); border-left: 5px solid var(--accent);
    padding-left: 11px;
  }
  .seg-item .seg-time {
    font-size: 11px; color: var(--accent); font-weight: 700;
    font-family: 'Space Mono', monospace; font-variant-numeric: tabular-nums;
    margin-bottom: 4px;
  }
  .seg-item .seg-text { font-size: 13px; color: var(--text); line-height: 1.5; }
  .seg-item .seg-keys { margin-top: 6px; display: flex; gap: 4px; flex-wrap: wrap; }
  .seg-item .seg-key-thumb {
    width: 36px; height: 22px; object-fit: cover;
    border: 2px solid var(--key); display: block;
  }
  .seg-item .seg-key-count {
    font-size: 10px; font-weight: 700; color: var(--key);
    font-family: 'Space Mono', monospace; margin-top: 4px;
  }
  .seg-item.has-keys .seg-time::after {
    content: ''; display: inline-block; width: 8px; height: 8px;
    background: var(--key); margin-left: 6px; vertical-align: middle;
  }

  .frames-area { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
  .frames-header {
    padding: 14px 20px; border-bottom: 2px solid var(--border);
    background: var(--surface2); flex-shrink: 0;
    display: flex; justify-content: space-between; align-items: center;
  }
  .frames-header .segment-title { font-size: 14px; font-weight: 700; }
  .frames-header .segment-title .time {
    color: var(--accent); font-family: 'Space Mono', monospace; font-size: 12px;
  }
  .frames-header .nav-btns { display: flex; gap: 6px; }
  .nav-btn {
    padding: 6px 14px; border: 2px solid var(--border); background: var(--surface);
    font-size: 12px; font-weight: 700; cursor: pointer; font-family: 'Space Mono', monospace;
    transition: all .1s;
  }
  .nav-btn:hover { background: var(--accent-bg); }
  .nav-btn:disabled { opacity: .3; cursor: not-allowed; }

  .frames-scroll { flex: 1; overflow-y: auto; padding: 16px 20px; }
  .no-segment-msg {
    display: flex; align-items: center; justify-content: center;
    height: 100%; color: var(--text2); font-size: 16px; font-weight: 500;
  }
  .frame-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 10px;
  }
  .frame-card {
    position: relative; overflow: hidden; border: var(--bw) solid var(--border);
    cursor: pointer; transition: all .15s; background: var(--surface);
  }
  .frame-card:hover { transform: translate(-1px, -1px); box-shadow: 5px 5px 0 var(--border); }
  .frame-card.selected {
    border-color: var(--key); border-width: 4px; background: var(--key-bg);
    box-shadow: 0 0 0 2px var(--key);
  }
  .frame-card img { width: 100%; display: block; }
  .frame-card .ts-bar {
    padding: 6px 8px; background: var(--surface);
    border-top: 2px solid var(--border); display: flex; justify-content: space-between; align-items: center;
  }
  .frame-card .ts {
    font-size: 12px; font-weight: 700; font-variant-numeric: tabular-nums;
    font-family: 'Space Mono', monospace; color: var(--text);
  }
  .frame-card .key-badge {
    font-size: 9px; font-weight: 800; text-transform: uppercase;
    background: var(--key); color: #000; padding: 2px 6px;
    border: 2px solid var(--border); letter-spacing: .5px;
    font-family: 'Space Mono', monospace; display: none;
  }
  .frame-card.selected .key-badge { display: inline-block; }

  .bottom-bar {
    position: fixed; bottom: 0; left: 0; right: 0;
    background: var(--surface); border-top: var(--bw) solid var(--border);
    padding: 12px 24px; z-index: 50; flex-shrink: 0;
  }
  .bottom-inner { display: flex; justify-content: space-between; align-items: center; }
  .selection-info { display: flex; align-items: center; gap: 16px; }
  .selection-count {
    font-size: 15px; font-weight: 700; font-family: 'Space Mono', monospace;
  }
  .selection-count .num { color: var(--key); font-size: 22px; }
  .progress-dots { display: flex; gap: 4px; align-items: center; }
  .progress-dot {
    width: 10px; height: 10px; border: 2px solid var(--border);
    background: var(--surface2); transition: all .15s;
  }
  .progress-dot.done { background: var(--key); }
  .progress-dot.active { background: var(--accent); transform: scale(1.2); }
  .bottom-actions { display: flex; gap: 10px; }
  .btn {
    padding: 10px 24px; border: var(--bw) solid var(--border); font-size: 13px;
    font-weight: 700; cursor: pointer; transition: all .1s; box-shadow: 3px 3px 0 var(--border);
    text-transform: uppercase; font-family: 'Space Mono', monospace; letter-spacing: .3px;
    text-decoration: none; display: inline-flex; align-items: center; gap: 6px;
  }
  .btn:hover { transform: translate(-1px, -1px); box-shadow: 4px 4px 0 var(--border); }
  .btn:active { transform: translate(2px, 2px); box-shadow: 1px 1px 0 var(--border); }
  .btn-key { background: var(--key); color: #fff; }
  .btn-outline { background: var(--surface); color: var(--text); }
  .btn:disabled { opacity: .4; cursor: not-allowed; transform: none; }

  .cluster-bar {
    padding: 10px 20px; border-bottom: 2px solid var(--border); background: var(--surface);
    display: flex; gap: 8px; align-items: center; flex-wrap: wrap; flex-shrink: 0;
  }
  .cluster-bar .bar-label {
    font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;
    color: var(--text2); font-family: 'Space Mono', monospace; margin-right: 4px;
  }
  .cluster-chip {
    display: inline-flex; align-items: center; gap: 6px; padding: 4px 6px;
    border: 2px solid var(--border); cursor: pointer; transition: all .1s;
    background: var(--surface); position: relative;
  }
  .cluster-chip:hover { transform: translate(-1px,-1px); box-shadow: 3px 3px 0 var(--border); }
  .cluster-chip.suppressed { opacity: .35; border-style: dashed; }
  .cluster-chip.suppressed:hover { opacity: .6; }
  .cluster-chip img { width: 40px; height: 26px; object-fit: cover; display: block; border: 1px solid var(--border); }
  .cluster-chip .chip-count {
    font-size: 10px; font-weight: 700; font-family: 'Space Mono', monospace; color: var(--text2);
  }
  .cluster-chip .chip-x {
    font-size: 11px; font-weight: 900; color: var(--accent2); margin-left: 2px;
  }

  .toast {
    position: fixed; bottom: 80px; right: 24px; background: var(--success); color: #fff;
    padding: 12px 20px; border: var(--bw) solid var(--border); font-weight: 700; font-size: 14px;
    transform: translateY(80px); opacity: 0; transition: all .2s; z-index: 60;
    box-shadow: var(--shadow); font-family: 'Space Mono', monospace;
  }
  .toast.show { transform: translateY(0); opacity: 1; }
</style>
</head>
<body>

<div class="top-header">
  <div class="left">
    <span class="logo">video2ai</span>
    <h1 id="videoTitle"></h1>
  </div>
  <div class="meta" id="metaTags"></div>
</div>

<div class="main-layout">
  <div class="sidebar">
    <div class="sidebar-header">
      Transcript
      <span class="seg-count" id="segCountLabel"></span>
    </div>
    <div class="seg-list" id="segList"></div>
  </div>

  <div class="frames-area">
    <div class="frames-header">
      <div class="segment-title" id="segTitle">Click a transcript segment to begin</div>
      <div class="nav-btns">
        <button class="nav-btn" id="prevBtn" onclick="navSegment(-1)" disabled>&larr; Prev</button>
        <button class="nav-btn" id="nextBtn" onclick="navSegment(1)" disabled>Next &rarr;</button>
      </div>
    </div>
    <div class="cluster-bar" id="clusterBar" style="display:none">
      <span class="bar-label">Visual themes:</span>
    </div>
    <div class="frames-scroll" id="framesScroll">
      <div class="no-segment-msg" id="noSegMsg">Select a transcript segment on the left to see its frames</div>
      <div class="frame-grid" id="frameGrid" style="display:none"></div>
    </div>
  </div>
</div>

<div class="bottom-bar">
  <div class="bottom-inner">
    <div class="selection-info">
      <div class="selection-count">
        <span class="num" id="selCount">0</span> key frames
      </div>
      <div class="progress-dots" id="progressDots"></div>
    </div>
    <div class="bottom-actions">
      <a href="/" class="btn btn-outline">New Video</a>
      <a id="downloadBtn" class="btn btn-key" style="pointer-events:none;opacity:.4">Download HTML</a>
    </div>
  </div>
</div>

<div class="toast" id="toast"></div>

<script>
var jobId = "{{ job_id }}";
var allFrames = {{ state.frames | tojson }};
var transcript = {{ state.transcript | tojson }};
var selectedSet = new Set({{ state.key_frame_indices | tojson }});
var clusters = {{ state.get('clusters', []) | tojson }};
var suppressedClusters = new Set({{ state.get('suppressed_clusters', []) | tojson }});
var activeSegIdx = -1;

// Populate header safely
document.getElementById('videoTitle').textContent = {{ state.source | tojson }};
var metaItems = [{{ state.duration_formatted | tojson }}, {{ state.resolution | tojson }}, {{ state.total_frames }} + ' frames'];
var metaEl = document.getElementById('metaTags');
metaItems.forEach(function(txt) {
  var s = document.createElement('span');
  s.textContent = txt;
  metaEl.appendChild(s);
});

// Build segments
var segments = [];
if (transcript.length > 0) {
  segments = transcript.map(function(s, i) {
    return { idx: i, start: s.start, end: s.end, text: s.text };
  });
} else {
  var dur = allFrames.length > 0 ? allFrames[allFrames.length - 1].timestamp : 0;
  for (var t = 0; t < dur; t += 10) {
    segments.push({ idx: segments.length, start: t, end: Math.min(t + 10, dur), text: '(no speech)' });
  }
}

// Build sidebar
document.getElementById('segCountLabel').textContent = segments.length;
var segListEl = document.getElementById('segList');
segments.forEach(function(seg, i) {
  var item = document.createElement('div');
  item.className = 'seg-item';
  item.dataset.idx = i;
  item.onclick = function() { activateSegment(i); };

  var timeDiv = document.createElement('div');
  timeDiv.className = 'seg-time';
  timeDiv.textContent = fmtTs(seg.start) + ' - ' + fmtTs(seg.end);

  var textDiv = document.createElement('div');
  textDiv.className = 'seg-text';
  textDiv.textContent = seg.text;

  var keysDiv = document.createElement('div');
  keysDiv.className = 'seg-keys';
  keysDiv.id = 'segKeys' + i;

  item.appendChild(timeDiv);
  item.appendChild(textDiv);
  item.appendChild(keysDiv);
  segListEl.appendChild(item);
});

// Progress dots
var dotsEl = document.getElementById('progressDots');
for (var i = 0; i < Math.min(segments.length, 40); i++) {
  var dot = document.createElement('div');
  dot.className = 'progress-dot';
  dot.dataset.idx = i;
  dotsEl.appendChild(dot);
}

buildClusterBar();
updateCount();
updateSegmentIndicators();
if (segments.length > 0) activateSegment(0);

function activateSegment(idx) {
  if (idx < 0 || idx >= segments.length) return;
  activeSegIdx = idx;
  var seg = segments[idx];

  document.querySelectorAll('.seg-item').forEach(function(el) { el.classList.remove('active'); });
  var activeEl = document.querySelector('.seg-item[data-idx="' + idx + '"]');
  if (activeEl) {
    activeEl.classList.add('active');
    activeEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  var titleEl = document.getElementById('segTitle');
  titleEl.textContent = '';
  var timeSpan = document.createElement('span');
  timeSpan.className = 'time';
  timeSpan.textContent = fmtTs(seg.start) + ' - ' + fmtTs(seg.end);
  titleEl.appendChild(timeSpan);
  var textNode = document.createTextNode('  ' + seg.text.substring(0, 80) + (seg.text.length > 80 ? '...' : ''));
  titleEl.appendChild(textNode);

  document.getElementById('prevBtn').disabled = (idx === 0);
  document.getElementById('nextBtn').disabled = (idx === segments.length - 1);

  var grid = document.getElementById('frameGrid');
  grid.textContent = '';
  grid.style.display = 'grid';
  document.getElementById('noSegMsg').style.display = 'none';

  var segFrames = allFrames.filter(function(f) {
    return f.timestamp >= seg.start && f.timestamp <= seg.end + 0.5;
  });

  if (segFrames.length === 0) {
    grid.style.display = 'block';
    var msg = document.createElement('div');
    msg.style.cssText = 'text-align:center;color:var(--text2);padding:40px;';
    msg.textContent = 'No frames in this time range';
    grid.appendChild(msg);
    return;
  }

  segFrames.forEach(function(f) {
    var card = document.createElement('div');
    card.className = 'frame-card' + (selectedSet.has(f.index) ? ' selected' : '');
    card.dataset.index = f.index;
    card.onclick = function() { toggleFrame(card); };

    var img = document.createElement('img');
    img.src = '/files/' + jobId + '/' + f.path;
    img.loading = 'lazy';

    var bar = document.createElement('div');
    bar.className = 'ts-bar';
    var ts = document.createElement('span');
    ts.className = 'ts';
    ts.textContent = fmtTs(f.timestamp);
    var badge = document.createElement('span');
    badge.className = 'key-badge';
    badge.textContent = 'KEY';
    bar.appendChild(ts);
    bar.appendChild(badge);

    card.appendChild(img);
    card.appendChild(bar);
    grid.appendChild(card);
  });

  document.getElementById('framesScroll').scrollTop = 0;
  updateProgressDots();
}

function navSegment(dir) { activateSegment(activeSegIdx + dir); }

function toggleFrame(el) {
  var idx = parseInt(el.dataset.index);
  fetch('/result/' + jobId + '/toggle-key-frame', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({index: idx})
  }).then(function(r) { return r.json(); }).then(function(d) {
    if (!d.ok) return;
    if (d.is_key) { selectedSet.add(idx); el.classList.add('selected'); }
    else { selectedSet.delete(idx); el.classList.remove('selected'); }
    updateCount();
    updateSegmentIndicators();
    updateProgressDots();
  });
}

function updateCount() {
  var n = selectedSet.size;
  document.getElementById('selCount').textContent = n;
  var btn = document.getElementById('downloadBtn');
  if (n > 0) {
    btn.href = '/result/' + jobId + '/download';
    btn.style.pointerEvents = 'auto';
    btn.style.opacity = '1';
  } else {
    btn.removeAttribute('href');
    btn.style.pointerEvents = 'none';
    btn.style.opacity = '.4';
  }
}

function updateSegmentIndicators() {
  segments.forEach(function(seg, i) {
    var keysEl = document.getElementById('segKeys' + i);
    if (!keysEl) return;
    keysEl.textContent = '';

    var segFrames = allFrames.filter(function(f) {
      return f.timestamp >= seg.start && f.timestamp <= seg.end + 0.5 && selectedSet.has(f.index);
    });

    var segItem = document.querySelector('.seg-item[data-idx="' + i + '"]');
    if (segFrames.length > 0) {
      if (segItem) segItem.classList.add('has-keys');
      segFrames.forEach(function(f) {
        var img = document.createElement('img');
        img.className = 'seg-key-thumb';
        img.src = '/files/' + jobId + '/' + f.path;
        keysEl.appendChild(img);
      });
      var countEl = document.createElement('div');
      countEl.className = 'seg-key-count';
      countEl.textContent = segFrames.length + ' key';
      keysEl.appendChild(countEl);
    } else {
      if (segItem) segItem.classList.remove('has-keys');
    }
  });
}

function updateProgressDots() {
  document.querySelectorAll('.progress-dot').forEach(function(dot) {
    var i = parseInt(dot.dataset.idx);
    dot.classList.remove('done', 'active');
    if (i === activeSegIdx) dot.classList.add('active');
    var seg = segments[i];
    if (seg) {
      var hasKeys = allFrames.some(function(f) {
        return f.timestamp >= seg.start && f.timestamp <= seg.end + 0.5 && selectedSet.has(f.index);
      });
      if (hasKeys) dot.classList.add('done');
    }
  });
}

function fmtTs(s) {
  var m = Math.floor(s / 60), sec = Math.floor(s % 60);
  return m + ':' + String(sec).padStart(2, '0');
}

function showToast(m) {
  var t = document.getElementById('toast');
  t.textContent = m; t.classList.add('show');
  setTimeout(function() { t.classList.remove('show'); }, 2500);
}

document.addEventListener('keydown', function(e) {
  if (e.key === 'ArrowRight' || e.key === 'ArrowDown') { e.preventDefault(); navSegment(1); }
  else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') { e.preventDefault(); navSegment(-1); }
});

function buildClusterBar() {
  if (!clusters || clusters.length === 0) return;
  var bar = document.getElementById('clusterBar');
  bar.style.display = 'flex';

  clusters.sort(function(a,b) { return b.size - a.size; });
  clusters.forEach(function(c) {
    var chip = document.createElement('div');
    chip.className = 'cluster-chip' + (suppressedClusters.has(c.id) ? ' suppressed' : '');
    chip.dataset.clusterId = c.id;
    chip.title = c.size + ' frames — click to ' + (suppressedClusters.has(c.id) ? 'restore' : 'suppress');
    chip.onclick = function() { toggleCluster(c.id, chip); };

    // Find representative frame for thumbnail
    var repFrame = allFrames.find(function(f) { return f.index === c.representative_index; });
    if (repFrame) {
      var img = document.createElement('img');
      img.src = '/files/' + jobId + '/' + repFrame.path;
      chip.appendChild(img);
    }

    var count = document.createElement('span');
    count.className = 'chip-count';
    count.textContent = c.size;
    chip.appendChild(count);

    bar.appendChild(chip);
  });
}

function toggleCluster(clusterId, chipEl) {
  var isSuppressed = suppressedClusters.has(clusterId);
  var action = isSuppressed ? 'restore' : 'suppress';

  fetch('/result/' + jobId + '/apply-filter', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({cluster_id: clusterId, action: action})
  }).then(function(r) { return r.json(); }).then(function(d) {
    if (!d.ok) return;

    if (action === 'suppress') {
      suppressedClusters.add(clusterId);
      chipEl.classList.add('suppressed');
      // Remove suppressed frames from selectedSet
      var cluster = clusters.find(function(c) { return c.id === clusterId; });
      if (cluster) {
        cluster.frame_indices.forEach(function(fi) { selectedSet.delete(fi); });
      }
      showToast('Suppressed theme — ' + d.removed + ' frames removed');
    } else {
      suppressedClusters.delete(clusterId);
      chipEl.classList.remove('suppressed');
      showToast('Restored theme');
    }

    chipEl.title = (suppressedClusters.has(clusterId) ? 'Click to restore' : 'Click to suppress');
    updateCount();
    updateSegmentIndicators();
    updateProgressDots();
    // Refresh current segment view to update frame selections
    if (activeSegIdx >= 0) activateSegment(activeSegIdx);
  });
}
</script>
</body></html>"""
)


def run_web(port: int = 8910, debug: bool = False):
    """Start the web UI server."""
    import webbrowser

    print(f"Starting video2ai web UI at http://localhost:{port}")
    webbrowser.open(f"http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)
