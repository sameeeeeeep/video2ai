# video2ai

**Turn any video into AI-ready structured content.** Extract frames, transcribe audio, auto-detect key moments — all running locally on Apple Silicon.

---

### What it does

Drop a video in. Get back:
- Timestamped transcript (Whisper, local)
- Key frames auto-selected per transcript segment
- Visual theme clusters for instant filtering
- Self-contained HTML export with embedded images + transcript

### How it works

```
Video
  |
  +-- ffmpeg --> frames (1/sec)
  |
  +-- Whisper --> transcript segments with timestamps
  |
  +-- Apple Vision (Neural Engine) --> 768-dim embedding per frame
       |
       +-- cosine distance --> visual state change detection --> key frame suggestions
       |
       +-- k-means clustering --> visual theme groups (person, screen, product, etc.)
```

**Zero Python ML overhead.** Frame embeddings run on the Neural Engine via `VNGenerateImageFeaturePrintRequest` — near-zero CPU/RAM. No PyTorch, no CLIP, no transformers.

### The workflow

1. **Upload or paste a URL** (YouTube, Vimeo, etc. via yt-dlp)
2. Pipeline runs: probe → extract frames → transcribe → embed → suggest key frames
3. **Review page**: transcript sidebar + frame grid per segment
4. **Visual theme filter**: cluster thumbnails at top — click to suppress a theme (e.g. "talking head") and keep another (e.g. "product shots")
5. **Export**: download self-contained HTML with key frames + transcript

### Install

```bash
# Clone
git clone <repo-url> && cd capabilities

# Install
pip install -e .

# Dependencies
brew install ffmpeg
pip install openai-whisper pyobjc-framework-Vision yt-dlp
```

### Run

```bash
# Web UI (recommended)
video2ai-web

# CLI
video2ai path/to/video.mp4 -o output/
```

### Requirements

- macOS (Apple Vision framework)
- Python 3.12+
- ffmpeg
- Whisper (for transcription)
- Ollama (optional, for LLM summaries)

### Architecture

| Module | Role |
|---|---|
| `probe.py` | ffprobe wrapper — duration, resolution, codecs |
| `frames.py` | ffmpeg frame extraction at configurable intervals |
| `transcribe.py` | Whisper audio transcription |
| `clip_match.py` | Apple Vision embeddings, change detection, k-means clustering |
| `vision.py` | Apple Vision OCR + image classification |
| `llm.py` | Ollama LLM analysis (optional) |
| `web.py` | Flask web UI — upload, process, review, export |
| `embed.py` | ffmpeg metadata embedding into video files |

---

Built with Claude Code.
