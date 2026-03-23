# video2ai

**Turn any video into AI-ready structured content.**
Extract frames, transcribe audio, auto-detect key moments — all running locally on your Mac's Neural Engine.

> No cloud. No API keys. No PyTorch. Just Apple Silicon doing what it does best.

[![PyPI version](https://img.shields.io/pypi/v/video2ai.svg)](https://pypi.org/project/video2ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## Installation

### Prerequisites

- **macOS** (Apple Vision framework required)
- **ffmpeg** — `brew install ffmpeg`

### Option 1: pip (recommended)

```bash
pip install video2ai
```

With Apple Vision support (OCR, embeddings, classification):

```bash
pip install "video2ai[vision]"
```

### Option 2: Homebrew

```bash
brew tap sameeeeeeep/video2ai https://github.com/sameeeeeeep/video2ai.git
brew install video2ai
```

### Option 3: Download binary

Grab the latest pre-built macOS binary from [Releases](https://github.com/sameeeeeeep/video2ai/releases) — no Python required:

```bash
curl -L https://github.com/sameeeeeeep/video2ai/releases/latest/download/video2ai -o video2ai
chmod +x video2ai
sudo mv video2ai /usr/local/bin/
```

### Option 4: Install from source

```bash
git clone https://github.com/sameeeeeeep/video2ai.git && cd video2ai
pip install -e ".[vision]"
```

### Optional extras

```bash
pip install openai-whisper    # transcription (local, base model is fine)
brew install yt-dlp           # URL downloads (YouTube, Vimeo, etc.)
```

---

## The Problem

You have a video. You need an AI to understand it. But LLMs can't watch videos — they need frames + text. Manually scrubbing through to pick the right frames is tedious. Existing tools are slow, memory-hungry, or require cloud APIs.

## The Solution

```
Video → ffmpeg + Whisper + Apple Vision → structured content in seconds
```

Drop a video in. Get back:
- **Timestamped transcript** — Whisper, fully local
- **Key frames auto-selected** per transcript segment — Apple Vision Neural Engine embeddings + cosine similarity
- **Visual theme clusters** — k-means on frame embeddings, filter out talking heads, keep product shots
- **Lightweight Markdown export** — local image paths, no base64 bloat, AI reads text instantly and loads images on demand
- **Self-contained HTML export** — images embedded inline, for human viewing
- **Screen capture** — record any tab/screen directly from the browser, bypasses all platform download restrictions

## Quick Start

### Web UI

```bash
video2ai --web
# → http://localhost:8910
```

Three input modes:
- **Upload** — drag a video file
- **Paste URL** — YouTube, Threads, Vimeo, anything yt-dlp supports
- **Screen Capture** — share any browser tab or screen, record at 1fps + audio, process through the same pipeline. Works with Instagram, TikTok, Netflix — anything on screen.

### CLI

```bash
video2ai video.mp4 -o output/
```

### Claude Code Skill

```bash
# Invoke from Claude Code:
/video2ai /path/to/video.mp4
```

The skill runs the full pipeline and outputs a lightweight Markdown file that Claude can read with local image paths.

## How It Works

```
Video file / URL / Screen capture
  │
  ├─ ffmpeg ──────────── frames (1/sec, JPEG)
  │
  ├─ Whisper ─────────── transcript segments + timestamps
  │
  ├─ Apple Vision ────── 768-dim embedding per frame (Neural Engine)
  │    │
  │    ├─ per-segment ── cosine distance → visual state changes → key frame suggestions
  │    │
  │    └─ global ─────── k-means clustering → visual theme groups
  │
  ├─ Apple Vision OCR ── optional, on-demand text extraction from key frames
  │
  └─ Apple Intelligence ── on-device OCR summary via FoundationModels (auto-launches server)
```

**The key insight:** frame selection is a vector math problem, not an LLM problem. Embed every frame, embed (or timestamp-match) every transcript segment, pick the frames with the highest visual distinctiveness per segment. Runs in seconds, not minutes.

**Zero ML overhead in Python.** `VNGenerateImageFeaturePrintRequest` runs on the Neural Engine — the Python process just shuffles bytes. No PyTorch, no CLIP, no transformers loaded into RAM.

## The Workflow

1. **Upload, paste URL, or screen capture** — any input mode
2. **Pipeline runs** — probe → extract → transcribe → embed → suggest
3. **Review** — transcript sidebar, frame grid per segment, pre-selected key frames
4. **Filter by visual theme** — click to deselect/select all frames in a theme, right-click to suppress
5. **OCR (optional)** — run Apple Vision OCR on selected key frames, auto-summarized by Apple Intelligence on-device
6. **Export** — Markdown (for AI) or HTML (for humans). OCR summary included by default, raw OCR opt-in.

## Export Formats

| Format | Mode | Best for |
|---|---|---|
| **Markdown** | `Download for AI` | AI consumption — lightweight text + local image paths, ~150 lines vs 170k tokens |
| **HTML** | `Download HTML` | Human viewing — self-contained, base64 images, opens in any browser |
| **HTML (AI)** | `?mode=ai` | Compressed thumbnails, still self-contained |

## Architecture

| Module | What it does |
|---|---|
| `probe.py` | ffprobe wrapper — duration, resolution, codecs, audio detection |
| `frames.py` | ffmpeg frame extraction at configurable intervals |
| `transcribe.py` | Whisper speech-to-text, returns timed segments |
| `clip_match.py` | Apple Vision embeddings, visual change detection, k-means clustering |
| `vision.py` | Apple Vision OCR + image classification + Apple Intelligence summarization |
| `llm.py` | Ollama LLM analysis — optional, for summaries |
| `web.py` | Flask web UI — upload, URL, screen capture, review, export |
| `embed.py` | Bake metadata into video via ffmpeg |

## Why Not Just Use CLIP?

We tried. CLIP + PyTorch eats ~2GB RAM and requires loading a 600MB model. Apple Vision's `VNGenerateImageFeaturePrintRequest` runs on the Neural Engine with near-zero memory overhead — it's already on your machine, already optimized, and produces 768-dim embeddings that work great for frame similarity.

For transcript↔frame matching, we don't even need cross-modal embeddings. The transcript gives us timestamps → we know which frames belong to which segment → we pick the most visually distinct ones within each segment. Simple, fast, accurate.

## Contributing

```bash
git clone https://github.com/sameeeeeeep/video2ai.git && cd video2ai
make dev
```

## Releasing

```bash
make release VERSION=0.2.0
```

This bumps the version, commits, tags, and pushes. GitHub Actions handles PyPI publishing, binary builds, and Homebrew formula updates automatically.

## License

[MIT](LICENSE)

---

Built with [Claude Code](https://claude.ai/claude-code).
