"""LLM-driven video analysis via Ollama — transcript-first approach.

Pass 1 (LLM): Analyze transcript → break video into logical sections with topics
Pass 2 (algorithmic): Score frames using vision metadata against section content → pick key frames
Pass 3 (LLM): Synthesize overall summary

Only 2 LLM calls total. Frame selection is driven by vision data (OCR text, labels)
matched against transcript-derived section topics — fast and deterministic.
"""

import json
import re
import urllib.request
import urllib.error
from dataclasses import dataclass, field

from .frames import ExtractedFrame
from .transcribe import Segment
from .vision import FrameAnalysis


OLLAMA_URL = "http://localhost:11434"


@dataclass
class VideoSection:
    """A logical section of the video identified by the LLM."""
    title: str
    start_time: float
    end_time: float
    topic: str
    key_points: list[str] = field(default_factory=list)


@dataclass
class LLMAnalysis:
    key_frame_indices: list[int] = field(default_factory=list)
    frame_reasoning: dict[int, str] = field(default_factory=dict)
    sections: list[VideoSection] = field(default_factory=list)
    summary: str = ""
    model: str = ""


def check_ollama() -> bool:
    """Check if Ollama is running and reachable."""
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def analyze_video(
    frames: list[ExtractedFrame],
    analyses: list[FrameAnalysis],
    segments: list[Segment],
    model: str = "llama3.2",
    on_progress: "callable | None" = None,
) -> LLMAnalysis:
    """
    Transcript-first video analysis. Only 2 LLM calls.

    Pass 1 (LLM): Break transcript into logical sections
    Pass 2 (algorithmic): Score vision metadata against section content → pick key frames
    Pass 3 (LLM): Synthesize summary
    """
    if not check_ollama():
        print("Warning: Ollama not running. Skipping LLM analysis.")
        return LLMAnalysis()

    analysis_map = {a.frame_index: a for a in analyses}

    # ── Pass 1: Segment the transcript into sections (1 LLM call) ─────────
    if on_progress:
        on_progress("segmenting", 0.0)

    print("  Pass 1: Analyzing transcript structure...")
    sections = _pass1_segment_transcript(segments, frames, model)
    print(f"  Found {len(sections)} sections")

    if on_progress:
        on_progress("segmenting_done", 0.30)

    # If no transcript or segmentation failed, fall back to time-based chunks
    if not sections:
        duration = frames[-1].timestamp if frames else 0
        sections = _fallback_sections(duration)

    # ── Pass 2: Vision-based key frame selection (0 LLM calls) ────────────
    if on_progress:
        on_progress("selecting", 0.35)

    print(f"  Pass 2: Scoring frames by vision metadata across {len(sections)} sections...")
    all_key_indices = []
    all_reasoning = {}

    for i, section in enumerate(sections):
        section_frames = [
            f for f in frames
            if section.start_time <= f.timestamp <= section.end_time
        ]
        if not section_frames:
            continue

        section_segments = [
            s for s in segments
            if s.start < section.end_time and s.end > section.start_time
        ]

        result = _pass2_score_frames(
            section=section,
            section_frames=section_frames,
            section_segments=section_segments,
            analysis_map=analysis_map,
        )

        print(f"    [{i+1}/{len(sections)}] {section.title} — {len(result['key_frames'])} key frames")
        all_key_indices.extend(result["key_frames"])
        all_reasoning.update(result["reasoning"])

    print(f"  Selected {len(all_key_indices)} key frames total")

    if on_progress:
        on_progress("summarizing", 0.70)

    # ── Pass 3: Synthesize overall summary (1 LLM call) ───────────────────
    print("  Pass 3: Synthesizing summary...")
    summary = _pass3_summarize(sections, all_key_indices, all_reasoning, segments, model)

    if on_progress:
        on_progress("done", 1.0)

    return LLMAnalysis(
        key_frame_indices=sorted(set(all_key_indices)),
        frame_reasoning=all_reasoning,
        sections=sections,
        summary=summary,
        model=model,
    )


# ── Pass 1: Transcript segmentation ──────────────────────────────────────────


def _pass1_segment_transcript(
    segments: list[Segment],
    frames: list[ExtractedFrame],
    model: str,
) -> list[VideoSection]:
    """Ask the LLM to break the transcript into logical sections."""
    if not segments:
        return []

    duration = frames[-1].timestamp if frames else 0

    transcript_block = "\n".join(
        f"[{_fmt_ts(s.start)}-{_fmt_ts(s.end)}] {s.text}"
        for s in segments
    )

    prompt = f"""You are analyzing a video transcript to identify its logical structure.

The video is {_fmt_ts(duration)} long.

## Full Transcript
{transcript_block}

## Task
Break this video into logical SECTIONS based on topic changes, transitions, and content structure. Each section should represent a coherent topic or activity.

Think carefully about:
- When does the speaker change topics?
- Are there introductions, transitions, demos, Q&A sections?
- What is the main point of each section?

Respond with ONLY valid JSON:
```json
{{
  "sections": [
    {{
      "title": "Short descriptive title",
      "start_time": 0.0,
      "end_time": 45.0,
      "topic": "What this section is about in 1-2 sentences",
      "key_points": ["Main point 1", "Main point 2"]
    }}
  ]
}}
```

Rules:
- Sections must cover the ENTIRE video — no gaps
- start_time of section N+1 should equal end_time of section N
- Use actual timestamps from the transcript
- Minimum 2 sections, aim for natural topic boundaries
- Return ONLY the JSON block"""

    response = _call_ollama(prompt, model)
    if not response:
        return []

    data = _extract_json(response)
    if not data or "sections" not in data:
        return []

    sections = []
    for s in data["sections"]:
        try:
            sections.append(VideoSection(
                title=str(s.get("title", "Untitled")),
                start_time=float(s.get("start_time", 0)),
                end_time=float(s.get("end_time", 0)),
                topic=str(s.get("topic", "")),
                key_points=[str(p) for p in s.get("key_points", [])],
            ))
        except (ValueError, TypeError):
            continue

    # Validate: sections should cover the video
    if sections and sections[-1].end_time < duration * 0.8:
        sections[-1].end_time = duration

    return sections


def _fallback_sections(duration: float) -> list[VideoSection]:
    """Create time-based sections when transcript segmentation fails."""
    chunk = 60.0  # 1-minute chunks
    sections = []
    t = 0.0
    i = 1
    while t < duration:
        end = min(t + chunk, duration)
        sections.append(VideoSection(
            title=f"Segment {i}",
            start_time=t,
            end_time=end,
            topic=f"Video content from {_fmt_ts(t)} to {_fmt_ts(end)}",
        ))
        t = end
        i += 1
    return sections


# ── Pass 2: Vision-based key frame scoring (no LLM) ─────────────────────────


def _pass2_score_frames(
    section: VideoSection,
    section_frames: list[ExtractedFrame],
    section_segments: list[Segment],
    analysis_map: dict[int, FrameAnalysis],
) -> dict:
    """Score and select key frames using vision metadata — no LLM call.

    Strategy:
    1. Build keyword set from section topic, key_points, and transcript
    2. Score each frame by: OCR text relevance + label relevance + visual change
    3. Detect visual state changes (OCR text differences between consecutive frames)
    4. Pick the best frame per visual state, plus first/last for section arc
    """
    valid_indices = [f.index for f in section_frames]
    result = {"key_frames": [], "reasoning": {}}

    # Build search terms from section content
    keywords = _extract_keywords(section, section_segments)

    # Score every frame
    scored_frames = []
    prev_ocr = ""
    for f in section_frames:
        a = analysis_map.get(f.index)
        score = 0.0
        reasons = []

        ocr = (a.ocr_text or "") if a else ""
        labels = (a.labels or []) if a else []
        ocr_lower = ocr.lower()
        labels_lower = " ".join(labels).lower()

        # 1. OCR text relevance — does the on-screen text match section content?
        keyword_hits = sum(1 for kw in keywords if kw in ocr_lower or kw in labels_lower)
        if keyword_hits:
            score += keyword_hits * 2.0
            reasons.append(f"{keyword_hits} keyword matches")

        # 2. Has meaningful OCR text at all (content-rich frame)
        if len(ocr) > 20:
            score += 1.5
            reasons.append("has on-screen text")

        # 3. Visual state change — OCR text differs significantly from previous frame
        if prev_ocr and ocr:
            similarity = _text_similarity(prev_ocr, ocr)
            if similarity < 0.5:
                score += 3.0
                reasons.append("visual state change")
        elif not prev_ocr and ocr:
            score += 2.0
            reasons.append("new text appears")
        elif prev_ocr and not ocr:
            score += 1.5
            reasons.append("text disappears")

        # 4. Label relevance
        if labels:
            score += 0.5
            reasons.append(f"labels: {', '.join(labels[:3])}")

        prev_ocr = ocr
        scored_frames.append((f.index, score, reasons, ocr))

    # Group frames into visual states by OCR similarity
    visual_states = _detect_visual_states(scored_frames)

    # Pick the highest-scoring frame per visual state
    for state_frames in visual_states:
        best_idx, best_score, best_reasons, _ = max(state_frames, key=lambda x: x[1])
        result["key_frames"].append(best_idx)
        reason = " | ".join(best_reasons) if best_reasons else "representative frame"
        result["reasoning"][best_idx] = reason

    # Ensure minimum 2 frames (first + last for section arc)
    return _ensure_min_frames(result, valid_indices, section)


def _extract_keywords(section: VideoSection, section_segments: list[Segment]) -> list[str]:
    """Extract meaningful keywords from section metadata and transcript."""
    text_parts = [section.title, section.topic]
    text_parts.extend(section.key_points)
    for seg in section_segments:
        text_parts.append(seg.text)

    combined = " ".join(text_parts).lower()

    # Split into words, keep meaningful ones (>3 chars, not stopwords)
    stopwords = {
        "this", "that", "with", "from", "have", "will", "been", "were",
        "they", "them", "their", "what", "when", "where", "which", "about",
        "into", "your", "more", "some", "than", "very", "just", "also",
        "then", "each", "only", "over", "such", "most", "like", "well",
        "here", "there", "could", "would", "should", "does", "doing",
        "being", "going", "want", "know", "think", "make", "take",
        "come", "look", "give", "good", "thing", "right", "don't",
        "it's", "i'm", "we're", "you're", "can't", "won't", "let's",
        "gonna", "gotta", "wanna", "really", "actually", "basically",
    }

    words = re.findall(r"[a-z]{4,}", combined)
    keywords = [w for w in set(words) if w not in stopwords]
    return keywords


def _text_similarity(text1: str, text2: str) -> float:
    """Quick word-overlap similarity between two texts. Returns 0.0–1.0."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union)


def _detect_visual_states(
    scored_frames: list[tuple[int, float, list[str], str]],
) -> list[list[tuple[int, float, list[str], str]]]:
    """Group consecutive frames into visual states based on OCR similarity.

    Frames with similar OCR text are grouped together. A new state starts
    when OCR content changes significantly.
    """
    if not scored_frames:
        return []

    states = [[scored_frames[0]]]

    for frame in scored_frames[1:]:
        _, _, _, curr_ocr = frame
        _, _, _, prev_ocr = states[-1][-1]

        # Both empty — same state (likely same scene without text)
        if not curr_ocr and not prev_ocr:
            # But cap groups at ~5 frames to avoid huge groups of empty frames
            if len(states[-1]) < 5:
                states[-1].append(frame)
            else:
                states.append([frame])
        elif not curr_ocr or not prev_ocr:
            # One has text, other doesn't — new state
            states.append([frame])
        elif _text_similarity(curr_ocr, prev_ocr) > 0.6:
            # Similar text — same visual state
            states[-1].append(frame)
        else:
            # Different text — new visual state
            states.append([frame])

    return states


def _ensure_min_frames(
    result: dict,
    valid_indices: list[int],
    section: "VideoSection",
) -> dict:
    """Ensure at least 2 key frames per section: first (setup) + last (result).

    If only 1 frame exists in the section, that single frame is used.
    """
    if not valid_indices:
        return result

    first_idx = valid_indices[0]
    last_idx = valid_indices[-1]

    existing = set(result["key_frames"])

    if len(valid_indices) == 1:
        # Section has only 1 frame — use it
        if first_idx not in existing:
            result["key_frames"].insert(0, first_idx)
            result["reasoning"][first_idx] = f"Only frame in section: {section.title}"
    else:
        # Ensure first frame (setup) is included
        if first_idx not in existing:
            result["key_frames"].insert(0, first_idx)
            result["reasoning"][first_idx] = f"Opening frame — establishes context for: {section.title}"
        # Ensure last frame (conclusion) is included
        if last_idx not in existing:
            result["key_frames"].append(last_idx)
            result["reasoning"][last_idx] = f"Closing frame — shows conclusion of: {section.title}"

    return result


# ── Pass 3: Summary synthesis ─────────────────────────────────────────────────


def _pass3_summarize(
    sections: list[VideoSection],
    key_indices: list[int],
    reasoning: dict[int, str],
    segments: list[Segment],
    model: str,
) -> str:
    """Generate an overall summary from the section analysis."""
    sections_block = "\n".join(
        f"- **{s.title}** ({_fmt_ts(s.start_time)}-{_fmt_ts(s.end_time)}): {s.topic}"
        for s in sections
    )

    transcript_block = ""
    if segments:
        transcript_block = "\n\n## Transcript Highlights\n" + "\n".join(
            f"[{_fmt_ts(s.start)}] {s.text}" for s in segments[:50]
        )

    prompt = f"""Based on the following video section analysis, write a comprehensive but concise summary of the entire video.

## Video Sections
{sections_block}
{transcript_block}

## Key Visual Moments
{len(key_indices)} key frames were identified across {len(sections)} sections.

## Task
Write a 3-5 sentence summary that captures:
1. What the video is about overall
2. The main topics/sections covered
3. Key takeaways or important information

Respond with ONLY the summary text, no JSON, no formatting. Just plain text."""

    response = _call_ollama(prompt, model)
    return response.strip() if response else ""


# ── Helpers ───────────────────────────────────────────────────────────────────


def _call_ollama(prompt: str, model: str) -> str:
    """Call Ollama API and return the response text."""
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 4096,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "")
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        print(f"Warning: Ollama call failed: {e}")
        return ""


def _extract_json(response: str) -> dict | None:
    """Extract JSON from LLM response, handling code blocks and common issues."""
    # Try markdown code block first
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try raw JSON (find the outermost braces)
        depth = 0
        start = None
        for i, c in enumerate(response):
            if c == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0 and start is not None:
                    json_str = response[start:i+1]
                    break
        else:
            json_str = response.strip()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            cleaned = re.sub(r",\s*([}\]])", r"\1", json_str)
            return json.loads(cleaned)
        except (json.JSONDecodeError, UnboundLocalError):
            print(f"Warning: Could not parse LLM response as JSON")
            return None


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"
