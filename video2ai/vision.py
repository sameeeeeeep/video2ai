"""Apple Vision framework integration for OCR and image analysis on macOS."""
from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field


@dataclass
class FrameAnalysis:
    frame_index: int
    timestamp: float
    ocr_text: str = ""
    labels: list[str] = field(default_factory=list)


def is_available() -> bool:
    """Check if Apple Vision framework is available (macOS only)."""
    if platform.system() != "Darwin":
        return False
    try:
        import Vision  # noqa: F401
        return True
    except ImportError:
        return False


def analyze_frames(frames: list, verbose: bool = True) -> list[FrameAnalysis]:
    """
    Run Apple Vision OCR and classification on extracted frames.

    Requires macOS with pyobjc-framework-Vision installed.
    Falls back gracefully if unavailable.
    """
    if not is_available():
        if verbose:
            print(
                "Apple Vision not available. "
                "Install with: pip install pyobjc-framework-Vision pyobjc-framework-CoreML"
            )
        return []

    import Vision
    from Foundation import NSURL
    from Quartz import CIImage

    if verbose:
        print(f"Analyzing {len(frames)} frames with Apple Vision...")

    results = []
    for i, frame in enumerate(frames):
        if verbose and (i + 1) % 10 == 0:
            print(f"  Analyzed {i + 1}/{len(frames)} frames...")

        analysis = FrameAnalysis(
            frame_index=frame.index,
            timestamp=frame.timestamp,
        )

        # OCR
        analysis.ocr_text = _recognize_text(frame.path, Vision, NSURL, CIImage)

        # Classification labels
        analysis.labels = _classify_image(frame.path, Vision, NSURL, CIImage)

        results.append(analysis)

    if verbose:
        texts = sum(1 for r in results if r.ocr_text)
        print(f"  Done. {texts}/{len(results)} frames had readable text.")

    return results


def _recognize_text(image_path: str, Vision, NSURL, CIImage) -> str:
    """Run OCR on a single image using VNRecognizeTextRequest."""
    try:
        url = NSURL.fileURLWithPath_(image_path)
        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(True)

        handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, None)
        success = handler.performRequests_error_([request], None)

        if not success[0]:
            return ""

        observations = request.results()
        if not observations:
            return ""

        texts = []
        for obs in observations:
            candidate = obs.topCandidates_(1)
            if candidate:
                texts.append(candidate[0].string())

        return "\n".join(texts)
    except Exception:
        return ""


def _classify_image(image_path: str, Vision, NSURL, CIImage) -> list[str]:
    """Classify image content using VNClassifyImageRequest."""
    try:
        url = NSURL.fileURLWithPath_(image_path)
        request = Vision.VNClassifyImageRequest.alloc().init()

        handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, None)
        success = handler.performRequests_error_([request], None)

        if not success[0]:
            return []

        observations = request.results()
        if not observations:
            return []

        # Keep high-confidence labels
        labels = []
        for obs in observations:
            if obs.confidence() > 0.5:
                labels.append(obs.identifier())

        return labels[:5]  # top 5
    except Exception:
        return []


def summarize_ocr_text(ocr_texts: list[str]) -> str | None:
    """Summarize OCR text from multiple frames using Apple Intelligence on-device model."""
    combined = "\n".join(t.strip() for t in ocr_texts if t.strip())
    if not combined:
        return None

    import urllib.request
    import json

    # Ensure the local Apple Intelligence server is running
    try:
        urllib.request.urlopen("http://127.0.0.1:11535/health", timeout=2)
    except Exception:
        import subprocess
        subprocess.Popen(["open", "/Applications/AppleOnDeviceOpenAI-v1.1.3.app"])
        import time
        for _ in range(10):
            time.sleep(1)
            try:
                urllib.request.urlopen("http://127.0.0.1:11535/health", timeout=2)
                break
            except Exception:
                pass

    try:
        body = json.dumps({
            "model": "apple-on-device",
            "messages": [{"role": "user", "content":
                "Summarize the following text extracted via OCR from video frames "
                "into a concise, coherent summary. Focus on key information, "
                "ignore OCR artifacts:\n\n" + combined[:4000]
            }],
        }).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:11535/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return None
