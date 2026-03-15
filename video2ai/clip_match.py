"""Apple Vision frame embedding + visual change detection.

Uses macOS VNGenerateImageFeaturePrintRequest to embed frames via the Neural
Engine — near-zero CPU/memory overhead.  Detects visual state changes by
cosine distance between consecutive frame embeddings, then picks the most
representative frame per state.  Works on ANY video, with or without transcript.

When a transcript exists, suggestions are matched per-segment: for each segment's
time range, we find the visually distinct frames and pick the best representatives.
Without transcript, segments are auto-generated from detected scene changes.
"""

import math
import os
import platform
import struct
from dataclasses import dataclass

from .frames import ExtractedFrame
from .transcribe import Segment


@dataclass
class FrameSuggestion:
    frame_index: int
    timestamp: float
    score: float
    segment_index: int


def is_available() -> bool:
    """Check if Apple Vision framework is available (macOS only)."""
    if platform.system() != "Darwin":
        return False
    try:
        import Vision  # noqa: F401
        return True
    except ImportError:
        return False


def suggest_key_frames(
    frames: list[ExtractedFrame],
    segments: list[Segment] | None = None,
    top_k: int = 3,
    on_progress: "callable | None" = None,
) -> list[FrameSuggestion]:
    """Suggest key frames using Apple Vision embeddings + change detection.

    Always works — transcript is optional.  Steps:
    1. Embed every frame via VNGenerateImageFeaturePrintRequest (Neural Engine)
    2. If transcript segments exist: per-segment visual analysis
    3. If no transcript: global visual state detection
    """
    if not frames:
        return []

    if on_progress:
        on_progress("embedding", 0.0)

    print("  Embedding frames via Apple Vision (Neural Engine)...")
    embeddings = _embed_frames(frames, on_progress)

    if not embeddings:
        print("  Warning: embedding failed, falling back to uniform sampling")
        return _uniform_fallback(frames, segments)

    if on_progress:
        on_progress("matching", 0.6)

    if segments:
        print(f"  Matching frames to {len(segments)} transcript segments...")
        suggestions = _match_per_segment(frames, embeddings, segments, top_k)
    else:
        print("  No transcript — detecting visual states globally...")
        suggestions = _detect_global_states(frames, embeddings)

    print(f"  Suggested {len(suggestions)} key frames")

    if on_progress:
        on_progress("done", 1.0)

    return suggestions


# ── Per-segment matching (transcript exists) ──────────────────────────────────


def _match_per_segment(
    frames: list[ExtractedFrame],
    embeddings: list[list[float]],
    segments: list[Segment],
    top_k: int,
) -> list[FrameSuggestion]:
    """For each transcript segment, find the most visually distinct frames.

    Strategy per segment:
    1. Gather all frames within the segment's time range
    2. Compute pairwise distances to find visual diversity
    3. Pick the most visually distinct frame (highest avg distance from others)
       — this is the frame that "stands out" visually
    4. Also pick the frame with the biggest change from its predecessor
       — this captures the key transition moment
    5. Return up to top_k frames per segment
    """
    suggestions = []

    for seg_idx, seg in enumerate(segments):
        # Find frames in this segment's time range (with small buffer)
        seg_frames = []
        seg_embs = []
        for i, f in enumerate(frames):
            if seg.start - 0.5 <= f.timestamp <= seg.end + 0.5:
                seg_frames.append((i, f))
                seg_embs.append(embeddings[i])

        if not seg_frames:
            continue

        if len(seg_frames) == 1:
            # Only one frame — use it
            idx, f = seg_frames[0]
            suggestions.append(FrameSuggestion(
                frame_index=f.index, timestamp=f.timestamp,
                score=1.0, segment_index=seg_idx,
            ))
            continue

        # Score each frame by visual distinctiveness within the segment
        scored = []
        for j in range(len(seg_frames)):
            # Average distance to all other frames in segment
            avg_dist = 0.0
            for k in range(len(seg_frames)):
                if k != j:
                    avg_dist += _cosine_distance(seg_embs[j], seg_embs[k])
            avg_dist /= (len(seg_frames) - 1)

            # Distance from previous frame (change magnitude)
            change_score = 0.0
            if j > 0:
                change_score = _cosine_distance(seg_embs[j - 1], seg_embs[j])

            # Combined score: visual distinctiveness + change detection
            score = 0.6 * avg_dist + 0.4 * change_score
            scored.append((j, score))

        # Sort by score, pick top_k
        scored.sort(key=lambda x: x[1], reverse=True)
        picked = set()
        for j, score in scored[:top_k]:
            idx, f = seg_frames[j]
            if f.index not in picked:
                suggestions.append(FrameSuggestion(
                    frame_index=f.index, timestamp=f.timestamp,
                    score=score, segment_index=seg_idx,
                ))
                picked.add(f.index)

        # Always ensure at least the first frame of the segment is included
        first_idx, first_f = seg_frames[0]
        if first_f.index not in picked:
            suggestions.append(FrameSuggestion(
                frame_index=first_f.index, timestamp=first_f.timestamp,
                score=0.3, segment_index=seg_idx,
            ))

    suggestions.sort(key=lambda s: s.timestamp)
    return suggestions


# ── Global visual state detection (no transcript) ────────────────────────────


def _detect_global_states(
    frames: list[ExtractedFrame],
    embeddings: list[list[float]],
) -> list[FrameSuggestion]:
    """Detect visual state changes globally when no transcript exists."""
    # Compute consecutive cosine distances
    distances = []
    for i in range(len(embeddings) - 1):
        d = _cosine_distance(embeddings[i], embeddings[i + 1])
        distances.append(d)

    # Find change points
    change_indices = _detect_change_points(distances)

    # Build visual state groups
    boundaries = [0] + [i + 1 for i in change_indices] + [len(frames)]
    states = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        if start < end:
            states.append((start, end))

    # Pick the most representative frame per state (closest to centroid)
    suggestions = []
    for state_idx, (start, end) in enumerate(states):
        state_embeddings = embeddings[start:end]
        centroid = [0.0] * len(state_embeddings[0])
        for emb in state_embeddings:
            for j in range(len(centroid)):
                centroid[j] += emb[j]
        n = len(state_embeddings)
        centroid = [c / n for c in centroid]

        best_i = start
        best_dist = float("inf")
        for i in range(start, end):
            d = _cosine_distance(embeddings[i], centroid)
            if d < best_dist:
                best_dist = d
                best_i = i

        f = frames[best_i]
        suggestions.append(FrameSuggestion(
            frame_index=f.index, timestamp=f.timestamp,
            score=1.0 - best_dist, segment_index=state_idx,
        ))

    return suggestions


# ── Apple Vision embedding ────────────────────────────────────────────────────


def _embed_frames(
    frames: list[ExtractedFrame],
    on_progress: "callable | None" = None,
) -> list[list[float]]:
    """Embed frames using VNGenerateImageFeaturePrintRequest."""
    try:
        import Vision
        from Foundation import NSURL
    except ImportError:
        return []

    embeddings = []
    total = len(frames)

    for i, frame in enumerate(frames):
        if not os.path.isfile(frame.path):
            embeddings.append(None)
            continue

        try:
            url = NSURL.fileURLWithPath_(frame.path)
            request = Vision.VNGenerateImageFeaturePrintRequest.alloc().init()
            handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, None)
            success = handler.performRequests_error_([request], None)

            if not success[0]:
                embeddings.append(None)
                continue

            results = request.results()
            if not results:
                embeddings.append(None)
                continue

            fp = results[0]
            count = fp.elementCount()
            raw = bytes(fp.data())
            vec = list(struct.unpack(f"<{count}f", raw))
            embeddings.append(vec)
        except Exception:
            embeddings.append(None)

        if on_progress and (i + 1) % 10 == 0:
            on_progress("embedding", 0.05 + 0.55 * (i + 1) / total)

    # Fill None gaps with nearest valid embedding
    last_valid = None
    for i in range(len(embeddings)):
        if embeddings[i] is not None:
            last_valid = embeddings[i]
        elif last_valid is not None:
            embeddings[i] = last_valid

    # Backfill start if needed
    first_valid = next((e for e in embeddings if e is not None), None)
    if first_valid:
        for i in range(len(embeddings)):
            if embeddings[i] is None:
                embeddings[i] = first_valid
            else:
                break

    return embeddings


# ── Change detection ──────────────────────────────────────────────────────────


def _cosine_distance(a: list[float], b: list[float]) -> float:
    """Cosine distance between two vectors. Returns 0.0 (identical) to 1.0."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 1.0
    similarity = dot / (norm_a * norm_b)
    return max(0.0, 1.0 - similarity)


def _detect_change_points(distances: list[float]) -> list[int]:
    """Find indices where visual state changes occur.

    Uses adaptive thresholding: a change happens when the distance between
    consecutive frames is significantly above the local median.
    """
    if not distances:
        return []

    sorted_d = sorted(distances)
    median = sorted_d[len(sorted_d) // 2]
    deviations = sorted(abs(d - median) for d in distances)
    mad = deviations[len(deviations) // 2] if deviations else 0.0

    threshold = median + 2.0 * mad
    threshold = max(threshold, 0.05)

    change_points = []
    for i, d in enumerate(distances):
        if d > threshold:
            if not change_points or (i - change_points[-1]) >= 2:
                change_points.append(i)

    return change_points


# ── Fallback ──────────────────────────────────────────────────────────────────


def _uniform_fallback(
    frames: list[ExtractedFrame],
    segments: list[Segment] | None,
) -> list[FrameSuggestion]:
    """Fallback when embedding fails — pick evenly spaced frames."""
    if not frames:
        return []
    step = max(1, len(frames) // 10)
    suggestions = []
    for i in range(0, len(frames), step):
        f = frames[i]
        seg_idx = _find_segment(f.timestamp, segments) if segments else i
        suggestions.append(FrameSuggestion(
            frame_index=f.index, timestamp=f.timestamp,
            score=0.5, segment_index=seg_idx,
        ))
    return suggestions


def _find_segment(timestamp: float, segments: list[Segment]) -> int:
    """Find the segment index that contains the given timestamp."""
    for i, seg in enumerate(segments):
        if seg.start - 0.5 <= timestamp <= seg.end + 0.5:
            return i
    best_i = 0
    best_d = float("inf")
    for i, seg in enumerate(segments):
        mid = (seg.start + seg.end) / 2
        d = abs(timestamp - mid)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


# ── Visual theme clustering ──────────────────────────────────────────────────


def cluster_frames(
    frames: list[ExtractedFrame],
    n_clusters: int = 0,
) -> dict:
    """Cluster frames into visual themes using their Apple Vision embeddings.

    Returns:
        {
            "clusters": [
                {
                    "id": 0,
                    "representative_index": 5,  # frame index of centroid
                    "frame_indices": [1, 2, 3, 5, 7],
                    "size": 5,
                }
            ],
            "frame_cluster_map": {1: 0, 2: 0, 3: 0, ...},  # frame_index -> cluster_id
            "embeddings_cache": [...]  # for reuse
        }
    """
    if not frames:
        return {"clusters": [], "frame_cluster_map": {}, "embeddings_cache": []}

    print("  Clustering frames by visual theme...")
    embeddings = _embed_frames(frames)
    if not embeddings:
        return {"clusters": [], "frame_cluster_map": {}, "embeddings_cache": []}

    # Auto-determine cluster count if not specified
    if n_clusters <= 0:
        # Heuristic: sqrt(n_frames), clamped between 3 and 12
        n_clusters = max(3, min(12, int(math.sqrt(len(frames)))))

    # Simple k-means clustering (no scipy/sklearn needed)
    assignments, centroids = _kmeans(embeddings, n_clusters, max_iter=20)

    # Build cluster info
    cluster_frames_map = {}
    for i, cluster_id in enumerate(assignments):
        cluster_frames_map.setdefault(cluster_id, []).append(i)

    clusters = []
    for cid in sorted(cluster_frames_map.keys()):
        members = cluster_frames_map[cid]
        # Find representative: frame closest to centroid
        best_i = members[0]
        best_dist = float("inf")
        for mi in members:
            d = _cosine_distance(embeddings[mi], centroids[cid])
            if d < best_dist:
                best_dist = d
                best_i = mi

        clusters.append({
            "id": cid,
            "representative_index": frames[best_i].index,
            "frame_indices": [frames[mi].index for mi in members],
            "size": len(members),
        })

    frame_cluster_map = {}
    for c in clusters:
        for fi in c["frame_indices"]:
            frame_cluster_map[fi] = c["id"]

    print(f"  Found {len(clusters)} visual themes: " +
          ", ".join(f"#{c['id']}({c['size']})" for c in clusters))

    return {
        "clusters": clusters,
        "frame_cluster_map": frame_cluster_map,
        "embeddings_cache": embeddings,
    }


def filter_by_clusters(
    suggestions: list[FrameSuggestion],
    frame_cluster_map: dict[int, int],
    suppressed_clusters: set[int],
    boosted_clusters: set[int] | None = None,
) -> list[FrameSuggestion]:
    """Re-filter suggestions by suppressing/boosting visual theme clusters.

    Suppressed clusters: keep only 1 frame per segment (lowest scored).
    Boosted clusters: double their score to prioritize them.
    """
    if not suppressed_clusters and not boosted_clusters:
        return suggestions

    boosted_clusters = boosted_clusters or set()
    filtered = []

    for s in suggestions:
        cid = frame_cluster_map.get(s.frame_index, -1)
        if cid in suppressed_clusters:
            continue  # drop frames from suppressed clusters
        new_score = s.score
        if cid in boosted_clusters:
            new_score = min(1.0, s.score * 1.5)
        filtered.append(FrameSuggestion(
            frame_index=s.frame_index,
            timestamp=s.timestamp,
            score=new_score,
            segment_index=s.segment_index,
        ))

    return filtered


def _kmeans(
    vectors: list[list[float]],
    k: int,
    max_iter: int = 20,
) -> tuple[list[int], list[list[float]]]:
    """Simple k-means clustering. Returns (assignments, centroids)."""
    import random
    n = len(vectors)
    dim = len(vectors[0])

    # Initialize centroids with k-means++ style: spread-out initial picks
    centroids = [vectors[random.randint(0, n - 1)][:]]
    for _ in range(k - 1):
        # Pick point with max min-distance to existing centroids
        best_idx = 0
        best_min_dist = -1.0
        for i in range(n):
            min_dist = min(_cosine_distance(vectors[i], c) for c in centroids)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = i
        centroids.append(vectors[best_idx][:])

    assignments = [0] * n

    for _ in range(max_iter):
        # Assign each point to nearest centroid
        changed = False
        for i in range(n):
            best_c = 0
            best_d = float("inf")
            for c in range(len(centroids)):
                d = _cosine_distance(vectors[i], centroids[c])
                if d < best_d:
                    best_d = d
                    best_c = c
            if assignments[i] != best_c:
                assignments[i] = best_c
                changed = True

        if not changed:
            break

        # Recompute centroids
        for c in range(len(centroids)):
            members = [i for i in range(n) if assignments[i] == c]
            if not members:
                continue
            new_centroid = [0.0] * dim
            for mi in members:
                for j in range(dim):
                    new_centroid[j] += vectors[mi][j]
            centroids[c] = [v / len(members) for v in new_centroid]

    return assignments, centroids
