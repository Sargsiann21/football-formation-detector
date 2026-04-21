from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def _get_axis_index(camera_view: str) -> int:
    return 1 if camera_view == "long_side" else 0


def normalize_orientation(
    positions: Sequence[Tuple[float, float]],
    field_width: float = 68.0,
    gk_y: float | None = None,
    camera_view: str = "long_side",
) -> np.ndarray:
    """Normalize team direction by optionally flipping Y to a shared orientation.

    When *gk_y* (the goalkeeper's Y coordinate) is provided, use it as the
    anchor for the defensive end.  Otherwise fall back to the median heuristic.
    """
    points = np.array(positions, dtype=np.float32)

    if len(points) == 0:
        return points

    normalized = points.copy()

    axis_idx = _get_axis_index(camera_view)

    # GK is always near the team's own goal -> use as the defensive reference.
    reference_y = gk_y if gk_y is not None else float(np.median(normalized[:, axis_idx]))

    if reference_y > (field_width / 2.0):
        normalized[:, axis_idx] = field_width - normalized[:, axis_idx]

    return normalized


def remove_goalkeeper_candidate(
    positions: np.ndarray,
    enabled: bool = True,
    camera_view: str = "long_side",
) -> np.ndarray:
    """Heuristically remove one extreme-depth player as goalkeeper candidate."""
    if not enabled or len(positions) <= 10:
        return positions

    axis_idx = _get_axis_index(camera_view)
    y_values = positions[:, axis_idx]
    low_idx = int(np.argmin(y_values))
    high_idx = int(np.argmax(y_values))

    low_span = float(np.median(y_values) - y_values[low_idx])
    high_span = float(y_values[high_idx] - np.median(y_values))

    remove_idx = low_idx if low_span > high_span else high_idx

    return np.delete(positions, remove_idx, axis=0)


def cluster_player_lines(
    positions: np.ndarray,
    distance_threshold: float = 9.0,
    min_cluster_size: int = 2,
    camera_view: str = "long_side",
) -> List[Dict]:
    """Cluster players into tactical lines using 1D agglomerative clustering on depth axis."""
    if len(positions) == 0:
        return []

    axis_idx = _get_axis_index(camera_view)
    clustering_axis_values = positions[:, axis_idx]
    y_only = clustering_axis_values.reshape(-1, 1)

    if len(positions) == 1:
        return [{"indices": np.array([0]), "count": 1, "mean_y": float(y_only[0, 0])}]

    y_spread = float(np.percentile(y_only, 75) - np.percentile(y_only, 25))
    adaptive_threshold = max(5.0, min(14.0, 0.28 * y_spread + 5.0))
    threshold = float(distance_threshold) if distance_threshold is not None else adaptive_threshold

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        linkage="ward",
    )
    labels = clustering.fit_predict(y_only)

    clusters: List[Dict] = []

    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        if len(indices) < min_cluster_size:
            continue

        cluster_points = positions[indices]

        clusters.append(
            {
                "indices": indices,
                "count": int(len(indices)),
                "mean_y": float(np.mean(cluster_points[:, axis_idx])),
                "points": cluster_points,
            }
        )

    if not clusters:
        order = np.argsort(clustering_axis_values)
        split = np.array_split(order, 3)
        for split_indices in split:
            if len(split_indices) == 0:
                continue
            cluster_points = positions[split_indices]
            clusters.append(
                {
                    "indices": split_indices,
                    "count": int(len(split_indices)),
                    "mean_y": float(np.mean(cluster_points[:, axis_idx])),
                    "points": cluster_points,
                }
            )

    clusters.sort(key=lambda cluster: cluster["mean_y"])

    return clusters


def line_counts_from_clusters(clusters: Sequence[Dict]) -> List[int]:
    """Convert sorted cluster metadata into formation counts per line."""
    return [int(cluster["count"]) for cluster in clusters]


def cluster_by_gaps(
    positions: np.ndarray,
    n_lines: int,
    camera_view: str = "long_side",
) -> List[Dict]:
    """Split sorted Y-positions into *n_lines* clusters at the largest gaps.

    This produces exactly *n_lines* groups with zero threshold tuning —
    it simply finds the (n_lines - 1) biggest consecutive-Y gaps and cuts there.
    """
    if len(positions) == 0:
        return []

    axis_idx = _get_axis_index(camera_view)
    clustering_axis_values = positions[:, axis_idx]
    sorted_order = np.argsort(clustering_axis_values)
    sorted_y = clustering_axis_values[sorted_order]

    if n_lines <= 1 or len(positions) < n_lines:
        # Single line or not enough players to split
        return [{
            "indices": sorted_order,
            "count": len(sorted_order),
            "mean_y": float(np.mean(sorted_y)),
            "points": positions[sorted_order],
        }]

    gaps = np.diff(sorted_y)
    # Pick the (n_lines - 1) largest gaps as split points
    split_gap_indices = np.argsort(gaps)[-(n_lines - 1):]
    split_gap_indices = np.sort(split_gap_indices)

    boundaries = [0] + list(split_gap_indices + 1) + [len(sorted_order)]

    clusters: List[Dict] = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        original_indices = sorted_order[start:end]
        cluster_points = positions[original_indices]
        clusters.append({
            "indices": original_indices,
            "count": len(original_indices),
            "mean_y": float(np.mean(cluster_points[:, axis_idx])),
            "points": cluster_points,
        })

    return clusters


def gap_quality_score(
    positions: np.ndarray,
    clusters: List[Dict],
    camera_view: str = "long_side",
) -> float:
    """Measure how natural a given split is: big gaps between lines, small within.

    Returns a penalty (lower = better split).  A clean separation yields ≈ 0.
    """
    if len(clusters) <= 1:
        return 0.0

    axis_idx = _get_axis_index(camera_view)

    # Within-line spread: average range of depth axis inside each cluster
    within_spreads = []
    for cl in clusters:
        ys = cl["points"][:, axis_idx]
        within_spreads.append(float(np.ptp(ys)) if len(ys) > 1 else 0.0)
    mean_within = float(np.mean(within_spreads)) if within_spreads else 0.0

    # Between-line gap: min gap between consecutive cluster means
    means = sorted(cl["mean_y"] for cl in clusters)
    min_between = min(means[i + 1] - means[i] for i in range(len(means) - 1))

    # Penalty: high within-spread relative to between-gap → bad split
    if min_between < 1e-3:
        return 1.0
    return float(np.clip(mean_within / min_between, 0.0, 1.0))


def build_structure_graph(
    positions: Sequence[Tuple[float, float]],
    distance_threshold: float = 9.0,
    camera_view: str = "long_side",
) -> Dict:
    """Build graph nodes/edges using within-line and between-line connectivity."""
    points = np.array(positions, dtype=np.float32)

    if len(points) == 0:
        return {"nodes": [], "edges": [], "lines": []}

    axis_idx = _get_axis_index(camera_view)
    lateral_idx = 0 if axis_idx == 1 else 1

    clusters = cluster_player_lines(
        points,
        distance_threshold=distance_threshold,
        camera_view=camera_view,
    )

    nodes = [{"id": idx, "position": (float(p[0]), float(p[1]))} for idx, p in enumerate(points)]
    edges = set()
    lines: List[Dict] = []

    prev_line_sorted = None

    for line_idx, cluster in enumerate(clusters):
        indices = list(cluster["indices"])
        sorted_indices = sorted(indices, key=lambda i: points[i, lateral_idx])

        for i in range(len(sorted_indices) - 1):
            edge = tuple(sorted((sorted_indices[i], sorted_indices[i + 1])))
            edges.add(edge)

        if prev_line_sorted is not None:
            for i, curr_idx in enumerate(sorted_indices):
                nearest_prev = prev_line_sorted[min(i, len(prev_line_sorted) - 1)]
                edge = tuple(sorted((curr_idx, nearest_prev)))
                edges.add(edge)

        line_points = points[sorted_indices]
        lines.append(
            {
                "line_index": line_idx,
                "player_ids": sorted_indices,
                "mean_y": float(np.mean(line_points[:, axis_idx])),
                "count": int(len(sorted_indices)),
            }
        )

        prev_line_sorted = sorted_indices

    return {
        "nodes": nodes,
        "edges": [list(edge) for edge in sorted(edges)],
        "lines": lines,
    }
