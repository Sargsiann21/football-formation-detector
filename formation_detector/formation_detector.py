from collections import defaultdict, deque
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .clustering_utils import (
    build_structure_graph,
    cluster_by_gaps,
    cluster_player_lines,
    line_counts_from_clusters,
    normalize_orientation,
    remove_goalkeeper_candidate,
)
from .formation_templates import (
    VALID_FORMATIONS,
    load_formations_from_csv,
    parse_formation,
    map_to_closest_valid_formation,
    to_formation_string,
)


class FormationDetector:
    """Detects and smooths tactical formations for each team from top-down player coordinates."""
    def __init__(
        self,
        history_size: int = 50,
        distance_threshold: Optional[float] = None,
        min_players: int = 7,
        ignore_goalkeeper: bool = True,
        field_width: float = 68.0,
        field_length: float = 105.0,
        transition_penalty: float = 0.08,
        transition_stability_frames: int = 72,
        transition_display_frames: int = 90,
        camera_view: str = "long_side",
        valid_formations: Optional[Sequence[str]] = None,
        formations_csv_path: Optional[str] = None,
    ):
        self.history_size = history_size
        self.distance_threshold = distance_threshold
        self.min_players = min_players
        self.ignore_goalkeeper = ignore_goalkeeper
        self.field_width = field_width
        self.field_length = field_length
        self.transition_penalty = transition_penalty
        self.transition_stability_frames = transition_stability_frames
        self.transition_display_frames = transition_display_frames
        self.camera_view = camera_view if camera_view in ("long_side", "short_side") else "long_side"
        # long_side: depth axis is X (0–105 m); short_side: depth axis is Y (0–68 m)
        self.depth_axis_idx = 0 if self.camera_view == "long_side" else 1
        self.depth_field_span = self.field_length if self.depth_axis_idx == 0 else self.field_width
        self._gk_filtered_upstream = False

        if valid_formations is not None:
            self.valid_formations = list(valid_formations)
        else:
            default_csv_path = Path(__file__).resolve().parent.parent / "Formations.csv"
            csv_path = formations_csv_path or str(default_csv_path)
            self.valid_formations = load_formations_from_csv(csv_path, fallback=VALID_FORMATIONS)

        self.formation_history: Dict[int, deque] = {
            1: deque(maxlen=history_size),
            2: deque(maxlen=history_size),
        }
        self.last_output_formations: Dict[int, str] = {1: "Unknown", 2: "Unknown"}
        self.confirmed_formations: Dict[int, str] = {1: "Unknown", 2: "Unknown"}
        self.pending_transitions: Dict[int, Optional[Dict[str, int]]] = {1: None, 2: None}
        self.transition_log: List[Dict[str, object]] = []

        self.latest_team_graph: Dict[int, Dict] = defaultdict(dict)
        self.previous_line_assignments: Dict[int, Dict[int, Tuple[int, float]]] = {1: {}, 2: {}}

        # Side detection: accumulate GK X coords until we have enough samples, then lock
        self._side_gk_accum: Dict[int, List[float]] = {1: [], 2: []}
        self._team_sides: Dict[int, Optional[str]] = {1: None, 2: None}
        self._side_lock_frames: int = 30  # frames of GK data before locking

    def team_side(self, team_id: int) -> str:
        """Return 'LEFT', 'RIGHT', or 'Unknown' for which half a team defends."""
        return self._team_sides.get(team_id) or "Unknown"

    def _update_side_detection(self, gk_positions: Dict[int, Optional[Tuple[float, float]]]) -> None:
        """Accumulate GK depth positions; lock in team sides once we have enough samples."""
        # If both already locked, nothing to do
        if self._team_sides[1] is not None and self._team_sides[2] is not None:
            return

        for team_id in (1, 2):
            if self._team_sides[team_id] is not None:
                continue
            gk_pos = gk_positions.get(team_id)
            if gk_pos is not None:
                self._side_gk_accum[team_id].append(float(gk_pos[self.depth_axis_idx]))

        # Try to lock once both teams have enough samples
        if (len(self._side_gk_accum[1]) >= self._side_lock_frames and
                len(self._side_gk_accum[2]) >= self._side_lock_frames):
            mean1 = float(np.mean(self._side_gk_accum[1]))
            mean2 = float(np.mean(self._side_gk_accum[2]))
            if mean1 < mean2:
                self._team_sides[1] = "LEFT"
                self._team_sides[2] = "RIGHT"
            else:
                self._team_sides[1] = "RIGHT"
                self._team_sides[2] = "LEFT"

    def _extract_team_positions(
        self,
        object_tracks: Dict,
        frame_num: int,
    ) -> Tuple[
        Dict[int, List[Tuple[float, float]]],
        Dict[int, Optional[Tuple[float, float]]],
        Dict[int, List[Tuple[int, Tuple[float, float]]]],
    ]:
        """Extract per-team transformed coordinates, GK positions, and player-id pairs."""
        team_positions: Dict[int, List[Tuple[float, float]]] = {1: [], 2: []}
        gk_positions: Dict[int, Optional[Tuple[float, float]]] = {1: None, 2: None}
        team_players: Dict[int, List[Tuple[int, Tuple[float, float]]]] = {1: [], 2: []}

        players_frame = object_tracks.get("Players", [])
        if frame_num >= len(players_frame):
            return team_positions, gk_positions, team_players

        for track_id, player in players_frame[frame_num].items():
            team_id = player.get("team")
            if team_id not in (1, 2):
                continue

            if player.get("exclude_from_team", False):
                continue

            transformed = player.get("position_transformed")
            if transformed is None:
                continue

            pos = np.array(transformed, dtype=np.float32).reshape(-1)
            if len(pos) < 2 or np.any(np.isnan(pos[:2])):
                continue

            if player.get("is_goalkeeper", False):
                gk_positions[team_id] = (float(pos[0]), float(pos[1]))
                continue

            player_pos = (float(pos[0]), float(pos[1]))
            team_positions[team_id].append(player_pos)
            team_players[team_id].append((int(track_id), player_pos))

        return team_positions, gk_positions, team_players

    def _normalize_points_for_matching(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return points

        normalized = points.copy().astype(np.float32)

        # Canonical matching coordinates: X=lateral axis, Y=depth axis.
        lateral_idx = 0 if self.depth_axis_idx == 1 else 1
        lateral_span = self.field_length if lateral_idx == 0 else self.field_width

        canonical = np.empty_like(normalized)
        canonical[:, 0] = normalized[:, lateral_idx] / max(lateral_span, 1.0)
        canonical[:, 1] = normalized[:, self.depth_axis_idx] / max(self.depth_field_span, 1.0)
        normalized = canonical
        return normalized

    def _assign_line_ids_for_team(
        self,
        object_tracks: Dict,
        frame_num: int,
        team_id: int,
        team_players: List[Tuple[int, Tuple[float, float]]],
        formation: str,
        gk_y: Optional[float] = None,
    ) -> None:
        """Assign L1/L2/... ids to players, respecting the confirmed formation line count."""
        if len(team_players) == 0:
            return

        raw_points = np.array([pos for _, pos in team_players], dtype=np.float32)
        points = normalize_orientation(
            raw_points.tolist(),
            self.depth_field_span,
            gk_y=gk_y,
            camera_view=self.camera_view,
        )

        # If we have a confirmed formation, force exactly that many lines using gap-based
        # splitting — this keeps connections in sync with the formation overlay.
        # Fall back to natural adaptive clustering when formation is unknown.
        formation_counts = parse_formation(formation) if formation not in ("Unknown", "") else []
        n_lines = len(formation_counts)

        if n_lines >= 2:
            clusters = cluster_by_gaps(
                points,
                n_lines=n_lines,
                camera_view=self.camera_view,
            )
        else:
            clusters = cluster_player_lines(
                points,
                distance_threshold=self.distance_threshold,
                min_cluster_size=1,
                camera_view=self.camera_view,
            )

        if not clusters:
            return

        # Sort lines by depth: own-goal side is L1
        ordered_clusters = sorted(clusters, key=lambda c: c["mean_y"])

        # When the formation line count changes, stale assignments from the old
        # formation (e.g., L4 in 4-4-2 persisting into 4-3-3) would create ghost
        # groups. Clear the cache so every player gets a fresh assignment.
        prev_n_lines = getattr(self, "_prev_n_lines", {}).get(team_id, 0)
        if not hasattr(self, "_prev_n_lines"):
            self._prev_n_lines: Dict[int, int] = {}
        if n_lines >= 2 and n_lines != prev_n_lines:
            self.previous_line_assignments[team_id] = {}
        self._prev_n_lines[team_id] = n_lines

        prev_map = self.previous_line_assignments.get(team_id, {})
        new_prev: Dict[int, Tuple[int, float]] = {}

        for line_idx, cluster in enumerate(ordered_clusters, start=1):
            for local_idx in cluster["indices"]:
                track_id, pos = team_players[int(local_idx)]
                depth_value = float(points[int(local_idx), self.depth_axis_idx])

                if n_lines >= 2:
                    # Formation-based (gap) clustering is deterministic — trust it directly.
                    # Temporal smoothing would override the correct formation-derived group
                    # and create stale or cross-group assignments.
                    assigned_line = line_idx
                else:
                    # Natural adaptive clustering can be noisy frame-to-frame; smooth it.
                    assigned_line = line_idx
                    prev = prev_map.get(track_id)
                    if prev is not None:
                        prev_line_id, prev_depth = prev
                        if abs(depth_value - prev_depth) < 8.0:
                            assigned_line = prev_line_id

                object_tracks["Players"][frame_num][track_id]["line_id"] = int(assigned_line)
                new_prev[track_id] = (int(assigned_line), depth_value)

        self.previous_line_assignments[team_id] = new_prev

    def _template_points_for_formation(self, formation: str) -> np.ndarray:
        counts = parse_formation(formation)
        if len(counts) == 0:
            return np.empty((0, 2), dtype=np.float32)

        y_values = np.linspace(0.18, 0.82, num=len(counts), dtype=np.float32)
        template_points = []

        for y_value, count in zip(y_values, counts):
            x_values = np.linspace(0.15, 0.85, num=max(count, 1), dtype=np.float32)
            for x_value in x_values:
                template_points.append((x_value, float(y_value)))

        return np.array(template_points, dtype=np.float32)

    def _symmetric_chamfer_distance(self, a_points: np.ndarray, b_points: np.ndarray) -> float:
        if len(a_points) == 0 or len(b_points) == 0:
            return float("inf")

        distances = np.linalg.norm(a_points[:, None, :] - b_points[None, :, :], axis=2)
        a_to_b = float(np.mean(np.min(distances, axis=1)))
        b_to_a = float(np.mean(np.min(distances, axis=0)))

        return 0.5 * (a_to_b + b_to_a)

    def _score_formation_candidate(
        self,
        norm_points: np.ndarray,
        raw_line_counts: Sequence[int],
        candidate: str,
        previous_formation: str,
    ) -> float:
        template_points = self._template_points_for_formation(candidate)
        chamfer = self._symmetric_chamfer_distance(norm_points, template_points)

        target_counts = parse_formation(candidate)
        line_mismatch = sum(
            abs(a - b)
            for a, b in zip(list(raw_line_counts) + [0] * 6, list(target_counts) + [0] * 6)
        )
        line_mismatch *= 0.05

        # Penalize player count mismatch between detected and template
        detected_total = sum(raw_line_counts)
        template_total = sum(target_counts)
        count_penalty = 0.02 * abs(detected_total - template_total)

        transition = self.transition_penalty if previous_formation not in ("Unknown", candidate) else 0.0

        return chamfer + line_mismatch + count_penalty + transition

    def detect_formation(
        self,
        team_positions: Sequence[Tuple[float, float]],
        previous_formation: str = "Unknown",
        gk_y: Optional[float] = None,
    ) -> Tuple[str, float]:
        """Estimate formation string and confidence from one team snapshot.

        Clusters players ONCE by natural Y-gaps (depth axis proximity), then
        scores every candidate template against those natural line counts.
        """
        if len(team_positions) < self.min_players:
            return "Unknown", 0.0

        points = normalize_orientation(
            team_positions,
            self.depth_field_span,
            gk_y=gk_y,
            camera_view=self.camera_view,
        )

        # Only remove goalkeeper heuristically if not filtered upstream
        if not self._gk_filtered_upstream:
            points = remove_goalkeeper_candidate(
                points,
                enabled=self.ignore_goalkeeper,
                camera_view=self.camera_view,
            )

        if len(points) < self.min_players:
            return "Unknown", 0.0

        # Cluster ONCE by natural Y-gaps — no forced number of lines per candidate
        clusters = cluster_player_lines(
            points,
            distance_threshold=self.distance_threshold,
            min_cluster_size=2,
            camera_view=self.camera_view,
        )
        line_counts = line_counts_from_clusters(clusters)

        if not line_counts:
            return "Unknown", 0.0

        norm_points = self._normalize_points_for_matching(points)

        best_candidate = None
        best_score = float("inf")

        for candidate in self.valid_formations:
            score = self._score_formation_candidate(
                norm_points,
                line_counts,
                candidate,
                previous_formation,
            )

            if score < best_score:
                best_score = score
                best_candidate = candidate

        if best_candidate is not None:
            confidence = float(np.clip(1.0 - (best_score / 0.35), 0.0, 1.0))
            return best_candidate, confidence

        return "Unknown", 0.0

    def _smoothed_formation(self, team_id: int) -> str:
        """Return confidence-weighted temporal mode of recent detections for one team."""
        history = self.formation_history[team_id]

        if not history:
            return "Unknown"

        weighted_scores = defaultdict(float)

        for idx, item in enumerate(history):
            formation = item["formation"]
            confidence = float(item["confidence"])
            player_count = item.get("player_count", 7)

            if formation == "Unknown":
                continue

            recency_weight = 0.65 + 0.35 * ((idx + 1) / len(history))
            count_weight = min(player_count / 10.0, 1.0)
            weighted_scores[formation] += confidence * recency_weight * count_weight

        if not weighted_scores:
            return "Unknown"

        return max(weighted_scores.items(), key=lambda x: x[1])[0]

    def _check_transition(self, team_id: int, formation: str, frame_num: int) -> None:
        if formation == "Unknown":
            return

        confirmed = self.confirmed_formations[team_id]

        if confirmed == "Unknown":
            self.confirmed_formations[team_id] = formation
            return

        if formation == confirmed:
            self.pending_transitions[team_id] = None
            return

        pending = self.pending_transitions[team_id]

        if pending and pending["to_formation"] == formation:
            pending["stable_frames"] += 1
        else:
            self.pending_transitions[team_id] = {
                "from_formation": confirmed,
                "to_formation": formation,
                "start_frame": frame_num,
                "stable_frames": 1,
            }
            return

        if pending["stable_frames"] < self.transition_stability_frames:
            return

        event = {
            "team_id": team_id,
            "from_formation": pending["from_formation"],
            "to_formation": pending["to_formation"],
            "start_frame": pending["start_frame"],
            "confirmed_frame": frame_num,
            "stable_frames": pending["stable_frames"],
        }
        self.transition_log.append(event)
        self.confirmed_formations[team_id] = pending["to_formation"]
        self.pending_transitions[team_id] = None

    def get_transition_status(self, team_id: int, frame_num: int) -> Optional[str]:
        pending = self.pending_transitions[team_id]
        if pending is not None:
            return (
                f"Team {team_id} pending: {pending['from_formation']} -> {pending['to_formation']} "
                f"({pending['stable_frames']}/{self.transition_stability_frames})"
            )

        for event in reversed(self.transition_log):
            if event["team_id"] != team_id:
                continue
            if frame_num - int(event["confirmed_frame"]) > self.transition_display_frames:
                break
            return (
                f"Team {team_id} changed: {event['from_formation']} -> {event['to_formation']}"
            )

        return None

    def export_transitions_csv(self, output_path: str) -> None:
        with open(output_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "team_id",
                    "from_formation",
                    "to_formation",
                    "start_frame",
                    "confirmed_frame",
                    "stable_frames",
                ],
            )
            writer.writeheader()
            writer.writerows(self.transition_log)

    def get_team_structure_graph(
        self,
        player_positions: Sequence[Tuple[float, float]],
    ) -> Dict:
        """Build a line-aware graph structure for UI drawing support."""
        points = normalize_orientation(
            player_positions,
            self.depth_field_span,
            camera_view=self.camera_view,
        )
        points = remove_goalkeeper_candidate(
            points,
            enabled=self.ignore_goalkeeper,
            camera_view=self.camera_view,
        )

        if len(points) == 0:
            return {"nodes": [], "edges": [], "lines": []}

        return build_structure_graph(
            points.tolist(),
            distance_threshold=self.distance_threshold,
            camera_view=self.camera_view,
        )

    def update(self, object_tracks: Dict, frame_num: int) -> Dict[str, str]:
        """Update both teams for the current frame and return smoothed formations."""
        team_positions, gk_positions, team_players = self._extract_team_positions(object_tracks, frame_num)

        # Update side detection using GK positions before processing formations
        self._update_side_detection(gk_positions)

        team_formations = {}

        for team_id in (1, 2):
            positions = team_positions[team_id]
            gk_pos = gk_positions[team_id]
            gk_y = gk_pos[self.depth_axis_idx] if gk_pos is not None else None

            prev = self.last_output_formations.get(team_id, "Unknown")
            formation, confidence = self.detect_formation(positions, previous_formation=prev, gk_y=gk_y)

            self.formation_history[team_id].append(
                {
                    "formation": formation,
                    "confidence": confidence,
                    "player_count": len(positions),
                }
            )

            team_formations[team_id] = self._smoothed_formation(team_id)
            self._check_transition(team_id, team_formations[team_id], frame_num)
            self.last_output_formations[team_id] = team_formations[team_id]
            self.latest_team_graph[team_id] = self.get_team_structure_graph(positions)
            self._assign_line_ids_for_team(
                object_tracks,
                frame_num,
                team_id,
                team_players[team_id],
                team_formations[team_id],
                gk_y=gk_y,
            )

        return {
            "team1_formation": team_formations[1],
            "team2_formation": team_formations[2],
        }

    def draw_overlay(
        self,
        frame,
        formations: Dict[str, str],
        frame_num: Optional[int] = None,
        panel_origin: Tuple[int, int] = (30, 120),
        team_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
    ):
        """Draw compact formation labels onto a video frame."""
        x, y = panel_origin

        overlay = frame.copy()
        panel_height = 190 if frame_num is not None else 110
        cv2.rectangle(overlay, (x, y), (x + 520, y + panel_height), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        if team_colors and 1 in team_colors:
            bar_overlay = frame.copy()
            cv2.rectangle(bar_overlay, (x + 8, y + 12), (x + 14, y + 38), tuple(int(c) for c in team_colors[1]), -1)
            cv2.addWeighted(bar_overlay, 0.9, frame, 0.1, 0, frame)

        cv2.putText(
            frame,
            f"Team 1 Formation: {formations.get('team1_formation', 'Unknown')}",
            (x + 20, y + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        if team_colors and 2 in team_colors:
            bar_overlay = frame.copy()
            cv2.rectangle(bar_overlay, (x + 8, y + 48), (x + 14, y + 74), tuple(int(c) for c in team_colors[2]), -1)
            cv2.addWeighted(bar_overlay, 0.9, frame, 0.1, 0, frame)

        cv2.putText(
            frame,
            f"Team 2 Formation: {formations.get('team2_formation', 'Unknown')}",
            (x + 20, y + 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        # Side detection row: show which half each team defends once locked
        side1 = self._team_sides.get(1)
        side2 = self._team_sides.get(2)
        if side1 is not None and side2 is not None:
            side_label = f"Sides:  Team 1 \u2192 {side1}    Team 2 \u2192 {side2}"
            cv2.putText(
                frame,
                side_label,
                (x + 12, y + 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (30, 30, 150),
                2,
                cv2.LINE_AA,
            )

        if frame_num is not None:
            transition_y = y + 130
            for team_id in (1, 2):
                status = self.get_transition_status(team_id, frame_num)
                if status is None:
                    continue

                color = (0, 0, 255) if "pending" in status else (0, 120, 0)
                cv2.putText(
                    frame,
                    status,
                    (x + 12, transition_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    color,
                    2,
                    cv2.LINE_AA,
                )
                transition_y += 30

        return frame
