import cv2
import numpy as np


class TeamStructureDrawer:

    def __init__(self, k_neighbors=3):
        self.k = k_neighbors

    def get_foot_position(self, bbox):
        x1, y1, x2, y2 = bbox
        x = int((x1 + x2) / 2)
        y = int(y2)
        return (x, y)

    def draw_team_structure(self, frame, player_tracks, frame_num):

        team1_positions = []
        team2_positions = []
        team1_color = (0, 0, 255)
        team2_color = (255, 0, 0)
        team1_line_ids = []
        team2_line_ids = []

        for player_id, player in player_tracks[frame_num].items():

            bbox = player["bbox"]
            team = player.get("team", None)

            if team is None:
                continue

            if player.get("is_goalkeeper", False):
                continue

            if player.get("exclude_from_team", False):
                continue

            position = self.get_foot_position(bbox)
            line_id = player.get("line_id", 0)

            if team == 1:
                team1_positions.append(position)
                team1_color = player.get("team_color", team1_color)
                team1_line_ids.append(line_id)

            elif team == 2:
                team2_positions.append(position)
                team2_color = player.get("team_color", team2_color)
                team2_line_ids.append(line_id)

        frame = self._connect_by_lines(frame, team1_positions, team1_line_ids, team1_color)
        frame = self._connect_by_lines(frame, team2_positions, team2_line_ids, team2_color)

        return frame

    def _connect_by_lines(self, frame, positions, line_ids, color):
        """Connect players within each formation line group using the team color."""
        if len(positions) < 2:
            return frame

        positions_np = np.array(positions, dtype=np.float32)
        line_ids_np = np.array(line_ids, dtype=int)
        unique_lines = sorted(set(line_ids_np.tolist()))

        # Group indices by line_id
        line_groups = {}
        for lid in unique_lines:
            idxs = np.where(line_ids_np == lid)[0]
            if len(idxs) > 0:
                line_groups[lid] = idxs

        # Draw connection lines within each group using the team color.
        # Sort by the axis with the most spread within the group (adaptive to
        # camera perspective: defenders spread vertically → sort by pixel Y).
        # Skip any connection where the gap is an outlier (> 2x median gap),
        # which prevents one out-of-position player from creating a long bridge.
        line_overlay = frame.copy()
        for lid, idxs in line_groups.items():
            pts = positions_np[idxs]
            if len(pts) < 2:
                continue

            # Choose sort axis: whichever has higher std (spread axis)
            sort_axis = 1 if np.std(pts[:, 1]) > np.std(pts[:, 0]) else 0
            sorted_order = np.argsort(pts[:, sort_axis])
            sorted_pts = pts[sorted_order]

            # Compute gaps between adjacent players in sorted order
            gaps = np.linalg.norm(np.diff(sorted_pts, axis=0), axis=1)
            if len(gaps) == 0:
                continue
            median_gap = float(np.median(gaps))
            max_allowed = max(median_gap * 2.5, 80.0)  # never reject tiny groups

            for i in range(len(sorted_order) - 1):
                if gaps[i] > max_allowed:
                    continue  # skip outlier-gap connections
                p1 = tuple(sorted_pts[i].astype(int))
                p2 = tuple(sorted_pts[i + 1].astype(int))
                cv2.line(line_overlay, p1, p2, color, 3, cv2.LINE_AA)
        cv2.addWeighted(line_overlay, 0.75, frame, 0.25, 0, frame)

        return frame

    def _connect_players(self, frame, positions, color):
        """Legacy k-NN connector — kept for backward compatibility."""
        if len(positions) < 2:
            return frame

        positions = np.array(positions)
        overlay = frame.copy()

        for i in range(len(positions)):
            distances = np.linalg.norm(positions - positions[i], axis=1)
            nearest_indices = np.argsort(distances)[1:self.k + 1]

            for j in nearest_indices:
                p1 = tuple(positions[i].astype(int))
                p2 = tuple(positions[j].astype(int))
                cv2.line(overlay, p1, p2, color, 2, cv2.LINE_AA)

            cv2.addWeighted(overlay, self.field_line_alpha, frame, 1 - self.field_line_alpha, 0, frame)

        return frame

    def draw_structure_panel(self, frames, tracks):

        for frame_num, frame in enumerate(frames):

            team1_positions = []
            team2_positions = []

            for player_id, player in tracks["Players"][frame_num].items():

                bbox = player["bbox"]
                team = player.get("team", None)

                position = self.get_foot_position(bbox)

                if team == 1:
                    team1_positions.append(position)

                elif team == 2:
                    team2_positions.append(position)

            h, w, _ = frame.shape

            panel_x1 = w - 300
            panel_x2 = w - 20
            panel_y1 = 120
            panel_y2 = 300

            cv2.rectangle(frame, (panel_x1, panel_y1), (panel_x2, panel_y2), (255, 255, 255), -1)

            cv2.putText(frame, "Team Structures", (panel_x1 + 20, panel_y1 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            self._draw_mini_structure(frame, team1_positions,
                                      panel_x1 + 20, panel_y1 + 40, (0, 0, 255))

            self._draw_mini_structure(frame, team2_positions,
                                      panel_x1 + 20, panel_y1 + 120, (255, 0, 0))

    def _draw_mini_structure(self, frame, positions, x_offset, y_offset, color):

        if len(positions) == 0:
            return

        positions = np.array(positions)

        min_x = np.min(positions[:, 0])
        max_x = np.max(positions[:, 0])

        min_y = np.min(positions[:, 1])
        max_y = np.max(positions[:, 1])

        width = max_x - min_x + 1
        height = max_y - min_y + 1

        mini_points = []

        for p in positions:
            x = int((p[0] - min_x) / width * 200) + x_offset
            y = int((p[1] - min_y) / height * 60) + y_offset
            mini_points.append((x, y))

        if len(mini_points) >= 2:
            mini_points_np = np.array(mini_points, dtype=np.float32)
            max_neighbors = min(self.k, len(mini_points) - 1)
            drawn_edges = set()

            for i in range(len(mini_points_np)):
                distances = np.linalg.norm(mini_points_np - mini_points_np[i], axis=1)
                nearest_indices = np.argsort(distances)[1:max_neighbors + 1]

                for j in nearest_indices:
                    edge = tuple(sorted((i, int(j))))
                    if edge in drawn_edges:
                        continue
                    drawn_edges.add(edge)
                    p1 = tuple(mini_points_np[i].astype(int))
                    p2 = tuple(mini_points_np[j].astype(int))
                    cv2.line(frame, p1, p2, color, 1, cv2.LINE_AA)

        for x, y in mini_points:
            cv2.circle(frame, (x, y), 4, color, -1)
