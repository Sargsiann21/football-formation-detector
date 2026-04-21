from utils import read_video, save_video, compute_iou
from trackers import Tracker
import cv2
import numpy as np
import os
import pickle
from ultralytics import YOLO
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement import CameraMovement
from view_transformer import ViewTransformer
from speed_distance import SpeedDistance
from team_structure import TeamStructureDrawer
from formation_detector import FormationDetector


CAMERA_VIEW = "long_side"  # "long_side" or "short_side"


def get_team_detections(model_path, frames, read_from_stub=False, stub_path=None):
    """Run team_assigner model and return per-frame list of {bbox, team/role} dicts."""
    if read_from_stub and stub_path and os.path.exists(stub_path):
        with open(stub_path, "rb") as f:
            return pickle.load(f)

    model = YOLO(model_path)
    team_detections = []
    batch_size = 20

    for i in range(0, len(frames), batch_size):
        batch_results = model.predict(frames[i:i + batch_size], conf=0.3, verbose=False)
        for result in batch_results:
            frame_dets = []
            for box in result.boxes:
                cls_id = int(box.cls[0])
                name = result.names[cls_id]
                if name in ("TEAM 1", "TEAM 2"):
                    bbox = box.xyxy[0].tolist()
                    team_id = 1 if name == "TEAM 1" else 2
                    frame_dets.append({"bbox": bbox, "team": team_id, "role": "player"})
                elif name == "GoalKeeper":
                    bbox = box.xyxy[0].tolist()
                    frame_dets.append({"bbox": bbox, "team": None, "role": "goalkeeper"})
                elif name in ("Referee", "Corner"):
                    bbox = box.xyxy[0].tolist()
                    frame_dets.append({"bbox": bbox, "team": None, "role": name.lower()})
            team_detections.append(frame_dets)

    if stub_path:
        with open(stub_path, "wb") as f:
            pickle.dump(team_detections, f)

    return team_detections


def main():

    # -------------------------
    # Read video
    # -------------------------
    video_frames = read_video('input_videos/input_video.mp4')

    # -------------------------
    # Initialize tracker
    # -------------------------
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/tracks_stub.pkl'
    )

    # Ensure all track lists are aligned with the video length when loading older/partial stubs.
    total_frames = len(video_frames)
    for object_name in ('Players', 'Referees', 'Ball'):
        object_tracks = tracks.get(object_name, [])

        if len(object_tracks) < total_frames:
            object_tracks = object_tracks + [{} for _ in range(total_frames - len(object_tracks))]
        elif len(object_tracks) > total_frames:
            object_tracks = object_tracks[:total_frames]

        tracks[object_name] = object_tracks

    # REQUIRED: adds object positions for later modules
    tracker.add_position_tracks(tracks)

    # -------------------------
    # Camera movement
    # -------------------------
    camera_movement_estimator = CameraMovement(video_frames[0])

    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/camera_movement.pkl'
    )

    camera_movement_estimator.adjust_positions_tracks(
        tracks,
        camera_movement_per_frame
    )

    # -------------------------
    # View Transformer (FIELD)
    # -------------------------
    view_transformer = ViewTransformer(video_frames[0])

    view_transformer.add_transformed_position_tracks(
        tracks,
        read_from_stub=True,
        stub_path="stubs/field_tracks.pkl"
    )

    # -------------------------
    # Interpolate ball
    # -------------------------
    tracks["Ball"] = tracker.interpolate_ball(tracks["Ball"])

    # -------------------------
    # Speed & Distance
    # -------------------------
    speeddistance_estimator = SpeedDistance()
    speeddistance_estimator.add_speeddistance_tracks(tracks)

    # -------------------------
    # Assign teams (model-based + KMeans fallback)
    # -------------------------
    team_assigner = TeamAssigner()

    # Fallback colors so downstream drawing works even if clustering cannot initialize.
    team_assigner.team_colors[1] = (0, 0, 255)
    team_assigner.team_colors[2] = (255, 0, 0)

    # Run team_assigner model for direct TEAM 1 / TEAM 2 detections
    team_model_path = 'models/team_assigner.pt'
    team_detections = None

    if os.path.exists(team_model_path):
        print("Running team classification model...")
        team_detections = get_team_detections(
            team_model_path,
            video_frames,
            read_from_stub=True,
            stub_path='stubs/team_class_stub.pkl',
        )
        # Align length with video
        if len(team_detections) < total_frames:
            team_detections += [[] for _ in range(total_frames - len(team_detections))]
        elif len(team_detections) > total_frames:
            team_detections = team_detections[:total_frames]
        print(f"Team classification model loaded ({sum(len(d) for d in team_detections)} detections across {total_frames} frames)")
    else:
        print("No team_assigner model found, using KMeans fallback only.")

    # Initialize KMeans fallback for players the model misses
    bootstrap_frame_num = None
    bootstrap_players = None

    for frame_num, player_track in enumerate(tracks['Players']):
        if len(player_track) < 2:
            continue

        frame_h, frame_w = video_frames[frame_num].shape[:2]
        valid_players = {}

        for player_id, track in player_track.items():
            bbox = track.get('bbox', None)
            if bbox is None or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox
            x1 = int(max(0, min(frame_w - 1, x1)))
            y1 = int(max(0, min(frame_h - 1, y1)))
            x2 = int(max(0, min(frame_w, x2)))
            y2 = int(max(0, min(frame_h, y2)))

            if x2 - x1 < 2 or y2 - y1 < 2:
                continue

            valid_players[player_id] = {'bbox': [x1, y1, x2, y2]}

        if len(valid_players) >= 2:
            bootstrap_frame_num = frame_num
            bootstrap_players = valid_players
            break

    if bootstrap_frame_num is not None:
        team_assigner.assign_team_color(
            video_frames[bootstrap_frame_num],
            bootstrap_players
        )
    else:
        print("Warning: Could not initialize team colors (no frame with >=2 valid players). Using fallback colors.")

    # Assign teams: prefer model, fall back to KMeans
    model_assigned = 0
    kmeans_assigned = 0

    for frame_num, player_track in enumerate(tracks['Players']):
        frame_team_dets = team_detections[frame_num] if team_detections else []

        for player_id, track in player_track.items():
            player_bbox = track['bbox']

            # Try matching to team_assigner model detection by IoU
            best_iou = 0.0
            best_team = None
            best_role = "player"
            for det in frame_team_dets:
                iou = compute_iou(player_bbox, det['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_team = det.get('team')
                    best_role = det.get('role', 'player')

            if best_role == "goalkeeper" and best_iou > 0.3:
                tracks['Players'][frame_num][player_id]['is_goalkeeper'] = True

            # Skip referee/corner misdetections from team assignment
            if best_role in ("referee", "corner") and best_iou > 0.3:
                tracks['Players'][frame_num][player_id]['exclude_from_team'] = True
                continue

            if best_team is not None and best_iou > 0.3:
                team = best_team
                model_assigned += 1
            elif hasattr(team_assigner, 'kmeans'):
                team = team_assigner.get_player_team(
                    video_frames[frame_num],
                    track['bbox'],
                    player_id
                )
                kmeans_assigned += 1
            else:
                team = 1
                kmeans_assigned += 1

            tracks['Players'][frame_num][player_id]['team'] = team
            tracks['Players'][frame_num][player_id]['team_color'] = team_assigner.team_colors.get(team, (0, 0, 255))

    print(f"Team assignment: {model_assigned} from model, {kmeans_assigned} from KMeans fallback")

    # -------------------------
    # Ball possession
    # -------------------------
    player_assigner = PlayerBallAssigner()

    team_ball_control = []
    last_ball_bbox = None

    for frame_num, player_track in enumerate(tracks['Players']):

        ball_dict = tracks['Ball'][frame_num]

        if 1 in ball_dict:
            last_ball_bbox = ball_dict[1]['bbox']

        if last_ball_bbox is None:
            continue

        assigned_player = player_assigner.assign_to_player(
            player_track,
            last_ball_bbox
        )

        if assigned_player != -1:

            assigned_track = tracks['Players'][frame_num][assigned_player]

            if 'team' not in assigned_track:
                if len(team_ball_control) > 0:
                    team_ball_control.append(team_ball_control[-1])
                continue

            assigned_track['has_ball'] = True

            team_ball_control.append(assigned_track['team'])

        else:

            if len(team_ball_control) > 0:
                team_ball_control.append(team_ball_control[-1])

    team_ball_control = np.array(team_ball_control)

    # -------------------------
    # Draw annotations
    # -------------------------
    output_video_frames = tracker.draw_annotations(
        video_frames,
        tracks,
        team_ball_control
    )

    # -------------------------
    # Draw camera movement
    # -------------------------
    output_video_frames = camera_movement_estimator.draw_camera_movement(
        output_video_frames,
        camera_movement_per_frame
    )

    # -------------------------
    # Draw speed & distance
    # -------------------------
    speeddistance_estimator.draw_speeddistance(
        output_video_frames,
        tracks
    )

    # -------------------------
    # Draw team structure & formations
    # -------------------------
    team_structure_drawer = TeamStructureDrawer(k_neighbors=3)
    formation_detector = FormationDetector(
        history_size=50,
        distance_threshold=None,
        min_players=6,
        camera_view=CAMERA_VIEW,
        formations_csv_path='Formations.csv'
    )
    # If team model already filters goalkeepers, skip heuristic GK removal
    if team_detections is not None:
        formation_detector._gk_filtered_upstream = True

    for frame_num in range(len(output_video_frames)):

        output_video_frames[frame_num] = team_structure_drawer.draw_team_structure(
            output_video_frames[frame_num],
            tracks['Players'],
            frame_num
        )

        current_formations = formation_detector.update(tracks, frame_num)

        output_video_frames[frame_num] = formation_detector.draw_overlay(
            output_video_frames[frame_num],
            current_formations,
            frame_num=frame_num,
            team_colors=team_assigner.team_colors,
        )

    formation_detector.export_transitions_csv(
        'output_videos/formation_transitions.csv'
    )

    # -------------------------
    # Save video
    # -------------------------
    save_video(
        output_video_frames,
        'output_videos/output_video.avi'
    )


if __name__ == '__main__':
    main()