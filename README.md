# Formation Inference and Transition Detection from Broadcast Football Video

A computer vision pipeline for tactical football formation analysis from broadcast video. The system detects players, assigns team identities, maps positions to a standardized pitch coordinate system, and infers team formations with transition detection — operating entirely on publicly available broadcast footage without any proprietary tracking data.

---

## Results

- Player detection: **96.47%**
- Referee detection: **91.89%**
- Ball detection: **86.08%**
- Team assignment: **79.25%**
- Ball possession attribution: **~77%**
- Formation transition precision: **67.64%**

---

## Project Structure

```
football-formation-tracker/
│
├── camera_movement/          # Camera motion estimation module
├── dev_analysis/             # Development and analysis scripts
├── formation_detector/       # Formation inference engine
├── input_videos/             # Place your input video here
├── models/                   # Model weights (see download instructions below)
├── output_videos/            # Output video and CSV will appear here
├── player_ball_assigner/     # Ball possession attribution module
├── player_role/              # Player role assignment
├── speed_distance/           # Speed and distance computation
├── stubs/                    # Cached intermediate outputs (must be empty on first run)
├── team_assigner/            # Team identity assignment module
├── team_structure/           # Team structure visualization
├── tools/                    # Utility tools
├── trackers/                 # ByteTrack multi-object tracking
├── utils/                    # Helper utilities
├── view_transformer/         # Homography and pitch coordinate mapping
├── visualization/            # Annotation and rendering
├── main.py                   # Main pipeline runner
├── yolo_inference.py         # YOLO inference script
├── Formations.csv            # Formation template definitions
└── README.md
```

---

## Model Weights

The model weights are too large to host on GitHub. Download them from the link below and place them inside the `models/` folder.

**Download:** [Google Drive — Model Weights](https://drive.google.com/drive/folders/1kOyVTeZPdBXY-C8eGR0gHLsSdJ-FtYNE?usp=drive_link)

After downloading, your `models/` folder should contain:

```
models/
├── best.pt           # Primary object detector (YOLOv8)
├── field_best.pt     # Field region detector
├── last.pt           # Alternative detector checkpoint
└── team_assigner.pt  # Team classification model
```

Also place `yolov8l.pt` in the root of the project directory:

```
football-formation-tracker/
└── yolov8l.pt
```

---

## Datasets

### Video Data
The pipeline was trained and tested using video data from the following Kaggle dataset:

**DFL Bundesliga — 460 MP4 Videos in 30sec**
[https://www.kaggle.com/datasets/saberghaderi/-dfl-bundesliga-460-mp4-videos-in-30sec-csv/data](https://www.kaggle.com/datasets/saberghaderi/-dfl-bundesliga-460-mp4-videos-in-30sec-csv/data)

### Formation Templates
The `Formations.csv` file used for formation template matching was sourced from:

**FIFA Formations — Football Formations**
[https://www.kaggle.com/datasets/farzammanafzadeh/fifa-formations-football-formations](https://www.kaggle.com/datasets/farzammanafzadeh/fifa-formations-football-formations)

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Sargsiann21/football-formation-tracker.git
cd football-formation-tracker
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

### 1. Prepare your input video

- Your input video must be in **MP4 format**
- Rename it to **`input_video.mp4`**
- Place it inside the **`input_videos/`** folder

```
input_videos/
└── input_video.mp4
```

### 2. Clear the stubs folder

On your **first run**, make sure the `stubs/` folder is **empty**. The pipeline caches intermediate outputs here to speed up reruns. If stubs from a previous run are present, the pipeline will load those instead of reprocessing your new video.

```
stubs/       ← must be empty before first run
```

### 3. Run the pipeline

```bash
python main.py
```

### 4. Collect your outputs

After processing, two output files will appear in the `output_videos/` folder:

- **Annotated video** — broadcast footage with player IDs, team colors, possession indicators, speed overlays, team structure links, and formation labels
- **Transition CSV** — a structured log of all confirmed formation changes, including team, previous formation, new formation, detection frame, confirmation frame, and stability duration

```
output_videos/
├── output_video.mp4
└── formation_transitions.csv
```

---

## Notes

- The pipeline runs in **offline mode** — the full video is processed before any output is produced
- Processing time depends on video length and hardware
- For best results, use broadcast footage with a clear view of the pitch
- The pipeline is designed for single-camera broadcast video; replays and close-up shots may affect formation inference quality

---

## Acknowledgments

Supervisor: **Arman Asryan, PhD**
American University of Armenia — College of Science and Engineering

---

## License

This project is released for academic and research purposes.
