# goth-dance-analyzer

Analyze and compare dance styles using MediaPipe pose estimation.

## What It Does

- Records dancers performing a setlist
- Extracts skeleton data from video
- Calculates movement metrics (arm velocity, range, symmetry, etc.)
- Compares dancers to each other

## Requirements

- Python 3.14+
- Webcam

## Install

```bash
pip install -r requirements.txt
```

## Scripts

### 1. Test Skeleton Tracking

Verify your camera and lighting work before recording.

```bash
python scripts/test_skeleton.py
```

- Shows live skeleton overlay
- Displays detection rate
- Press `Q` to quit

Aim for >90% detection rate. If lower, improve lighting or adjust camera angle.

### 2. Record Dance Sessions

Record dancers performing the setlist.

```bash
python scripts/capture.py
```

- Enter dancer name
- Select song from list
- Press `SPACE` to start/stop recording
- Press `S` to save and exit
- Press `Q` to quit without saving

Saves to: `recordings/<dancer_name>/<dancer_name>_<song>.mp4`

### 3. Extract Skeleton Data

Extract pose landmarks from recorded videos.

```bash
# Single video
python scripts/extract_skeleton.py recordings/alice/alice_gallowdance.mp4

# All videos in recordings folder
python scripts/extract_skeleton.py --batch recordings
```

Saves to: `skeleton_data/<dancer_name>/<video>_skeleton.json`

### 4. Analyze and Compare

Calculate metrics and generate comparison charts.

```bash
# Analyze all skeleton files
python scripts/analyze.py

# Analyze single file
python scripts/analyze.py skeleton_data/alice/alice_gallowdance_skeleton.json

# Analyze specific directory
python scripts/analyze.py --dir skeleton_data
```

Outputs:

- `analysis/metrics.json` — raw and normalized metrics
- `analysis/charts/comparison_radar.png` — radar chart
- `analysis/charts/comparison_bars.png` — bar chart

## Metrics

| Metric           | Description                    |
| ---------------- | ------------------------------ |
| Arm Velocity     | How fast arms move             |
| Movement Range   | How big/expansive gestures are |
| Vertical Motion  | Amount of jumping/bouncing     |
| Symmetry         | Left/right mirror movement     |
| Stillness Ratio  | How often dancer pauses        |
| Upper Body Focus | Arms vs legs emphasis          |

## Workflow

1. `python scripts/test_skeleton.py` — verify setup
2. `python scripts/capture.py` — record each dancer
3. `python scripts/extract_skeleton.py --batch recordings` — extract all
4. `python scripts/analyze.py` — compare everyone

## License

MIT
