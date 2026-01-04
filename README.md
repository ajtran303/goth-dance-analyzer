# Goth Dance Analyzer

Analyze and compare dance styles using MediaPipe pose estimation and FFT rhythm detection.

## What It Does

- Records dancers performing a setlist
- Extracts skeleton data from video
- Calculates movement metrics (velocity, range, symmetry, etc.)
- Detects rhythm patterns using frequency domain analysis
- Generates summary scores and dancer fingerprints
- Compares dancers across songs

## Setlist

Default setlist (goth classics, slow to fast):

| Song                    | Artist           | BPM  |
| ----------------------- | ---------------- | ---- |
| Bela Lugosi's Dead      | Bauhaus          | ~75  |
| Lucretia My Reflection  | Sisters of Mercy | ~115 |
| Cities in Dust          | Siouxsie         | ~120 |
| Gallowdance             | Lebanon Hanover  | ~138 |
| Love Will Tear Us Apart | Joy Division     | ~147 |

Use any setlist you want — just add your music files to `setlist/` and update the song list in `scripts/capture.py`.

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
python scripts/analyze.py --dir skeleton_data

# Analyze single file
python scripts/analyze.py skeleton_data/alice/alice_gallowdance_skeleton.json
```

Outputs:

- `analysis/metrics.json` — raw and normalized metrics
- `analysis/charts/summary.pdf` — fingerprints + per-song summary scores
- `analysis/charts/detailed_metrics.pdf` — per-song metric breakdown
- `analysis/charts/rhythm_spectrums.pdf` — FFT frequency analysis

### 5. Export Video with Skeleton Overlay

Render skeleton and live metrics onto original footage.

```bash
# Single video
python scripts/export_video.py recordings/alice/alice_gallowdance.mp4

# All videos
python scripts/export_video.py --batch recordings

# Without metrics panel
python scripts/export_video.py --no-metrics recordings/alice/alice_gallowdance.mp4
```

Saves to: `exports/<video_name>_skeleton.mp4`

## Metrics

### Raw Metrics

| Metric             | Description                         |
| ------------------ | ----------------------------------- |
| Arm Velocity       | How fast arms move                  |
| Movement Range     | How big/expansive gestures are      |
| Vertical Motion    | Amount of jumping/bouncing          |
| Symmetry           | Left/right mirror movement          |
| Stillness Ratio    | How often dancer pauses             |
| Upper Body Focus   | Arms vs legs emphasis               |
| Movement BPM       | Dominant rhythm frequency (via FFT) |
| Rhythm Strength    | How pronounced the rhythm is        |
| Rhythm Consistency | How stable the rhythm is over time  |

### Summary Scores

| Score   | Formula                                               |
| ------- | ----------------------------------------------------- |
| Energy  | (Arm Velocity + Movement Range + Vertical Motion) / 3 |
| Control | (Symmetry + Rhythm Consistency) / 2                   |
| Groove  | (Rhythm Strength + Rhythm Consistency) / 2            |
| Flow    | 1 - Stillness Ratio                                   |

### Dancer Archetypes

Based on highest summary score:

- **Energetic** — high movement intensity
- **Precise** — controlled, symmetric movement
- **Groovy** — strong rhythmic patterns
- **Fluid** — continuous, flowing motion

## Workflow

1. `python scripts/test_skeleton.py` — verify setup
2. `python scripts/capture.py` — record each dancer
3. `python scripts/extract_skeleton.py --batch recordings` — extract all
4. `python scripts/analyze.py --dir skeleton_data` — compare everyone
5. `python scripts/export_video.py --batch recordings` — export shareable videos

## License

MIT
