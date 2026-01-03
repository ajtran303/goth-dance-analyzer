# pyright: reportAttributeAccessIssue=false

"""
Export video with skeleton overlay.
Renders pose landmarks onto original footage for shareable clips.
"""

import cv2
import json
import numpy as np
import os
import sys

# Pose connections for drawing skeleton
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),  # Left eye
    (0, 4), (4, 5), (5, 6), (6, 8),  # Right eye
    (9, 10),  # Mouth
    (11, 12),  # Shoulders
    (11, 13), (13, 15),  # Left arm
    (12, 14), (14, 16),  # Right arm
    (11, 23), (12, 24),  # Torso
    (23, 24),  # Hips
    (23, 25), (25, 27),  # Left leg
    (24, 26), (26, 28),  # Right leg
    (27, 29), (29, 31),  # Left foot
    (28, 30), (30, 32),  # Right foot
    (15, 17), (15, 19), (15, 21),  # Left hand
    (16, 18), (16, 20), (16, 22),  # Right hand
]

# Skeleton colors (BGR)
LINE_COLOR = (0, 255, 0)  # Green
JOINT_COLOR = (0, 0, 255)  # Red
TEXT_COLOR = (255, 255, 255)  # White
TEXT_BG_COLOR = (0, 0, 0)  # Black


def find_skeleton_file(video_path):
    """Find matching skeleton JSON for a video."""
    video_name = os.path.basename(video_path).rsplit('.', 1)[0]
    
    # Check skeleton_data folder with same structure
    rel_dir = os.path.dirname(video_path)
    if rel_dir.startswith("recordings"):
        rel_dir = rel_dir.replace("recordings", "", 1).lstrip(os.sep)
    
    skeleton_path = os.path.join("skeleton_data", rel_dir, f"{video_name}_skeleton.json")
    
    if os.path.exists(skeleton_path):
        return skeleton_path
    
    # Check same folder as video
    skeleton_path = video_path.rsplit('.', 1)[0] + '_skeleton.json'
    if os.path.exists(skeleton_path):
        return skeleton_path
    
    return None


def load_skeleton(path):
    """Load skeleton data from JSON file."""
    with open(path) as f:
        return json.load(f)


def draw_skeleton(frame, landmarks, line_thickness=2, joint_radius=5):
    """Draw skeleton on a frame."""
    if landmarks is None:
        return frame
    
    height, width = frame.shape[:2]
    
    # Draw connections
    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx >= len(landmarks) or end_idx >= len(landmarks):
            continue
            
        start = landmarks[start_idx]
        end = landmarks[end_idx]
        
        # Skip if low visibility
        if start.get('visibility', 1) < 0.5 or end.get('visibility', 1) < 0.5:
            continue
        
        start_point = (int(start['x'] * width), int(start['y'] * height))
        end_point = (int(end['x'] * width), int(end['y'] * height))
        
        cv2.line(frame, start_point, end_point, LINE_COLOR, line_thickness)
    
    # Draw joints
    for lm in landmarks:
        if lm.get('visibility', 1) < 0.5:
            continue
        x = int(lm['x'] * width)
        y = int(lm['y'] * height)
        cv2.circle(frame, (x, y), joint_radius, JOINT_COLOR, -1)
    
    return frame


def draw_text_with_bg(frame, text, position, font_scale=0.7, thickness=2):
    """Draw text with background for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Draw background rectangle
    x, y = position
    padding = 5
    cv2.rectangle(frame, 
                  (x - padding, y - text_height - padding),
                  (x + text_width + padding, y + baseline + padding),
                  TEXT_BG_COLOR, -1)
    
    # Draw text
    cv2.putText(frame, text, position, font, font_scale, TEXT_COLOR, thickness)
    
    return frame


def calculate_rolling_metrics(frames_data, current_idx, window_size=60):
    """
    Calculate metrics over a rolling window of frames.
    
    Args:
        frames_data: List of all frame data
        current_idx: Current frame index
        window_size: Number of frames to include (e.g., 60 = 2 sec at 30fps)
    
    Returns:
        Dict of metrics or None if insufficient data
    """
    
    # Get window of frames
    start_idx = max(0, current_idx - window_size)
    window = frames_data[start_idx:current_idx + 1]
    
    # Filter frames with landmarks
    valid_frames = [f for f in window if f.get('landmarks') is not None]
    
    if len(valid_frames) < 10:
        return None
    
    # Landmark indices
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    
    # Extract positions
    left_wrist_pos = []
    right_wrist_pos = []
    left_hip_pos = []
    right_hip_pos = []
    left_ankle_pos = []
    right_ankle_pos = []
    
    for f in valid_frames:
        lm = f['landmarks']
        left_wrist_pos.append([lm[LEFT_WRIST]['x'], lm[LEFT_WRIST]['y']])
        right_wrist_pos.append([lm[RIGHT_WRIST]['x'], lm[RIGHT_WRIST]['y']])
        left_hip_pos.append([lm[LEFT_HIP]['x'], lm[LEFT_HIP]['y']])
        right_hip_pos.append([lm[RIGHT_HIP]['x'], lm[RIGHT_HIP]['y']])
        left_ankle_pos.append([lm[LEFT_ANKLE]['x'], lm[LEFT_ANKLE]['y']])
        right_ankle_pos.append([lm[RIGHT_ANKLE]['x'], lm[RIGHT_ANKLE]['y']])
    
    left_wrist_arr = np.array(left_wrist_pos)
    right_wrist_arr = np.array(right_wrist_pos)
    left_hip_arr = np.array(left_hip_pos)
    right_hip_arr = np.array(right_hip_pos)
    left_ankle_arr = np.array(left_ankle_pos)
    right_ankle_arr = np.array(right_ankle_pos)
    
    # 1. Arm velocity
    left_vel = np.mean(np.linalg.norm(np.diff(left_wrist_arr, axis=0), axis=1))
    right_vel = np.mean(np.linalg.norm(np.diff(right_wrist_arr, axis=0), axis=1))
    arm_velocity = (left_vel + right_vel) / 2
    
    # 2. Movement range
    movement_range = (np.std(left_wrist_arr) + np.std(right_wrist_arr)) / 2
    
    # 3. Vertical motion
    hip_y = (left_hip_arr[:, 1] + right_hip_arr[:, 1]) / 2
    vertical_motion = np.std(hip_y)
    
    # 4. Symmetry
    left_movement = np.linalg.norm(np.diff(left_wrist_arr, axis=0), axis=1)
    right_movement = np.linalg.norm(np.diff(right_wrist_arr, axis=0), axis=1)
    if np.std(left_movement) > 0 and np.std(right_movement) > 0:
        symmetry = np.corrcoef(left_movement, right_movement)[0, 1]
        symmetry = max(0, symmetry)
    else:
        symmetry = 0
    
    # 5. Stillness ratio
    total_movement = left_movement + right_movement
    threshold = np.percentile(total_movement, 20) if len(total_movement) > 0 else 0
    stillness_ratio = np.mean(total_movement < threshold) if len(total_movement) > 0 else 0
    
    # 6. Upper body focus
    left_ankle_vel = np.mean(np.linalg.norm(np.diff(left_ankle_arr, axis=0), axis=1))
    right_ankle_vel = np.mean(np.linalg.norm(np.diff(right_ankle_arr, axis=0), axis=1))
    leg_velocity = (left_ankle_vel + right_ankle_vel) / 2
    if leg_velocity > 0:
        upper_body_focus = arm_velocity / (arm_velocity + leg_velocity)
    else:
        upper_body_focus = 1.0
    
    # Calculate rhythm metrics using FFT on velocity signal
    combined_vel = (left_movement + right_movement) / 2
    rhythm_strength = 0.0
    movement_bpm = 0.0

    if len(combined_vel) >= 30:  # Need enough samples for FFT
        fps = 30.0  # Assume 30fps for rolling calculation
        fft_result = np.fft.rfft(combined_vel)
        frequencies = np.fft.rfftfreq(len(combined_vel), 1/fps)
        magnitudes = np.abs(fft_result)

        # Focus on dance-relevant frequency range: 0.5-4 Hz (30-240 BPM)
        min_freq, max_freq = 0.5, 4.0
        dance_band_mask = (frequencies >= min_freq) & (frequencies <= max_freq)

        if np.any(dance_band_mask):
            dance_freqs = frequencies[dance_band_mask]
            dance_mags = magnitudes[dance_band_mask]

            peak_idx = np.argmax(dance_mags)
            dominant_freq = dance_freqs[peak_idx]
            peak_magnitude = dance_mags[peak_idx]

            movement_bpm = dominant_freq * 60

            mean_magnitude = np.mean(dance_mags)
            if mean_magnitude > 0:
                rhythm_strength = min((peak_magnitude / mean_magnitude - 1) / 5, 1.0)
                rhythm_strength = max(0, rhythm_strength)

    # Scale metrics to roughly 0-1 range for display
    return {
        'arm_velocity': min(arm_velocity * 50, 1.0),
        'movement_range': min(movement_range * 10, 1.0),
        'vertical_motion': min(vertical_motion * 50, 1.0),
        'symmetry': symmetry,
        'stillness_ratio': stillness_ratio,
        'upper_body_focus': upper_body_focus,
        'rhythm_strength': rhythm_strength,
        'movement_bpm': movement_bpm
    }


def draw_metrics_panel(frame, metrics, x=10, y_start=None):
    """Draw all metrics as a panel with bars."""
    if metrics is None:
        return frame

    height, width = frame.shape[:2]

    if y_start is None:
        y_start = height - 200

    labels = [
        ('Arm Velocity', metrics['arm_velocity']),
        ('Movement Range', metrics['movement_range']),
        ('Vertical Motion', metrics['vertical_motion']),
        ('Symmetry', metrics['symmetry']),
        ('Stillness', metrics['stillness_ratio']),
        ('Upper Body', metrics['upper_body_focus']),
        ('Rhythm Strength', metrics.get('rhythm_strength', 0)),
    ]

    bar_width = 150
    bar_height = 15
    spacing = 22

    # Draw background
    panel_height = len(labels) * spacing + 35
    cv2.rectangle(frame,
                  (x - 5, y_start - 5),
                  (x + bar_width + 100, y_start + panel_height),
                  (0, 0, 0), -1)

    for i, (label, value) in enumerate(labels):
        y = y_start + i * spacing

        # Label
        cv2.putText(frame, label, (x, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Bar background
        bar_x = x + 95
        cv2.rectangle(frame,
                      (bar_x, y),
                      (bar_x + bar_width, y + bar_height),
                      (50, 50, 50), -1)

        # Bar fill
        fill_width = int(bar_width * min(value, 1.0))
        if fill_width > 0:
            # Color gradient: green to yellow to red
            if value < 0.5:
                color = (0, 255, int(255 * value * 2))
            else:
                color = (0, int(255 * (1 - value) * 2), 255)
            cv2.rectangle(frame,
                          (bar_x, y),
                          (bar_x + fill_width, y + bar_height),
                          color, -1)

        # Value text
        cv2.putText(frame, f"{value:.2f}", (bar_x + bar_width + 5, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Draw BPM as separate text display
    bpm = metrics.get('movement_bpm', 0)
    bpm_y = y_start + len(labels) * spacing + 5
    cv2.putText(frame, f"Movement BPM: {bpm:.0f}",
                (x, bpm_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    return frame


def calculate_frame_metrics(landmarks, prev_landmarks):
    """Calculate real-time metrics for a single frame."""
    if landmarks is None or prev_landmarks is None:
        return None
    
    # Wrist indices
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    
    # Calculate arm velocity
    left_vel = 0
    right_vel = 0
    
    if LEFT_WRIST < len(landmarks) and LEFT_WRIST < len(prev_landmarks):
        dx = landmarks[LEFT_WRIST]['x'] - prev_landmarks[LEFT_WRIST]['x']
        dy = landmarks[LEFT_WRIST]['y'] - prev_landmarks[LEFT_WRIST]['y']
        left_vel = (dx**2 + dy**2) ** 0.5
    
    if RIGHT_WRIST < len(landmarks) and RIGHT_WRIST < len(prev_landmarks):
        dx = landmarks[RIGHT_WRIST]['x'] - prev_landmarks[RIGHT_WRIST]['x']
        dy = landmarks[RIGHT_WRIST]['y'] - prev_landmarks[RIGHT_WRIST]['y']
        right_vel = (dx**2 + dy**2) ** 0.5
    
    arm_velocity = (left_vel + right_vel) / 2
    
    return {
        'arm_velocity': arm_velocity
    }


def export_video(video_path, skeleton_path=None, output_path=None, 
                 show_metrics=True, show_title=True):
    """
    Export video with skeleton overlay.
    
    Args:
        video_path: Path to input video
        skeleton_path: Path to skeleton JSON (auto-detected if None)
        output_path: Path for output video (default: exports/<name>_skeleton.mp4)
        show_metrics: Display real-time metrics on video
        show_title: Display dancer name and song
    
    Returns:
        Path to output video
    """
    
    # Find skeleton data
    if skeleton_path is None:
        skeleton_path = find_skeleton_file(video_path)
        if skeleton_path is None:
            print(f"Error: No skeleton data found for {video_path}")
            print("Run extract_skeleton.py first.")
            return None
    
    print(f"Video: {video_path}")
    print(f"Skeleton: {skeleton_path}")
    
    # Load skeleton data
    skeleton_data = load_skeleton(skeleton_path)
    frames_data = skeleton_data['frames']
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Frames: {total_frames}")
    
    # Set output path
    if output_path is None:
        os.makedirs("exports", exist_ok=True)
        video_name = os.path.basename(video_path).rsplit('.', 1)[0]
        output_path = os.path.join("exports", f"{video_name}_skeleton.mp4")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Extract title from filename
    video_name = os.path.basename(video_path).rsplit('.', 1)[0]
    parts = video_name.split('_')
    dancer_name = parts[0].title() if parts else "Unknown"
    song_name = ' '.join(parts[1:]).title() if len(parts) > 1 else ""
    title = f"{dancer_name} - {song_name}" if song_name else dancer_name
    
    print(f"\nExporting: {title}")
    print(f"Output: {output_path}\n")
    
    # Process frames
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get skeleton for this frame
        landmarks = None
        if frame_idx < len(frames_data):
            landmarks = frames_data[frame_idx].get('landmarks')
        
        # Draw skeleton
        frame = draw_skeleton(frame, landmarks)
        
        # Draw title
        if show_title:
            frame = draw_text_with_bg(frame, title, (10, 30))
        
        # Draw metrics panel
        if show_metrics and frame_idx > 30:  # Wait for enough frames
            metrics = calculate_rolling_metrics(frames_data, frame_idx, window_size=60)
            frame = draw_metrics_panel(frame, metrics)
        
        # Write frame
        out.write(frame)
        
        frame_idx += 1
        
        # Progress
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames ({100*frame_idx/total_frames:.1f}%)")
    
    cap.release()
    out.release()
    
    print(f"\nComplete!")
    print(f"Saved: {output_path}")
    
    return output_path


def batch_export(input_dir="recordings", output_dir="exports"):
    """Export all videos with skeleton overlays."""
    
    video_files = []
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.mp4'):
                video_files.append(os.path.join(root, file))
    
    if not video_files:
        print(f"No video files found in {input_dir}")
        return
    
    print(f"Found {len(video_files)} video files\n")
    
    for i, video_path in enumerate(video_files, 1):
        print(f"\n{'='*50}")
        print(f"Exporting {i}/{len(video_files)}")
        print(f"{'='*50}")
        
        export_video(video_path)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python export_video.py <video_path>           # Single video")
        print("  python export_video.py --batch [input_dir]    # All videos")
        print("  python export_video.py --no-metrics <video>   # Without metrics overlay")
        print("\nExamples:")
        print("  python export_video.py recordings/alice/alice_gallowdance.mp4")
        print("  python export_video.py --batch recordings")
        return
    
    if sys.argv[1] == '--batch':
        input_dir = sys.argv[2] if len(sys.argv) > 2 else "recordings"
        batch_export(input_dir)
    elif sys.argv[1] == '--no-metrics':
        if len(sys.argv) < 3:
            print("Error: Provide video path")
            return
        video_path = sys.argv[2]
        if not os.path.exists(video_path):
            print(f"Error: File not found: {video_path}")
            return
        export_video(video_path, show_metrics=False)
    else:
        video_path = sys.argv[1]
        if not os.path.exists(video_path):
            print(f"Error: File not found: {video_path}")
            return
        export_video(video_path)


if __name__ == "__main__":
    main()
