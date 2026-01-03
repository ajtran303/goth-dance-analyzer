# pyright: reportAttributeAccessIssue=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportOptionalSubscript=false

"""
Analyze skeleton data and compare dance styles.
Calculates metrics like arm velocity, movement range, symmetry, etc.
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# Landmark indices
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28


def load_skeleton(path):
    """Load skeleton data from JSON file."""
    with open(path) as f:
        return json.load(f)


def get_landmark_positions(skeleton_data, landmark_idx):
    """Extract positions for a specific landmark across all frames."""
    positions = []
    for frame in skeleton_data['frames']:
        if frame['landmarks'] is not None:
            lm = frame['landmarks'][landmark_idx]
            positions.append([lm['x'], lm['y'], lm['z']])
        else:
            positions.append(None)
    return positions


def calculate_velocity(positions):
    """Calculate average velocity (movement per frame) for a landmark."""
    velocities = []
    for i in range(1, len(positions)):
        if positions[i] is not None and positions[i-1] is not None:
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            velocity = np.sqrt(dx**2 + dy**2)
            velocities.append(velocity)
    return np.mean(velocities) if velocities else 0


def get_velocity_timeseries(positions):
    """Get velocity as a time series (for FFT analysis)."""
    velocities = []
    for i in range(1, len(positions)):
        if positions[i] is not None and positions[i-1] is not None:
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            velocity = np.sqrt(dx**2 + dy**2)
            velocities.append(velocity)
        else:
            # Interpolate with 0 for missing frames to maintain timing
            velocities.append(0)
    return np.array(velocities)


def calculate_rhythm_metrics(skeleton_data, fps=30.0):
    """
    Calculate rhythm metrics using FFT frequency domain analysis.

    Analyzes the periodicity of arm movements to detect rhythmic patterns.

    Returns dict with:
        - movement_bpm: Dominant movement frequency in beats per minute
        - rhythm_strength: How pronounced the rhythm is (0-1)
        - rhythm_consistency: How stable the rhythm is over time (0-1)
    """
    # Get wrist positions for arm movement analysis
    left_wrist = get_landmark_positions(skeleton_data, LEFT_WRIST)
    right_wrist = get_landmark_positions(skeleton_data, RIGHT_WRIST)

    # Get velocity time series
    left_vel = get_velocity_timeseries(left_wrist)
    right_vel = get_velocity_timeseries(right_wrist)

    # Combine arm velocities
    combined_vel = (left_vel + right_vel) / 2

    if len(combined_vel) < 60:  # Need at least 2 seconds of data
        return None

    # Apply FFT
    fft_result = np.fft.rfft(combined_vel)
    frequencies = np.fft.rfftfreq(len(combined_vel), 1/fps)
    magnitudes = np.abs(fft_result)

    # Focus on dance-relevant frequency range: 0.5-4 Hz (30-240 BPM)
    min_freq, max_freq = 0.5, 4.0
    dance_band_mask = (frequencies >= min_freq) & (frequencies <= max_freq)

    if not np.any(dance_band_mask):
        return None

    dance_freqs = frequencies[dance_band_mask]
    dance_mags = magnitudes[dance_band_mask]

    # Find dominant frequency
    peak_idx = np.argmax(dance_mags)
    dominant_freq = dance_freqs[peak_idx]
    peak_magnitude = dance_mags[peak_idx]

    # Calculate movement BPM
    movement_bpm = dominant_freq * 60

    # Calculate rhythm strength (peak vs noise floor)
    mean_magnitude = np.mean(dance_mags)
    if mean_magnitude > 0:
        rhythm_strength = min((peak_magnitude / mean_magnitude - 1) / 5, 1.0)
        rhythm_strength = max(0, rhythm_strength)
    else:
        rhythm_strength = 0

    # Calculate rhythm consistency using windowed analysis
    window_size = int(fps * 4)  # 4-second windows
    step_size = int(fps * 2)    # 2-second step
    window_bpms = []

    for start in range(0, len(combined_vel) - window_size, step_size):
        window = combined_vel[start:start + window_size]
        w_fft = np.fft.rfft(window)
        w_freqs = np.fft.rfftfreq(len(window), 1/fps)
        w_mags = np.abs(w_fft)

        w_mask = (w_freqs >= min_freq) & (w_freqs <= max_freq)
        if np.any(w_mask):
            w_dance_freqs = w_freqs[w_mask]
            w_dance_mags = w_mags[w_mask]
            w_peak_idx = np.argmax(w_dance_mags)
            window_bpms.append(w_dance_freqs[w_peak_idx] * 60)

    # Consistency is inverse of coefficient of variation
    if len(window_bpms) > 1 and np.mean(window_bpms) > 0:
        cv = np.std(window_bpms) / np.mean(window_bpms)
        rhythm_consistency = max(0, 1 - cv)
    else:
        rhythm_consistency = 0

    return {
        'movement_bpm': float(movement_bpm),
        'rhythm_strength': float(rhythm_strength),
        'rhythm_consistency': float(rhythm_consistency)
    }


def calculate_metrics(skeleton_data):
    """
    Calculate all dance metrics from skeleton data.
    
    Returns dict with:
        - arm_velocity: Average speed of arm movements
        - movement_range: How expansive the movements are
        - vertical_motion: Amount of up/down movement (jumping/bouncing)
        - symmetry: How symmetrical left/right movements are
        - stillness_ratio: Proportion of time spent relatively still
        - upper_body_focus: Ratio of upper vs lower body movement
    """
    
    # Get landmark positions
    left_wrist = get_landmark_positions(skeleton_data, LEFT_WRIST)
    right_wrist = get_landmark_positions(skeleton_data, RIGHT_WRIST)
    left_shoulder = get_landmark_positions(skeleton_data, LEFT_SHOULDER)
    right_shoulder = get_landmark_positions(skeleton_data, RIGHT_SHOULDER)
    left_hip = get_landmark_positions(skeleton_data, LEFT_HIP)
    right_hip = get_landmark_positions(skeleton_data, RIGHT_HIP)
    left_ankle = get_landmark_positions(skeleton_data, LEFT_ANKLE)
    right_ankle = get_landmark_positions(skeleton_data, RIGHT_ANKLE)
    nose = get_landmark_positions(skeleton_data, NOSE)
    
    # Filter out None values for calculations
    def filter_none(positions):
        return [p for p in positions if p is not None]
    
    left_wrist_clean = filter_none(left_wrist)
    right_wrist_clean = filter_none(right_wrist)
    left_hip_clean = filter_none(left_hip)
    right_hip_clean = filter_none(right_hip)
    left_ankle_clean = filter_none(left_ankle)
    right_ankle_clean = filter_none(right_ankle)
    
    if len(left_wrist_clean) < 10:
        return None
    
    # 1. Arm velocity (average of both wrists)
    left_vel = calculate_velocity(left_wrist)
    right_vel = calculate_velocity(right_wrist)
    arm_velocity = (left_vel + right_vel) / 2
    
    # 2. Movement range (standard deviation of wrist positions)
    left_wrist_arr = np.array(left_wrist_clean)
    right_wrist_arr = np.array(right_wrist_clean)
    movement_range = (np.std(left_wrist_arr[:, :2]) + np.std(right_wrist_arr[:, :2])) / 2
    
    # 3. Vertical motion (variance in hip Y position)
    hip_y = [(left_hip_clean[i][1] + right_hip_clean[i][1]) / 2 
             for i in range(min(len(left_hip_clean), len(right_hip_clean)))]
    vertical_motion = np.std(hip_y) if hip_y else 0
    
    # 4. Symmetry (correlation between left and right wrist movement)
    min_len = min(len(left_wrist), len(right_wrist)) - 1
    left_movement = []
    right_movement = []
    for i in range(1, min_len + 1):
        if left_wrist[i] is not None and left_wrist[i-1] is not None:
            left_movement.append(np.sqrt(
                (left_wrist[i][0] - left_wrist[i-1][0])**2 + 
                (left_wrist[i][1] - left_wrist[i-1][1])**2
            ))
        else:
            left_movement.append(0)
        if right_wrist[i] is not None and right_wrist[i-1] is not None:
            right_movement.append(np.sqrt(
                (right_wrist[i][0] - right_wrist[i-1][0])**2 + 
                (right_wrist[i][1] - right_wrist[i-1][1])**2
            ))
        else:
            right_movement.append(0)
    
    if len(left_movement) > 1 and np.std(left_movement) > 0 and np.std(right_movement) > 0:
        symmetry = np.corrcoef(left_movement, right_movement)[0, 1]
        symmetry = max(0, symmetry)  # Clamp to 0-1
    else:
        symmetry = 0
    
    # 5. Stillness ratio (frames with very low total movement)
    total_movement = [left_movement[i] + right_movement[i] for i in range(len(left_movement))]
    if total_movement:
        threshold = np.percentile(total_movement, 20)
        stillness_ratio = np.mean([1 if m < threshold else 0 for m in total_movement])
    else:
        stillness_ratio = 0
    
    # 6. Upper body focus (arm movement vs leg movement)
    left_ankle_vel = calculate_velocity(left_ankle)
    right_ankle_vel = calculate_velocity(right_ankle)
    leg_velocity = (left_ankle_vel + right_ankle_vel) / 2
    
    if leg_velocity > 0:
        upper_body_focus = arm_velocity / (arm_velocity + leg_velocity)
    else:
        upper_body_focus = 1.0
    
    # Get rhythm metrics from FFT analysis
    fps = skeleton_data['metadata'].get('fps', 30.0)
    rhythm_metrics = calculate_rhythm_metrics(skeleton_data, fps)

    metrics = {
        'arm_velocity': float(arm_velocity),
        'movement_range': float(movement_range),
        'vertical_motion': float(vertical_motion),
        'symmetry': float(symmetry),
        'stillness_ratio': float(stillness_ratio),
        'upper_body_focus': float(upper_body_focus)
    }

    # Add rhythm metrics if available
    if rhythm_metrics:
        metrics.update(rhythm_metrics)
    else:
        metrics['movement_bpm'] = 0.0
        metrics['rhythm_strength'] = 0.0
        metrics['rhythm_consistency'] = 0.0

    return metrics


def normalize_metrics(all_metrics):
    """Normalize metrics to 0-1 range across all dancers for comparison."""
    if not all_metrics:
        return {}
    
    metric_names = list(next(iter(all_metrics.values())).keys())
    normalized = {}
    
    for name, metrics in all_metrics.items():
        normalized[name] = {}
    
    for metric in metric_names:
        values = [m[metric] for m in all_metrics.values()]
        min_val, max_val = min(values), max(values)
        range_val = max_val - min_val if max_val > min_val else 1
        
        for name, metrics in all_metrics.items():
            normalized[name][metric] = (metrics[metric] - min_val) / range_val
    
    return normalized


def calculate_similarity(metrics1, metrics2):
    """Calculate similarity score between two dancers (0-1, higher = more similar)."""
    keys = metrics1.keys()
    differences = [abs(metrics1[k] - metrics2[k]) for k in keys]
    avg_diff = np.mean(differences)
    return 1 - avg_diff


def find_most_similar(normalized_metrics):
    """Find the most similar pair of dancers."""
    names = list(normalized_metrics.keys())
    if len(names) < 2:
        return None
    
    best_pair = None
    best_similarity = -1
    
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            sim = calculate_similarity(
                normalized_metrics[names[i]], 
                normalized_metrics[names[j]]
            )
            if sim > best_similarity:
                best_similarity = sim
                best_pair = (names[i], names[j])
    
    return best_pair, best_similarity


def create_radar_chart(metrics_dict, title="Dance Style Comparison", output_path=None):
    """Create radar chart comparing multiple dancers."""
    labels = ['Arm Velocity', 'Movement Range', 'Vertical Motion',
              'Symmetry', 'Stillness', 'Upper Body Focus',
              'Rhythm Strength', 'Rhythm Consistency']
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    colors = plt.cm.Dark2(np.linspace(0, 1, len(metrics_dict)))

    for (name, metrics), color in zip(metrics_dict.items(), colors):
        values = [
            metrics['arm_velocity'],
            metrics['movement_range'],
            metrics['vertical_motion'],
            metrics['symmetry'],
            metrics['stillness_ratio'],
            metrics['upper_body_focus'],
            metrics.get('rhythm_strength', 0),
            metrics.get('rhythm_consistency', 0)
        ]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title(title, size=16, y=1.08)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved chart: {output_path}")

    plt.show()


def create_bar_comparison(metrics_dict, output_path=None):
    """Create grouped bar chart comparing dancers."""
    labels = ['Arm\nVelocity', 'Movement\nRange', 'Vertical\nMotion',
              'Symmetry', 'Stillness', 'Upper Body\nFocus',
              'Rhythm\nStrength', 'Rhythm\nConsistency']

    x = np.arange(len(labels))
    width = 0.8 / len(metrics_dict)

    fig, ax = plt.subplots(figsize=(16, 6))

    colors = plt.cm.Dark2(np.linspace(0, 1, len(metrics_dict)))

    for i, ((name, metrics), color) in enumerate(zip(metrics_dict.items(), colors)):
        values = [
            metrics['arm_velocity'],
            metrics['movement_range'],
            metrics['vertical_motion'],
            metrics['symmetry'],
            metrics['stillness_ratio'],
            metrics['upper_body_focus'],
            metrics.get('rhythm_strength', 0),
            metrics.get('rhythm_consistency', 0)
        ]
        offset = (i - len(metrics_dict) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=name, color=color)

    ax.set_ylabel('Normalized Score')
    ax.set_title('Dance Style Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.1)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved chart: {output_path}")

    plt.show()


def analyze_directory(skeleton_dir="skeleton_data", output_dir="analysis"):
    """Analyze all skeleton files and generate comparisons."""
    
    # Find all skeleton JSON files
    skeleton_files = list(Path(skeleton_dir).rglob("*_skeleton.json"))
    
    if not skeleton_files:
        print(f"No skeleton files found in {skeleton_dir}")
        return
    
    print(f"Found {len(skeleton_files)} skeleton files\n")
    
    # Calculate metrics for each
    all_metrics = {}
    
    for filepath in skeleton_files:
        # Extract dancer name from path
        parts = filepath.stem.replace('_skeleton', '').split('_')
        dancer_name = parts[0].title()
        song_name = '_'.join(parts[1:]) if len(parts) > 1 else 'unknown'
        
        key = f"{dancer_name} - {song_name}"
        
        print(f"Analyzing: {key}")
        skeleton_data = load_skeleton(filepath)
        metrics = calculate_metrics(skeleton_data)
        
        if metrics:
            all_metrics[key] = metrics
            print(f"  Detection rate: {skeleton_data['metadata'].get('detection_rate', 0):.1%}")
        else:
            print(f"  Skipped (insufficient data)")
    
    if not all_metrics:
        print("\nNo valid data to analyze")
        return
    
    # Normalize for comparison
    normalized = normalize_metrics(all_metrics)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'charts'), exist_ok=True)
    
    # Save raw metrics
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({
            'raw': all_metrics,
            'normalized': normalized
        }, f, indent=2)
    print(f"\nSaved metrics: {metrics_path}")
    
    # Find most similar pair
    if len(normalized) >= 2:
        pair, similarity = find_most_similar(normalized)
        print(f"\nMost similar: {pair[0]} & {pair[1]} ({similarity:.1%} similar)")
    
    # Generate charts
    print("\nGenerating charts...")
    
    # Radar chart
    radar_path = os.path.join(output_dir, 'charts', 'comparison_radar.png')
    create_radar_chart(normalized, "Dance Style Comparison", radar_path)
    
    # Bar chart
    bar_path = os.path.join(output_dir, 'charts', 'comparison_bars.png')
    create_bar_comparison(normalized, bar_path)
    
    # Print summary
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)

    for name, metrics in normalized.items():
        raw_metrics = all_metrics[name]
        print(f"\n{name}:")
        print(f"  Arm Velocity:       {'█' * int(metrics['arm_velocity'] * 20):<20} {metrics['arm_velocity']:.2f}")
        print(f"  Movement Range:     {'█' * int(metrics['movement_range'] * 20):<20} {metrics['movement_range']:.2f}")
        print(f"  Vertical Motion:    {'█' * int(metrics['vertical_motion'] * 20):<20} {metrics['vertical_motion']:.2f}")
        print(f"  Symmetry:           {'█' * int(metrics['symmetry'] * 20):<20} {metrics['symmetry']:.2f}")
        print(f"  Stillness:          {'█' * int(metrics['stillness_ratio'] * 20):<20} {metrics['stillness_ratio']:.2f}")
        print(f"  Upper Body Focus:   {'█' * int(metrics['upper_body_focus'] * 20):<20} {metrics['upper_body_focus']:.2f}")
        print(f"  Movement BPM:       {raw_metrics.get('movement_bpm', 0):.1f}")
        print(f"  Rhythm Strength:    {'█' * int(metrics.get('rhythm_strength', 0) * 20):<20} {metrics.get('rhythm_strength', 0):.2f}")
        print(f"  Rhythm Consistency: {'█' * int(metrics.get('rhythm_consistency', 0) * 20):<20} {metrics.get('rhythm_consistency', 0):.2f}")


def analyze_single(skeleton_path):
    """Analyze a single skeleton file."""
    print(f"Analyzing: {skeleton_path}\n")

    skeleton_data = load_skeleton(skeleton_path)
    metrics = calculate_metrics(skeleton_data)

    if not metrics:
        print("Error: Insufficient data for analysis")
        return

    print("Metrics:")
    print(f"  Arm Velocity:       {metrics['arm_velocity']:.4f}")
    print(f"  Movement Range:     {metrics['movement_range']:.4f}")
    print(f"  Vertical Motion:    {metrics['vertical_motion']:.4f}")
    print(f"  Symmetry:           {metrics['symmetry']:.4f}")
    print(f"  Stillness Ratio:    {metrics['stillness_ratio']:.4f}")
    print(f"  Upper Body Focus:   {metrics['upper_body_focus']:.4f}")

    print("\nRhythm Analysis (FFT):")
    print(f"  Movement BPM:       {metrics.get('movement_bpm', 0):.1f}")
    print(f"  Rhythm Strength:    {metrics.get('rhythm_strength', 0):.4f}")
    print(f"  Rhythm Consistency: {metrics.get('rhythm_consistency', 0):.4f}")

    print(f"\nDetection rate: {skeleton_data['metadata'].get('detection_rate', 0):.1%}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python analyze.py                      # Analyze all in skeleton_data/")
        print("  python analyze.py <skeleton.json>      # Analyze single file")
        print("  python analyze.py --dir <directory>    # Analyze specific directory")
        return
    
    if sys.argv[1] == '--dir':
        skeleton_dir = sys.argv[2] if len(sys.argv) > 2 else "skeleton_data"
        analyze_directory(skeleton_dir)
    elif sys.argv[1].endswith('.json'):
        if not os.path.exists(sys.argv[1]):
            print(f"Error: File not found: {sys.argv[1]}")
            return
        analyze_single(sys.argv[1])
    else:
        analyze_directory()


if __name__ == "__main__":
    main()
