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
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for CLI
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
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


def create_similarity_matrix_pdf(normalized_metrics, output_path=None):
    """Create similarity matrix heatmaps, one page per song."""
    import pandas as pd

    # Organize by song
    by_song = {}
    for name, metrics in normalized_metrics.items():
        parts = name.split(' - ', 1)
        dancer = parts[0]
        song = parts[1].replace('_', ' ').title() if len(parts) > 1 else 'Unknown'
        if song not in by_song:
            by_song[song] = {}
        by_song[song][dancer] = metrics

    with PdfPages(output_path) as pdf:
        for song, dancers in by_song.items():
            dancer_names = list(dancers.keys())
            n = len(dancer_names)

            if n < 2:
                continue

            # Build similarity matrix
            matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i == j:
                        matrix[i, j] = 1.0
                    else:
                        matrix[i, j] = calculate_similarity(
                            dancers[dancer_names[i]],
                            dancers[dancer_names[j]]
                        )

            df = pd.DataFrame(matrix, index=dancer_names, columns=dancer_names)

            fig, ax = plt.subplots(figsize=(max(6, n * 2), max(5, n * 1.5)))

            sns.heatmap(df, annot=True, fmt='.0%', cmap='YlGn',
                        vmin=0, vmax=1, linewidths=2,
                        annot_kws={'size': 18, 'weight': 'bold'},
                        cbar=False, ax=ax)

            ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=14)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=14)

            fig.suptitle(song, fontsize=20, fontweight='bold', y=1.02)
            fig.text(0.5, -0.02,
                     'Similarity = 1 − avg difference. Higher % = more alike.',
                     ha='center', fontsize=10, style='italic', color='gray')

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

    print(f"Saved PDF: {output_path}")


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

    plt.close()


def create_bar_comparison(metrics_dict, output_path=None):
    """Create grouped bar chart comparing dancers using seaborn."""
    import pandas as pd

    metric_keys = ['arm_velocity', 'movement_range', 'vertical_motion',
                   'symmetry', 'stillness_ratio', 'upper_body_focus',
                   'rhythm_strength', 'rhythm_consistency']
    metric_labels = ['Arm Vel.', 'Move Range', 'Vert. Motion',
                     'Symmetry', 'Stillness', 'Upper Body',
                     'Rhythm Str.', 'Rhythm Cons.']

    # Build long-form dataframe for seaborn
    rows = []
    for dancer, metrics in metrics_dict.items():
        for key, label in zip(metric_keys, metric_labels):
            rows.append({
                'Dancer': dancer,
                'Metric': label,
                'Score': metrics.get(key, 0)
            })

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(14, 6))

    sns.barplot(data=df, x='Metric', y='Score', hue='Dancer',
                palette='Dark2', ax=ax)

    ax.set_ylabel('Normalized Score', fontsize=11)
    ax.set_xlabel('')
    ax.set_title('Dance Style Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.legend(title='Dancer', bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved chart: {output_path}")

    plt.close()


def calculate_summary_scores(normalized, raw_metrics):
    """Calculate composite summary scores from individual metrics."""
    summary = {}

    for name, metrics in normalized.items():
        raw = raw_metrics[name]

        # Energy: how much movement overall
        energy = (metrics['arm_velocity'] +
                  metrics['movement_range'] +
                  metrics['vertical_motion']) / 3

        # Control: precision and consistency
        control = (metrics['symmetry'] +
                   metrics.get('rhythm_consistency', 0)) / 2

        # Groove: rhythmic quality
        groove = (metrics.get('rhythm_strength', 0) +
                  metrics.get('rhythm_consistency', 0)) / 2

        # Flow: smooth vs static (inverse of stillness)
        flow = 1 - metrics['stillness_ratio']

        summary[name] = {
            'Energy': energy,
            'Control': control,
            'Groove': groove,
            'Flow': flow,
            'BPM': raw.get('movement_bpm', 0)
        }

    return summary


def get_archetype(scores):
    """Determine dancer archetype based on strongest score."""
    dominated = max(scores.items(), key=lambda x: x[1] if x[0] != 'BPM' else 0)
    archetypes = {
        'Energy': 'Energetic',
        'Control': 'Precise',
        'Groove': 'Groovy',
        'Flow': 'Fluid'
    }
    return archetypes.get(dominated[0], 'Balanced')


def calculate_fingerprints(summary_scores):
    """Calculate dancer fingerprints by averaging across all songs."""
    score_keys = ['Energy', 'Control', 'Groove', 'Flow', 'BPM']

    # Group by dancer
    by_dancer = {}
    for name, scores in summary_scores.items():
        parts = name.split(' - ', 1)
        dancer = parts[0]
        if dancer not in by_dancer:
            by_dancer[dancer] = []
        by_dancer[dancer].append(scores)

    # Average across songs
    fingerprints = {}
    for dancer, score_list in by_dancer.items():
        avg = {}
        for key in score_keys:
            values = [s[key] for s in score_list]
            avg[key] = np.mean(values)
            if key != 'BPM':
                avg[f'{key}_std'] = np.std(values) if len(values) > 1 else 0
        avg['n_songs'] = len(score_list)
        avg['archetype'] = get_archetype(avg)
        fingerprints[dancer] = avg

    return fingerprints


def create_summary_chart_pdf(summary_scores, output_path=None):
    """Create multi-page PDF with summary scores, one page per song, plus fingerprints."""
    import pandas as pd

    score_keys = ['Energy', 'Control', 'Groove', 'Flow']

    # Organize by song
    by_song = {}
    for name, scores in summary_scores.items():
        parts = name.split(' - ', 1)
        dancer = parts[0]
        song = parts[1].replace('_', ' ').title() if len(parts) > 1 else 'Unknown'
        if song not in by_song:
            by_song[song] = {}
        by_song[song][dancer] = scores

    # Calculate fingerprints
    fingerprints = calculate_fingerprints(summary_scores)

    with PdfPages(output_path) as pdf:
        # First page: Dancer Fingerprints
        dancer_names = list(fingerprints.keys())
        n_dancers = len(dancer_names)

        fig, axes = plt.subplots(1, 2, figsize=(12, max(3, n_dancers * 1.4)),
                                  gridspec_kw={'width_ratios': [3, 1.5]})

        # Fingerprint heatmap
        ax1 = axes[0]
        data = [[fingerprints[d][k] for k in score_keys] for d in dancer_names]
        labels = [f"{d}\n({fingerprints[d]['archetype']})" for d in dancer_names]
        df = pd.DataFrame(data, index=labels, columns=score_keys)

        sns.heatmap(df, annot=True, fmt='.2f', cmap='YlOrRd',
                    vmin=0, vmax=1, linewidths=2,
                    annot_kws={'size': 15, 'weight': 'bold'},
                    cbar=False, ax=ax1)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0, ha='center', fontsize=13)
        ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=12)
        ax1.set_ylabel('')

        # Stats panel
        ax2 = axes[1]
        ax2.axis('off')

        stats_text = ""
        for dancer in dancer_names:
            fp = fingerprints[dancer]
            stats_text += f"**{dancer}**\n"
            stats_text += f"  Archetype: {fp['archetype']}\n"
            stats_text += f"  Avg BPM: {fp['BPM']:.0f}\n"
            stats_text += f"  Songs: {fp['n_songs']}\n\n"

        # Simple text display
        y_pos = 0.95
        for dancer in dancer_names:
            fp = fingerprints[dancer]
            ax2.text(0.1, y_pos, dancer, fontsize=14, fontweight='bold',
                     transform=ax2.transAxes, va='top')
            ax2.text(0.1, y_pos - 0.08, f"Style: {fp['archetype']}", fontsize=11,
                     transform=ax2.transAxes, va='top')
            ax2.text(0.1, y_pos - 0.15, f"Avg BPM: {fp['BPM']:.0f}", fontsize=11,
                     transform=ax2.transAxes, va='top')
            ax2.text(0.1, y_pos - 0.22, f"Songs: {fp['n_songs']}", fontsize=11,
                     transform=ax2.transAxes, va='top')
            y_pos -= 0.35

        fig.suptitle('Dancer Fingerprints', fontsize=20, fontweight='bold', y=1.02)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Remaining pages: per-song breakdown
        for song, dancers in by_song.items():
            dancer_names_song = list(dancers.keys())
            n = len(dancer_names_song)

            data = [[dancers[d][k] for k in score_keys] for d in dancer_names_song]
            df = pd.DataFrame(data, index=dancer_names_song, columns=score_keys)

            fig, axes = plt.subplots(1, 2, figsize=(11, max(2.5, n * 1.2)),
                                      gridspec_kw={'width_ratios': [3, 1]})

            ax1 = axes[0]
            sns.heatmap(df, annot=True, fmt='.2f', cmap='YlOrRd',
                        vmin=0, vmax=1, linewidths=2,
                        annot_kws={'size': 16, 'weight': 'bold'},
                        cbar=False, ax=ax1)
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0, ha='center', fontsize=13)
            ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=13)
            ax1.set_ylabel('')

            ax2 = axes[1]
            bpms = [dancers[d]['BPM'] for d in dancer_names_song]
            colors = plt.cm.YlOrRd([b / 200 for b in bpms])

            bars = ax2.barh(dancer_names_song, bpms, color=colors, edgecolor='black', linewidth=1)
            ax2.set_xlabel('BPM', fontsize=12)
            ax2.set_title('Tempo', fontsize=13, fontweight='bold')
            ax2.set_xlim(0, max(bpms) * 1.3 if bpms else 150)

            for bar, bpm in zip(bars, bpms):
                ax2.text(bar.get_width() + 3, bar.get_y() + bar.get_height()/2,
                         f'{bpm:.0f}', va='center', fontsize=12, fontweight='bold')

            ax2.set_yticklabels([])

            fig.suptitle(song, fontsize=18, fontweight='bold', y=1.02)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

    print(f"Saved PDF: {output_path}")


def simplify_label(name):
    """Shorten 'Sev1 - bela_lugosis_dead' to 'Sev1 / Bela'."""
    parts = name.split(' - ', 1)
    dancer = parts[0]
    if len(parts) > 1:
        song = parts[1].replace('_', ' ').title()
        # Take first word of song
        song_short = song.split()[0] if song else ''
        return f"{dancer} / {song_short}"
    return dancer


def create_grouped_heatmaps_pdf(metrics_dict, output_path=None):
    """Create multi-page PDF with grouped heatmaps, one page per song."""
    import pandas as pd

    groups = {
        'Movement': {
            'keys': ['arm_velocity', 'movement_range', 'vertical_motion'],
            'labels': ['Arm Speed', 'Range', 'Bounce']
        },
        'Style': {
            'keys': ['symmetry', 'stillness_ratio', 'upper_body_focus'],
            'labels': ['Symmetry', 'Stillness', 'Upper Focus']
        },
        'Rhythm': {
            'keys': ['rhythm_strength', 'rhythm_consistency'],
            'labels': ['Strength', 'Consistency']
        }
    }

    # Organize by song
    by_song = {}
    for name, metrics in metrics_dict.items():
        parts = name.split(' - ', 1)
        dancer = parts[0]
        song = parts[1].replace('_', ' ').title() if len(parts) > 1 else 'Unknown'
        if song not in by_song:
            by_song[song] = {}
        by_song[song][dancer] = metrics

    with PdfPages(output_path) as pdf:
        for song, dancers in by_song.items():
            dancer_names = list(dancers.keys())
            n_dancers = len(dancer_names)

            fig, axes = plt.subplots(1, 3, figsize=(14, max(2.5, n_dancers * 1.0)))

            for ax, (group_name, group) in zip(axes, groups.items()):
                data = [[dancers[d].get(k, 0) for k in group['keys']] for d in dancer_names]
                df = pd.DataFrame(data, index=dancer_names, columns=group['labels'])

                sns.heatmap(df, annot=True, fmt='.2f', cmap='YlOrRd',
                            vmin=0, vmax=1, linewidths=1, cbar=False,
                            annot_kws={'size': 13, 'weight': 'bold'},
                            ax=ax)

                ax.set_title(group_name, fontsize=14, fontweight='bold')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=11)
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)

                if ax != axes[0]:
                    ax.set_ylabel('')

            fig.suptitle(song, fontsize=18, fontweight='bold', y=1.02)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

    print(f"Saved PDF: {output_path}")


def create_heatmap(metrics_dict, title="Dance Metrics Heatmap", output_path=None):
    """Create a heatmap showing dancers x metrics using seaborn."""
    import pandas as pd

    metric_labels = ['Arm Vel.', 'Move Range', 'Vert. Motion',
                     'Symmetry', 'Stillness', 'Upper Body',
                     'Rhythm Str.', 'Rhythm Cons.']
    metric_keys = ['arm_velocity', 'movement_range', 'vertical_motion',
                   'symmetry', 'stillness_ratio', 'upper_body_focus',
                   'rhythm_strength', 'rhythm_consistency']

    # Simplify labels
    dancers_orig = list(metrics_dict.keys())
    dancers_short = [simplify_label(d) for d in dancers_orig]
    data = [[metrics_dict[d].get(k, 0) for k in metric_keys] for d in dancers_orig]

    df = pd.DataFrame(data, index=dancers_short, columns=metric_labels)

    fig, ax = plt.subplots(figsize=(12, max(4, len(dancers_orig) * 1.0)))

    sns.heatmap(df, annot=True, fmt='.2f', cmap='YlOrRd',
                vmin=0, vmax=1, linewidths=0.5,
                cbar_kws={'label': 'Score', 'shrink': 0.8},
                ax=ax)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved chart: {output_path}")

    plt.close()
    return fig


def create_parallel_coordinates(metrics_dict, title="Dance Style Profiles", output_path=None):
    """Create parallel coordinates plot using pandas."""
    import pandas as pd
    from pandas.plotting import parallel_coordinates

    metric_keys = ['arm_velocity', 'movement_range', 'vertical_motion',
                   'symmetry', 'stillness_ratio', 'upper_body_focus',
                   'rhythm_strength', 'rhythm_consistency']
    metric_labels = ['Arm Vel.', 'Move Range', 'Vert. Motion',
                     'Symmetry', 'Stillness', 'Upper Body',
                     'Rhythm Str.', 'Rhythm Cons.']

    # Build dataframe
    rows = []
    for dancer, metrics in metrics_dict.items():
        row = {'Dancer': dancer}
        for key, label in zip(metric_keys, metric_labels):
            row[label] = metrics.get(key, 0)
        rows.append(row)

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(14, 7))

    parallel_coordinates(df, 'Dancer', ax=ax, colormap='Dark2', linewidth=2.5, alpha=0.8)

    ax.set_ylim(-0.05, 1.1)
    ax.set_ylabel('Normalized Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved chart: {output_path}")

    plt.close()


def create_clustermap(metrics_dict, title="Dance Metrics Clustermap", output_path=None):
    """Create a clustered heatmap showing similarity between dancers and metrics."""
    import pandas as pd

    metric_labels = ['Arm Vel.', 'Move Range', 'Vert. Motion',
                     'Symmetry', 'Stillness', 'Upper Body',
                     'Rhythm Str.', 'Rhythm Cons.']
    metric_keys = ['arm_velocity', 'movement_range', 'vertical_motion',
                   'symmetry', 'stillness_ratio', 'upper_body_focus',
                   'rhythm_strength', 'rhythm_consistency']

    dancers = list(metrics_dict.keys())
    data = [[metrics_dict[d].get(k, 0) for k in metric_keys] for d in dancers]

    df = pd.DataFrame(data, index=dancers, columns=metric_labels)

    g = sns.clustermap(df, annot=True, fmt='.2f', cmap='RdYlGn',
                       vmin=0, vmax=1, linewidths=0.5,
                       figsize=(12, max(6, len(dancers) * 1.2)),
                       dendrogram_ratio=(0.15, 0.15),
                       cbar_pos=(0.02, 0.8, 0.03, 0.15))

    g.fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    if output_path:
        g.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved chart: {output_path}")

    plt.close()


def create_stripplot(metrics_dict, output_path=None):
    """Create strip plot showing metric distributions by dancer."""
    import pandas as pd

    metric_keys = ['arm_velocity', 'movement_range', 'vertical_motion',
                   'symmetry', 'stillness_ratio', 'upper_body_focus',
                   'rhythm_strength', 'rhythm_consistency']
    metric_labels = ['Arm Vel.', 'Move Range', 'Vert. Motion',
                     'Symmetry', 'Stillness', 'Upper Body',
                     'Rhythm Str.', 'Rhythm Cons.']

    rows = []
    for dancer, metrics in metrics_dict.items():
        for key, label in zip(metric_keys, metric_labels):
            rows.append({
                'Dancer': dancer,
                'Metric': label,
                'Score': metrics.get(key, 0)
            })

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(14, 6))

    sns.stripplot(data=df, x='Metric', y='Score', hue='Dancer',
                  palette='Dark2', size=12, ax=ax, dodge=True, jitter=False)

    ax.set_ylabel('Normalized Score', fontsize=11)
    ax.set_xlabel('')
    ax.set_title('Dance Metrics by Dancer', fontsize=14, fontweight='bold')
    ax.set_ylim(-0.05, 1.1)
    ax.legend(title='Dancer', bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved chart: {output_path}")

    plt.close()


def create_rhythm_spectrum(skeleton_data, title="Rhythm Spectrum", fps=30.0):
    """Create FFT spectrum visualization showing frequency components of movement."""
    left_wrist = get_landmark_positions(skeleton_data, LEFT_WRIST)
    right_wrist = get_landmark_positions(skeleton_data, RIGHT_WRIST)

    left_vel = get_velocity_timeseries(left_wrist)
    right_vel = get_velocity_timeseries(right_wrist)
    combined_vel = (left_vel + right_vel) / 2

    if len(combined_vel) < 60:
        return None

    # Apply FFT
    fft_result = np.fft.rfft(combined_vel)
    frequencies = np.fft.rfftfreq(len(combined_vel), 1/fps)
    magnitudes = np.abs(fft_result)

    # Normalize magnitudes
    if np.max(magnitudes) > 0:
        magnitudes = magnitudes / np.max(magnitudes)

    # Focus on dance range
    mask = frequencies <= 5.0  # Up to 300 BPM
    freqs_plot = frequencies[mask]
    mags_plot = magnitudes[mask]

    fig, ax = plt.subplots(figsize=(10, 4))

    # Plot spectrum
    ax.fill_between(freqs_plot, mags_plot, alpha=0.3, color='purple')
    ax.plot(freqs_plot, mags_plot, color='purple', linewidth=1.5)

    # Mark dance-relevant range
    ax.axvspan(0.5, 4.0, alpha=0.1, color='green', label='Dance range (30-240 BPM)')

    # Find and mark peak
    dance_mask = (freqs_plot >= 0.5) & (freqs_plot <= 4.0)
    if np.any(dance_mask):
        dance_freqs = freqs_plot[dance_mask]
        dance_mags = mags_plot[dance_mask]
        peak_idx = np.argmax(dance_mags)
        peak_freq = dance_freqs[peak_idx]
        peak_bpm = peak_freq * 60

        ax.axvline(x=peak_freq, color='red', linestyle='--', linewidth=2,
                   label=f'Peak: {peak_bpm:.0f} BPM')
        ax.scatter([peak_freq], [dance_mags[peak_idx]], color='red', s=100, zorder=5)

    # Add BPM scale on top
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    bpm_ticks = [30, 60, 90, 120, 150, 180, 240, 300]
    ax2.set_xticks([b/60 for b in bpm_ticks])
    ax2.set_xticklabels([str(b) for b in bpm_ticks])
    ax2.set_xlabel('BPM', fontsize=10)

    ax.set_xlabel('Frequency (Hz)', fontsize=11)
    ax.set_ylabel('Magnitude (normalized)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_xlim(0, 5)

    plt.tight_layout()
    return fig


def create_per_song_radar(metrics_by_song, output_path=None):
    """Create a multi-page PDF with one radar chart per song."""
    metric_labels = ['Arm Velocity', 'Movement Range', 'Vertical Motion',
                     'Symmetry', 'Stillness', 'Upper Body',
                     'Rhythm Strength', 'Rhythm Consistency']
    metric_keys = ['arm_velocity', 'movement_range', 'vertical_motion',
                   'symmetry', 'stillness_ratio', 'upper_body_focus',
                   'rhythm_strength', 'rhythm_consistency']

    num_vars = len(metric_labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    with PdfPages(output_path) as pdf:
        # Page for each song
        for song, dancers in metrics_by_song.items():
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

            colors = plt.cm.Dark2(np.linspace(0, 1, len(dancers)))

            for (dancer, metrics), color in zip(dancers.items(), colors):
                values = [metrics.get(k, 0) for k in metric_keys]
                values += values[:1]

                ax.plot(angles, values, 'o-', linewidth=2, label=dancer, color=color)
                ax.fill(angles, values, alpha=0.15, color=color)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metric_labels, size=10)
            ax.set_ylim(0, 1)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax.set_title(f'Song: {song}', size=16, fontweight='bold', y=1.08)

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

        # Summary page: per-dancer averages
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        # Compute per-dancer averages across all songs
        dancer_averages = {}
        for song, dancers in metrics_by_song.items():
            for dancer, metrics in dancers.items():
                if dancer not in dancer_averages:
                    dancer_averages[dancer] = {k: [] for k in metric_keys}
                for k in metric_keys:
                    dancer_averages[dancer][k].append(metrics.get(k, 0))

        for dancer in dancer_averages:
            dancer_averages[dancer] = {k: np.mean(v) for k, v in dancer_averages[dancer].items()}

        colors = plt.cm.Dark2(np.linspace(0, 1, len(dancer_averages)))

        for (dancer, metrics), color in zip(dancer_averages.items(), colors):
            values = [metrics.get(k, 0) for k in metric_keys]
            values += values[:1]

            ax.plot(angles, values, 'o-', linewidth=2, label=dancer, color=color)
            ax.fill(angles, values, alpha=0.15, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, size=10)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.set_title('Dancer Averages (All Songs)', size=16, fontweight='bold', y=1.08)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    print(f"Saved multi-page PDF: {output_path}")


def create_spectrum_report(skeleton_files, output_path=None):
    """Create a PDF with rhythm spectrum for each recording."""
    with PdfPages(output_path) as pdf:
        for filepath in skeleton_files:
            skeleton_data = load_skeleton(filepath)

            # Extract name
            parts = filepath.stem.replace('_skeleton', '').split('_')
            dancer_name = parts[0].title()
            song_name = '_'.join(parts[1:]) if len(parts) > 1 else 'unknown'
            title = f"{dancer_name} - {song_name}"

            fps = skeleton_data['metadata'].get('fps', 30.0)
            fig = create_rhythm_spectrum(skeleton_data, title=f"Rhythm Spectrum: {title}", fps=fps)

            if fig:
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

    print(f"Saved spectrum report: {output_path}")


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

    # Calculate summary scores
    summary_scores = calculate_summary_scores(normalized, all_metrics)

    # Summary scores per song
    summary_path = os.path.join(output_dir, 'charts', 'summary.pdf')
    create_summary_chart_pdf(summary_scores, summary_path)

    # Grouped heatmaps per song (detailed breakdown)
    grouped_path = os.path.join(output_dir, 'charts', 'detailed_metrics.pdf')
    create_grouped_heatmaps_pdf(normalized, grouped_path)

    # Rhythm spectrum report
    spectrum_path = os.path.join(output_dir, 'charts', 'rhythm_spectrums.pdf')
    create_spectrum_report(skeleton_files, spectrum_path)

    # Similarity matrix
    if len(normalized) >= 2:
        similarity_path = os.path.join(output_dir, 'charts', 'similarity.pdf')
        create_similarity_matrix_pdf(normalized, similarity_path)
    
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
