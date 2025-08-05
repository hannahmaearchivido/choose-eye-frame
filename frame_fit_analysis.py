# frame_fit_analysis.py

import cv2
import numpy as np


def extract_frame_metrics(image_path, landmarks):
    image = cv2.imread(image_path)
    if image is None or landmarks is None:
        return None

    # Get eye and frame landmarks (example indices only)
    eye_left = landmarks.get(33)  # Inner left eye corner
    eye_right = landmarks.get(263)  # Inner right eye corner

    frame_top = landmarks.get(10)  # Top of frame (approx. forehead)
    frame_bottom = landmarks.get(152)  # Bottom of chin/frame

    if not all([eye_left, eye_right, frame_top, frame_bottom]):
        return None

    # Metric 1: Eye distance (horizontal fit)
    eye_distance = np.linalg.norm(np.array(eye_right) - np.array(eye_left))

    # Metric 2: Frame height (vertical fit)
    frame_height = np.linalg.norm(np.array(frame_bottom) - np.array(frame_top))

    # Metric 3: Symmetry (difference between eye positions)
    symmetry = abs(eye_left[1] - eye_right[1])

    return {
        "eye_distance": eye_distance,
        "frame_height": frame_height,
        "symmetry": symmetry
    }


def score_frame_fit(metrics):
    if metrics is None:
        return 0

    # Example scoring logic (tune as needed)
    score = 0
    if 80 <= metrics["eye_distance"] <= 150:
        score += 40
    if 100 <= metrics["frame_height"] <= 200:
        score += 40
    if metrics["symmetry"] <= 10:
        score += 20

    return score


def analyze_frame_fit(image_paths, landmark_data):
    scores = []
    results_dict = {}

    for path in image_paths:
        if path not in landmark_data:
            print(f"[!] Warning: No landmarks for {path}")
            continue

        result = extract_frame_metrics(path, landmark_data[path])
        score = result["fit_score"]
        scores.append(score)
        results_dict[path] = result

    if not scores:
        print("\n⚠️ No valid scores to analyze.")
        return None, []

    # Normalize scores to probabilities
    total_score = sum(scores)
    probabilities = [s / total_score if total_score else 0 for s in scores]

    # Zip everything together
    ranked = sorted(
        zip(image_paths, probabilities, scores),
        key=lambda x: x[1],
        reverse=True
    )

    # Print Results
    print("\n📊 Frame Rankings (by Fit Probability):")
    for i, (path, prob, raw_score) in enumerate(ranked, 1):
        r = results_dict[path]
        print(f"{i}. {path} - Probability: {prob:.2f} (Raw Score: {raw_score}, "
              f"Width Ratio: {r['frame_width_ratio']}, "
              f"Symmetry Offset: {r['symmetry_offset']}, "
              f"Vertical Balance: {r['vertical_balance']})")

    best_frame = ranked[0][0]
    print(f"\n✅ Best fitting frame: {best_frame} with probability {ranked[0][1]:.2f}")
    return best_frame, ranked

