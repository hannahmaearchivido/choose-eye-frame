import os
from get_landmarks_for_image import get_face_landmarks
from frame_fit_analysis import extract_frame_metrics, score_frame_fit

folder = "static/Photos"
results = []

def run_fit_test(image_paths):
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder, filename)
            print(f"[INFO] Processing: {image_path}")
            landmarks = get_face_landmarks(image_path)

            if landmarks is None:
                continue  # skip if no landmarks

            metrics = extract_frame_metrics(image_path, landmarks)
            score = score_frame_fit(metrics)

            if metrics is None:
                continue  # skip if metric extraction failed

            results.append({
                "filename": filename,
                "score": score,
                "contour_metrics": metrics
            })

    # Optional: print or save results
    for res in results:
        print(f"{res['filename']} -> Score: {res['score']}, Metrics: {res['contour_metrics']}")

    # Get best fitting frame
    if results:
        best = max(results, key=lambda r: r.get("score", 0))
        print(f"\n[✓] Best Fit: {best['filename']} with Score: {best['score']}")
    else:
        print("\n[!] No valid frame fits detected.")


