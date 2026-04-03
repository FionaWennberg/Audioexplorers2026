from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# Reuse from your pipeline file
from processing import (
    load_wav_multichannel,
    normalize_audio,
    sliding_doa_analysis,
    cluster_azimuths_simple,
)

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

EXAMPLE_WAV = "example_scene.wav"
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

# True talker positions from the case description / figure.
# Adjust these if your angle convention ends up mirrored.
TRUE_AZIMUTHS = [0, 90, 180, -90]
MAX_SOURCES = 4

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def wrap_angle_deg(a: np.ndarray) -> np.ndarray:
    return (a + 180) % 360 - 180


def circular_distance_deg(a: float, b: float) -> float:
    return abs(((a - b + 180) % 360) - 180)


def estimate_cluster_centers(segments, labels: np.ndarray) -> List[float]:
    centers = []
    for cluster_id in sorted(set(labels.tolist())):
        az = [s.azimuth_deg for s, lab in zip(segments, labels) if lab == cluster_id]
        if len(az) == 0:
            continue
        # circular mean
        az_rad = np.deg2rad(az)
        mean_angle = np.rad2deg(np.arctan2(np.mean(np.sin(az_rad)), np.mean(np.cos(az_rad))))
        centers.append(float(mean_angle))
    return centers


def greedy_match(predicted: List[float], truth: List[float]) -> List[Tuple[float, float, float]]:
    """
    Returns list of tuples: (predicted_angle, true_angle, absolute_circular_error)
    Greedy matching is simple and fine here because there are only 4 speakers.
    """
    remaining_truth = truth.copy()
    matches = []

    for p in predicted:
        if len(remaining_truth) == 0:
            break
        distances = [circular_distance_deg(p, t) for t in remaining_truth]
        best_idx = int(np.argmin(distances))
        t = remaining_truth.pop(best_idx)
        matches.append((p, t, distances[best_idx]))

    return matches


def maybe_flip_or_mirror(predicted: List[float], truth: List[float]) -> Dict:
    """
    Since microphone orientation/sign convention can be flipped,
    test a few simple transforms and keep the best one.
    """
    candidates = {
        "raw": predicted,
        "negated": [-p for p in predicted],
        "plus_180": [wrap_angle_deg(np.array([p + 180]))[0] for p in predicted],
        "negated_plus_180": [wrap_angle_deg(np.array([-p + 180]))[0] for p in predicted],
    }

    best = None
    for name, pred in candidates.items():
        matches = greedy_match(pred, truth.copy())
        if len(matches) == 0:
            continue
        mae = float(np.mean([m[2] for m in matches]))
        if best is None or mae < best["mae"]:
            best = {
                "transform": name,
                "predicted": pred,
                "matches": matches,
                "mae": mae,
            }
    return best


def plot_matches(matches: List[Tuple[float, float, float]], outpath: Path):
    idx = np.arange(len(matches))
    pred = [m[0] for m in matches]
    true = [m[1] for m in matches]

    plt.figure(figsize=(8, 4))
    plt.plot(idx, true, marker="o", label="True azimuth")
    plt.plot(idx, pred, marker="x", label="Predicted azimuth")
    plt.xticks(idx, [f"Speaker {i}" for i in idx])
    plt.ylabel("Azimuth [deg]")
    plt.title("True vs predicted speaker directions")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

# ------------------------------------------------------------
# MAIN EVALUATION
# ------------------------------------------------------------

def evaluate_example_scene():
    print("Loading example WAV...")
    x, sr = load_wav_multichannel(EXAMPLE_WAV)
    x = normalize_audio(x)
    print("Loaded:", x.shape, "sr =", sr)

    print("Running DoA analysis...")
    segments = sliding_doa_analysis(x, sr)
    print("Segments found:", len(segments))
    if len(segments) == 0:
        raise RuntimeError("No active segments found in example WAV.")

    labels = cluster_azimuths_simple(segments, max_sources=MAX_SOURCES)
    predicted_centers = estimate_cluster_centers(segments, labels)
    print("Predicted cluster centers (raw):", np.round(predicted_centers, 1).tolist())

    best = maybe_flip_or_mirror(predicted_centers, TRUE_AZIMUTHS)
    matches = best["matches"]

    print("Best transform:", best["transform"])
    print("Mean absolute angular error:", round(best["mae"], 2), "deg")
    for i, (p, t, e) in enumerate(matches):
        print(f"Match {i}: predicted={p:.1f} deg, true={t:.1f} deg, error={e:.1f} deg")

    plot_matches(matches, OUT_DIR / "doa_evaluation_matches.png")

    report_path = OUT_DIR / "doa_evaluation.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("DOA EVALUATION ON EXAMPLE SCENE\n")
        f.write("===============================\n\n")
        f.write(f"True azimuths: {TRUE_AZIMUTHS}\n")
        f.write(f"Predicted azimuths (raw): {[round(v, 1) for v in predicted_centers]}\n")
        f.write(f"Best transform: {best['transform']}\n")
        f.write(f"Mean absolute angular error: {best['mae']:.2f} deg\n\n")

        for i, (p, t, e) in enumerate(matches):
            f.write(f"Match {i}: predicted={p:.1f} deg, true={t:.1f} deg, error={e:.1f} deg\n")

    print("Saved evaluation files to:", OUT_DIR.resolve())


if __name__ == "__main__":
    try:
        evaluate_example_scene()
    except Exception as e:
        print("EVALUATION FAILED:")
        print(repr(e))
