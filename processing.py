import os
from pathlib import Path
import numpy as np
import soundfile as sf
import scipy.signal as sig
from scipy.fft import rfft, rfftfreq
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# Optional packages:
# pip install librosa matplotlib openai-whisper pyroomacoustics
import librosa
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

EXAMPLE_WAV = "example_scene.wav"
MAIN_WAV = "main_scene.wav"
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

# Channel order from the case:
# [left front, left rear, right front, right rear]
CH_LF = 0
CH_LR = 1
CH_RF = 2
CH_RR = 3

# Approximate geometry assumptions.
# These do not need to be exact for a baseline system, but document them in the report.
MIC_FRONT_REAR_DISTANCE = 0.012  # 1.2 cm within a hearing aid
EAR_DISTANCE = 0.18              # left-right spacing across head, approx 18 cm
C = 343.0                        # speed of sound (m/s)

FRAME_SEC = 1.0
HOP_SEC = 0.25
MAX_SOURCES = 4
TARGET_RULE = "front_most"  # one of: front_most, loudest

# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class SegmentDoA:
    start_s: float
    end_s: float
    tau_lr: float
    tau_fb_left: float
    tau_fb_right: float
    azimuth_deg: float
    energy: float

@dataclass
class SpeakerHypothesis:
    cluster_id: int
    azimuth_deg: float
    mean_energy: float
    gender_guess: str
    transcript: str
    language_guess: str

# ============================================================
# I/O
# ============================================================

def load_wav_multichannel(path: str) -> Tuple[np.ndarray, int]:
    x, sr = sf.read(path)
    if x.ndim != 2 or x.shape[1] != 4:
        raise ValueError(f"Expected a 4-channel WAV, got shape {x.shape}")
    x = x.astype(np.float32)
    return x, sr

# ============================================================
# BASIC UTILITIES
# ============================================================

def normalize_audio(x: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(x)) + 1e-8
    return x / peak


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def dominant_f0_basic(y: np.ndarray, sr: int, fmin: float = 70, fmax: float = 300) -> Optional[float]:
    """
    Very simple pitch estimate for voiced speech regions.
    This is only a rough baseline.
    """
    y = y - np.mean(y)
    if rms(y) < 1e-4:
        return None

    spectrum = np.abs(rfft(y * np.hanning(len(y))))
    freqs = rfftfreq(len(y), 1 / sr)
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return None
    peak_idx = np.argmax(spectrum[mask])
    return float(freqs[mask][peak_idx])


def guess_gender_from_f0(f0: Optional[float]) -> str:
    if f0 is None:
        return "unknown"
    if f0 < 165:
        return "likely male"
    return "likely female"

# ============================================================
# GCC-PHAT FOR TDOA
# ============================================================

def gcc_phat(sig1: np.ndarray, sig2: np.ndarray, fs: int, max_tau: Optional[float] = None) -> float:
    n = sig1.shape[0] + sig2.shape[0]
    SIG1 = np.fft.rfft(sig1, n=n)
    SIG2 = np.fft.rfft(sig2, n=n)
    R = SIG1 * np.conj(SIG2)
    R /= np.abs(R) + 1e-12
    cc = np.fft.irfft(R, n=n)

    max_shift = int(n / 2)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))

    if max_tau is not None:
        max_shift = min(int(fs * max_tau), max_shift)
        mid = len(cc) // 2
        cc = cc[mid - max_shift: mid + max_shift + 1]
        shift = np.argmax(np.abs(cc)) - max_shift
    else:
        shift = np.argmax(np.abs(cc)) - (len(cc) // 2)

    return shift / float(fs)

# ============================================================
# DOA ESTIMATION
# ============================================================

def estimate_segment_doa(frame: np.ndarray, sr: int) -> SegmentDoA:
    """
    frame shape: [samples, 4]
    We estimate:
      - left/right delay using average of left pair vs right pair
      - front/back delay inside each ear
    Then derive a heuristic azimuth.
    """
    left_mix = 0.5 * (frame[:, CH_LF] + frame[:, CH_LR])
    right_mix = 0.5 * (frame[:, CH_RF] + frame[:, CH_RR])

    tau_lr = gcc_phat(left_mix, right_mix, sr, max_tau=EAR_DISTANCE / C)
    tau_fb_left = gcc_phat(frame[:, CH_LF], frame[:, CH_LR], sr, max_tau=MIC_FRONT_REAR_DISTANCE / C)
    tau_fb_right = gcc_phat(frame[:, CH_RF], frame[:, CH_RR], sr, max_tau=MIC_FRONT_REAR_DISTANCE / C)

    # Left-right azimuth proxy: arcsin(c*tau/d)
    arg = np.clip((C * tau_lr) / EAR_DISTANCE, -1.0, 1.0)
    az_lr = np.degrees(np.arcsin(arg))  # approx in [-90, 90]

    # Front/back heuristic from average front-rear delay across ears.
    fb = 0.5 * (tau_fb_left + tau_fb_right)
    # If front mics lead rear mics, treat as front hemisphere.
    if fb < 0:
        azimuth = az_lr
    else:
        # Push estimate toward back hemisphere.
        azimuth = 180 - az_lr if az_lr >= 0 else -180 - az_lr

    energy = rms(np.mean(frame, axis=1))

    return SegmentDoA(
        start_s=0.0,
        end_s=0.0,
        tau_lr=tau_lr,
        tau_fb_left=tau_fb_left,
        tau_fb_right=tau_fb_right,
        azimuth_deg=float(azimuth),
        energy=energy,
    )


def sliding_doa_analysis(x: np.ndarray, sr: int, frame_sec: float = FRAME_SEC, hop_sec: float = HOP_SEC) -> List[SegmentDoA]:
    frame_len = int(frame_sec * sr)
    hop_len = int(hop_sec * sr)
    segments = []
    for start in range(0, len(x) - frame_len + 1, hop_len):
        stop = start + frame_len
        frame = x[start:stop]
        if rms(np.mean(frame, axis=1)) < 1e-3:
            continue
        seg = estimate_segment_doa(frame, sr)
        seg.start_s = start / sr
        seg.end_s = stop / sr
        segments.append(seg)
    return segments

# ============================================================
# CLUSTER SEGMENTS INTO SPEAKER DIRECTIONS
# ============================================================

def wrap_angle_deg(a: np.ndarray) -> np.ndarray:
    return (a + 180) % 360 - 180


def cluster_azimuths_simple(segments: List[SegmentDoA], max_sources: int = MAX_SOURCES) -> np.ndarray:
    """
    Very simple 1D clustering based on histogram peaks.
    For a stronger version, replace this with KMeans on [cos(theta), sin(theta)].
    """
    az = np.array([s.azimuth_deg for s in segments])
    hist, edges = np.histogram(az, bins=36, range=(-180, 180))
    peak_bins = np.argsort(hist)[::-1][:max_sources]
    centers = 0.5 * (edges[peak_bins] + edges[peak_bins + 1])

    labels = np.zeros(len(az), dtype=int)
    for i, a in enumerate(az):
        d = np.abs(wrap_angle_deg(a - centers))
        labels[i] = int(np.argmin(d))
    return labels

# ============================================================
# SPEAKER-WISE EXTRACTION
# ============================================================

def make_mono_reference(x: np.ndarray) -> np.ndarray:
    return np.mean(x, axis=1)


def extract_time_regions_for_cluster(x: np.ndarray, sr: int, segments: List[SegmentDoA], labels: np.ndarray, cluster_id: int) -> np.ndarray:
    parts = []
    for seg, lab in zip(segments, labels):
        if lab != cluster_id:
            continue
        s = int(seg.start_s * sr)
        e = int(seg.end_s * sr)
        parts.append(make_mono_reference(x[s:e]))
    if len(parts) == 0:
        return np.zeros(1, dtype=np.float32)
    return np.concatenate(parts)

# ============================================================
# TRANSCRIPTION
# ============================================================

def transcribe_audio_whisper(path: str, model_name: str = "small") -> Dict:
    """
    pip install openai-whisper
    """
    import whisper
    model = whisper.load_model(model_name)
    result = model.transcribe(path)
    return result

# ============================================================
# TARGET SELECTION
# ============================================================

def choose_target_speaker(speakers: List[SpeakerHypothesis]) -> int:
    if TARGET_RULE == "loudest":
        return int(np.argmax([s.mean_energy for s in speakers]))

    # Default: speaker closest to 0 degrees (front)
    return int(np.argmin([abs(wrap_angle_deg(np.array([s.azimuth_deg]))[0]) for s in speakers]))

# ============================================================
# VERY SIMPLE ENHANCEMENT
# ============================================================

def simple_front_beam_like_mix(x: np.ndarray, target_azimuth_deg: float) -> np.ndarray:
    """
    This is NOT a true delay-and-sum beamformer because exact geometry is unknown.
    It is a simple baseline:
      - for frontish targets: average front mics
      - for backish targets: average rear mics
      - for leftish targets: average left mics
      - for rightish targets: average right mics
    """
    a = wrap_angle_deg(np.array([target_azimuth_deg]))[0]
    if abs(a) <= 45:
        y = 0.5 * (x[:, CH_LF] + x[:, CH_RF])
    elif abs(a) >= 135:
        y = 0.5 * (x[:, CH_LR] + x[:, CH_RR])
    elif a > 0:
        y = 0.5 * (x[:, CH_RF] + x[:, CH_RR])
    else:
        y = 0.5 * (x[:, CH_LF] + x[:, CH_LR])
    return normalize_audio(y)

# ============================================================
# PLOTTING
# ============================================================

def plot_doa_timeline(segments: List[SegmentDoA], outpath: Path):
    if len(segments) == 0:
        return
    t = [0.5 * (s.start_s + s.end_s) for s in segments]
    a = [s.azimuth_deg for s in segments]
    plt.figure(figsize=(10, 4))
    plt.scatter(t, a, s=10)
    plt.xlabel("Time [s]")
    plt.ylabel("Estimated azimuth [deg]")
    plt.title("DoA estimates over time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(example_wav: str, main_wav: str):
    print("Current working directory:", os.getcwd())
    print("Example path:", example_wav)
    print("Main path:", main_wav)
    print("Example exists:", Path(example_wav).exists())
    print("Main exists:", Path(main_wav).exists())

    print("Loading example...")
    x_ex, sr_ex = load_wav_multichannel(example_wav)
    print("Example loaded:", x_ex.shape, "sr =", sr_ex)

    print("Loading main...")
    x_main, sr_main = load_wav_multichannel(main_wav)
    print("Main loaded:", x_main.shape, "sr =", sr_main)
    print("Loading example...")
    x_ex, sr_ex = load_wav_multichannel(example_wav)
    print("Loading main...")
    x_main, sr_main = load_wav_multichannel(main_wav)

    if sr_ex != sr_main:
        raise ValueError("Example and main WAV files should use the same sample rate for this baseline")

    sr = sr_main
    x_ex = normalize_audio(x_ex)
    x_main = normalize_audio(x_main)

    # 1) Example scene analysis
    print("Running DoA analysis on example scene...")
    ex_segments = sliding_doa_analysis(x_ex, sr)
    plot_doa_timeline(ex_segments, OUT_DIR / "example_doa_timeline.png")

    # 2) Main scene analysis
    print("Running DoA analysis on main scene...")
    main_segments = sliding_doa_analysis(x_main, sr)
    plot_doa_timeline(main_segments, OUT_DIR / "main_doa_timeline.png")

    if len(main_segments) == 0:
        raise RuntimeError("No active speech segments found")

    labels = cluster_azimuths_simple(main_segments, max_sources=MAX_SOURCES)

    # 3) Build speaker hypotheses
    speakers: List[SpeakerHypothesis] = []
    for cluster_id in sorted(set(labels.tolist())):
        cluster_segments = [s for s, lab in zip(main_segments, labels) if lab == cluster_id]
        audio_cluster = extract_time_regions_for_cluster(x_main, sr, main_segments, labels, cluster_id)

        f0 = dominant_f0_basic(audio_cluster, sr)
        gender = guess_gender_from_f0(f0)
        azimuth = float(np.mean([s.azimuth_deg for s in cluster_segments]))
        energy = float(np.mean([s.energy for s in cluster_segments]))

        wav_path = OUT_DIR / f"speaker_cluster_{cluster_id}.wav"
        sf.write(wav_path, normalize_audio(audio_cluster), sr)

        # Optional transcription
        transcript = ""
        lang = "unknown"
        try:
            result = transcribe_audio_whisper(str(wav_path), model_name="small")
            transcript = result.get("text", "").strip()
            lang = result.get("language", "unknown")
        except Exception as e:
            transcript = f"[transcription unavailable: {e}]"

        speakers.append(
            SpeakerHypothesis(
                cluster_id=cluster_id,
                azimuth_deg=azimuth,
                mean_energy=energy,
                gender_guess=gender,
                transcript=transcript,
                language_guess=lang,
            )
        )

    # 4) Choose target speaker
    target_idx = choose_target_speaker(speakers)
    target = speakers[target_idx]

    # 5) Enhance target speaker (very simple baseline)
    enhanced = simple_front_beam_like_mix(x_main, target.azimuth_deg)
    out_enhanced = OUT_DIR / "enhanced_target.wav"
    sf.write(out_enhanced, enhanced, sr)

    # 6) Save summary
    summary_path = OUT_DIR / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("AUDIO CASE SUMMARY\n")
        f.write("==================\n\n")

        f.write(f"Estimated number of talkers: {len(speakers)}\n\n")

        for s in speakers:
            f.write(f"Speaker {s.cluster_id}\n")
            f.write(f"  Estimated azimuth: {s.azimuth_deg:.1f} deg\n")
            f.write(f"  Mean energy: {s.mean_energy:.4f}\n")
            f.write(f"  Gender guess: {s.gender_guess}\n")
            f.write(f"  Language guess: {s.language_guess}\n")
            f.write(f"  Transcript: {s.transcript}\n\n")

        f.write(f"Chosen target speaker: {target.cluster_id}\n")
        f.write(f"Reason: TARGET_RULE = {TARGET_RULE}\n")
        f.write(f"Enhanced file: {out_enhanced}\n")

print("Script loaded. About to start pipeline...")

if __name__ == "__main__":
    try:
        run_pipeline(EXAMPLE_WAV, MAIN_WAV)
    except Exception as e:
        print("PIPELINE FAILED:")
        print(repr(e))
