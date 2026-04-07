from pathlib import Path
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

MAIN_WAV = "main_scene.wav"
ENHANCED_WAV = "outputs/enhanced_target.wav"
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

CH_LF = 0
CH_LR = 1
CH_RF = 2
CH_RR = 3

START_SEC = 0
END_SEC = 4


def load_wav(path):
    x, sr = sf.read(path)
    return x.astype(np.float32), sr


def crop(x, sr, start_sec, end_sec):
    s = int(start_sec * sr)
    e = int(end_sec * sr)
    return x[s:e]


def rms_envelope(x, sr, win_sec=0.05, hop_sec=0.01):
    win = int(win_sec * sr)
    hop = int(hop_sec * sr)
    vals, times = [], []
    for start in range(0, len(x) - win + 1, hop):
        frame = x[start:start+win]
        vals.append(np.sqrt(np.mean(frame**2) + 1e-12))
        times.append((start + win/2) / sr)
    return np.array(times), np.array(vals)


def normalize(v):
    vmax = np.max(np.abs(v)) + 1e-12
    return v / vmax


def main():
    x, sr = load_wav(MAIN_WAV)
    y, sr_y = load_wav(ENHANCED_WAV)

    if sr != sr_y:
        raise ValueError("Sample rates do not match.")

    x = crop(x, sr, START_SEC, END_SEC)
    y = crop(y, sr, START_SEC, END_SEC)

    if y.ndim == 2:
        y = np.mean(y, axis=1)

    front = 0.5 * (x[:, CH_LF] + x[:, CH_RF])
    rear  = 0.5 * (x[:, CH_LR] + x[:, CH_RR])
    left  = 0.5 * (x[:, CH_LF] + x[:, CH_LR])
    right = 0.5 * (x[:, CH_RF] + x[:, CH_RR])

    t_f, e_front = rms_envelope(front, sr)
    t_r, e_rear  = rms_envelope(rear, sr)
    t_l, e_left  = rms_envelope(left, sr)
    t_rt, e_right = rms_envelope(right, sr)
    t_y, e_enh   = rms_envelope(y, sr)

    e_front_n = normalize(e_front)
    e_rear_n = normalize(e_rear)
    e_left_n = normalize(e_left)
    e_right_n = normalize(e_right)
    e_enh_n = normalize(e_enh)

    plt.figure(figsize=(10, 5))
    plt.plot(t_f, e_front_n, label="Front pair")
    plt.plot(t_r, e_rear_n, label="Rear pair")
    plt.plot(t_l, e_left_n, label="Left pair")
    plt.plot(t_rt, e_right_n, label="Right pair")
    plt.plot(t_y, e_enh_n, label="Enhanced", linewidth=2)
    plt.xlabel("Time [s]")
    plt.ylabel("Normalized RMS envelope")
    plt.title("Directional RMS envelope comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "directional_rms_comparison.png", dpi=200)
    plt.close()

    def corr(a, b):
        n = min(len(a), len(b))
        return np.corrcoef(a[:n], b[:n])[0, 1]

    print("Correlation with enhanced:")
    print("Front:", corr(e_front_n, e_enh_n))
    print("Rear :", corr(e_rear_n, e_enh_n))
    print("Left :", corr(e_left_n, e_enh_n))
    print("Right:", corr(e_right_n, e_enh_n))


if __name__ == "__main__":
    main()