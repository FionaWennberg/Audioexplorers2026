# from pathlib import Path
# import numpy as np
# import soundfile as sf
# import matplotlib.pyplot as plt
# import scipy.signal as sig


# MAIN_WAV = "main_scene.wav"
# ENHANCED_WAV = "outputs/enhanced_target.wav"
# OUT_DIR = Path("outputs")
# OUT_DIR.mkdir(exist_ok=True)


# def load_audio(path: str):
#     x, sr = sf.read(path)
#     x = x.astype(np.float32)

#     # If multichannel, convert to mono for plotting
#     if x.ndim == 2:
#         x = 0.5 * (x[:, 0] + x[:, 2])

#     return x, sr


# def plot_single_spectrogram(x, sr, title, outpath):
#     f, t, Sxx = sig.spectrogram(
#         x,
#         fs=sr,
#         window="hann",
#         nperseg=1024,
#         noverlap=768,
#         scaling="spectrum",
#         mode="magnitude"
#     )

#     Sxx_db = 20 * np.log10(Sxx + 1e-10)

#     plt.figure(figsize=(10, 4))
#     plt.pcolormesh(t, f, Sxx_db, shading="gouraud")
#     plt.ylabel("Frequency [Hz]")
#     plt.xlabel("Time [s]")
#     plt.title(title)
#     plt.colorbar(label="Magnitude [dB]")
#     plt.ylim(0, 5000)  # speech-relevant range
#     plt.tight_layout()
#     plt.savefig(outpath, dpi=200)
#     plt.close()


# def plot_comparison_spectrograms(x1, sr1, x2, sr2, outpath):
#     if sr1 != sr2:
#         raise ValueError("Sample rates do not match.")

#     f1, t1, S1 = sig.spectrogram(
#         x1,
#         fs=sr1,
#         window="hann",
#         nperseg=1024,
#         noverlap=768,
#         scaling="spectrum",
#         mode="magnitude"
#     )
#     f2, t2, S2 = sig.spectrogram(
#         x2,
#         fs=sr2,
#         window="hann",
#         nperseg=1024,
#         noverlap=768,
#         scaling="spectrum",
#         mode="magnitude"
#     )

#     S1_db = 20 * np.log10(S1 + 1e-10)
#     S2_db = 20 * np.log10(S2 + 1e-10)

#     plt.figure(figsize=(10, 8))

#     plt.subplot(2, 1, 1)
#     plt.pcolormesh(t1, f1, S1_db, shading="gouraud")
#     plt.ylabel("Frequency [Hz]")
#     plt.title("Original mixture")
#     plt.ylim(0, 5000)
#     plt.colorbar(label="Magnitude [dB]")

#     plt.subplot(2, 1, 2)
#     plt.pcolormesh(t2, f2, S2_db, shading="gouraud")
#     plt.ylabel("Frequency [Hz]")
#     plt.xlabel("Time [s]")
#     plt.title("Enhanced target signal")
#     plt.ylim(0, 5000)
#     plt.colorbar(label="Magnitude [dB]")

#     plt.tight_layout()
#     plt.savefig(outpath, dpi=200)
#     plt.close()


# def main():
#     x_main, sr_main = load_audio(MAIN_WAV)
#     x_enh, sr_enh = load_audio(ENHANCED_WAV)

#     plot_single_spectrogram(
#         x_main,
#         sr_main,
#         "Spectrogram of original mixture",
#         OUT_DIR / "spectrogram_original.png"
#     )

#     plot_single_spectrogram(
#         x_enh,
#         sr_enh,
#         "Spectrogram of enhanced signal",
#         OUT_DIR / "spectrogram_enhanced.png"
#     )

#     plot_comparison_spectrograms(
#         x_main,
#         sr_main,
#         x_enh,
#         sr_enh,
#         OUT_DIR / "spectrogram_comparison.png"
#     )

#     print("Saved:")
#     print(OUT_DIR / "spectrogram_original.png")
#     print(OUT_DIR / "spectrogram_enhanced.png")
#     print(OUT_DIR / "spectrogram_comparison.png")


# if __name__ == "__main__":
#     main()


from pathlib import Path
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import scipy.signal as sig

MAIN_WAV = "main_scene.wav"
ENHANCED_WAV = "outputs/enhanced_target.wav"
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

START_SEC = 5
END_SEC = 9


def load_audio(path: str, use_front_pair=False):
    x, sr = sf.read(path)
    x = x.astype(np.float32)

    if x.ndim == 2:
        if use_front_pair:
            x = 0.5 * (x[:, 0] + x[:, 2])   # left front + right front
        else:
            x = np.mean(x, axis=1)

    return x, sr


def crop_signal(x, sr, start_sec, end_sec):
    s = int(start_sec * sr)
    e = int(end_sec * sr)
    return x[s:e]


def compute_spectrogram_db(x, sr):
    f, t, Sxx = sig.spectrogram(
        x,
        fs=sr,
        window="hann",
        nperseg=1024,
        noverlap=768,
        scaling="spectrum",
        mode="magnitude"
    )
    Sxx_db = 20 * np.log10(Sxx + 1e-10)
    return f, t, Sxx_db


def plot_three_way_spectrogram(original, enhanced, sr, outpath):
    f1, t1, S1_db = compute_spectrogram_db(original, sr)
    f2, t2, S2_db = compute_spectrogram_db(enhanced, sr)

    diff_db = S2_db - S1_db

    vmin = min(S1_db.min(), S2_db.min())
    vmax = max(S1_db.max(), S2_db.max())

    plt.figure(figsize=(10, 10))

    plt.subplot(3, 1, 1)
    plt.pcolormesh(t1, f1, S1_db, shading="gouraud", vmin=vmin, vmax=vmax)
    plt.title("Original signal")
    plt.ylabel("Frequency [Hz]")
    plt.ylim(0, 5000)
    plt.colorbar(label="Magnitude [dB]")

    plt.subplot(3, 1, 2)
    plt.pcolormesh(t2, f2, S2_db, shading="gouraud", vmin=vmin, vmax=vmax)
    plt.title("Enhanced signal")
    plt.ylabel("Frequency [Hz]")
    plt.ylim(0, 5000)
    plt.colorbar(label="Magnitude [dB]")

    plt.subplot(3, 1, 3)
    plt.pcolormesh(t2, f2, diff_db, shading="gouraud")
    plt.title("Difference spectrogram (enhanced - original)")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [s]")
    plt.ylim(0, 5000)
    plt.colorbar(label="Difference [dB]")

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def rms_envelope(x, sr, win_sec=0.05, hop_sec=0.01):
    win = int(win_sec * sr)
    hop = int(hop_sec * sr)
    values = []
    times = []
    for start in range(0, len(x) - win + 1, hop):
        frame = x[start:start+win]
        values.append(np.sqrt(np.mean(frame**2) + 1e-12))
        times.append((start + win/2) / sr)
    return np.array(times), np.array(values)


def plot_rms_comparison(original, enhanced, sr, outpath):
    t1, e1 = rms_envelope(original, sr)
    t2, e2 = rms_envelope(enhanced, sr)

    plt.figure(figsize=(10, 4))
    plt.plot(t1, e1, label="Original")
    plt.plot(t2, e2, label="Enhanced")
    plt.xlabel("Time [s]")
    plt.ylabel("RMS energy")
    plt.title("Energy envelope comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    x_main, sr_main = load_audio(MAIN_WAV, use_front_pair=True)
    x_enh, sr_enh = load_audio(ENHANCED_WAV, use_front_pair=False)

    if sr_main != sr_enh:
        raise ValueError("Sample rates do not match.")

    x_main = crop_signal(x_main, sr_main, START_SEC, END_SEC)
    x_enh = crop_signal(x_enh, sr_enh, START_SEC, END_SEC)

    plot_three_way_spectrogram(
        x_main, x_enh, sr_main,
        OUT_DIR / "spectrogram_three_way.png"
    )

    plot_rms_comparison(
        x_main, x_enh, sr_main,
        OUT_DIR / "rms_comparison.png"
    )

    print("Saved:")
    print(OUT_DIR / "spectrogram_three_way.png")
    print(OUT_DIR / "rms_comparison.png")


if __name__ == "__main__":
    main()