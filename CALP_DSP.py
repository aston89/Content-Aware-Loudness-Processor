import argparse
from collections import deque

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, resample_poly
import pyloudnorm as pyln

# ============================================================
# CONFIG
# ============================================================

TARGET_LUFS_DEFAULT = -14.0

# analysis
K_WEIGHT_HP_HZ = 38.0
K_WEIGHT_SHELF_HZ = 4000.0
K_WEIGHT_SHELF_GAIN_DB = 4.0
K_WEIGHT_SHELF_SLOPE = 0.7

BAND_LOW_CUTOFF_HZ = 200.0
BAND_HIGH_CUTOFF_HZ = 4000.0

ANALYSIS_WINDOW_SEC = 3.0
ANALYSIS_HOP_SEC = 0.5

# control curve
GATE_CENTER_DB = -50.0
GATE_SLOPE = 0.30
CREST_CENTER_DB = 10.0
CREST_SLOPE = 0.75
TRANSIENT_STRENGTH = 7.0
MAX_GAIN_DB = 12.0

BAND_BIAS = {
    "low": 0.85,
    "mid": 1.00,
    "high": 0.92,
}

BAND_SHIFT_DB = {
    "low": -1.0,
    "mid": 0.0,
    "high": 0.8,
}

BAND_TRANSIENT_SCALE = {
    "low": 0.65,
    "mid": 1.0,
    "high": 0.9,
}

# smoothing
GLOBAL_ATTACK = 0.10
GLOBAL_RELEASE = 0.03
BAND_ATTACK = {
    "low": 0.08,
    "mid": 0.10,
    "high": 0.08,
}
BAND_RELEASE = {
    "low": 0.03,
    "mid": 0.03,
    "high": 0.02,
}

# limiter
LIMITER_CEILING = 0.98
LIMITER_LOOKAHEAD_MS = 6.0
LIMITER_OVERSAMPLE = 4
LIMITER_ATTACK = 0.25
LIMITER_RELEASE = 0.004

# export
OUTPUT_SUBTYPE = "PCM_16"
OUTPUT_FORMAT = "WAV"
DITHER_ENABLED = True

# convergence
FINAL_LUFS_TOLERANCE_DB = 0.15
FINAL_LUFS_ITERATIONS = 2
MAX_INPUT_NORMALIZATION = 1.5

EPS = 1e-12


# ============================================================
# UTILS
# ============================================================

def db(x):
    return 20.0 * np.log10(np.maximum(x, EPS))


def lin(x_db):
    return 10.0 ** (x_db / 20.0)


def sigmoid(x, center, slope):
    return 1.0 / (1.0 + np.exp(-(x - center) * slope))


def attack_release_smooth(desired, attack=0.08, release=0.015):
    desired = np.asarray(desired, dtype=np.float64)
    if len(desired) == 0:
        return desired

    out = np.empty_like(desired)
    out[0] = desired[0]
    for i in range(1, len(desired)):
        coef = attack if desired[i] < out[i - 1] else release
        out[i] = coef * desired[i] + (1.0 - coef) * out[i - 1]
    return out


def audio_safety(x):
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, -1.0, 1.0)
    return np.ascontiguousarray(x.astype(np.float32))


# ============================================================
# FILTERS
# ============================================================

def biquad_sos(b0, b1, b2, a0, a1, a2):
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)


def high_shelf_sos(fs, f0=4000.0, gain_db=4.0, slope=0.7):
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / fs
    cosw0 = np.cos(w0)
    sinw0 = np.sin(w0)

    alpha = sinw0 / 2.0 * np.sqrt((A + 1.0 / A) * (1.0 / slope - 1.0) + 2.0)
    beta = 2.0 * np.sqrt(A) * alpha

    b0 = A * ((A + 1.0) + (A - 1.0) * cosw0 + beta)
    b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cosw0)
    b2 = A * ((A + 1.0) + (A - 1.0) * cosw0 - beta)
    a0 = (A + 1.0) - (A - 1.0) * cosw0 + beta
    a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cosw0)
    a2 = (A + 1.0) - (A - 1.0) * cosw0 - beta

    return biquad_sos(b0, b1, b2, a0, a1, a2)


def k_weighting_sos(fs):
    hp = butter(2, K_WEIGHT_HP_HZ / (fs / 2.0), btype="highpass", output="sos")
    shelf = high_shelf_sos(fs, f0=K_WEIGHT_SHELF_HZ, gain_db=K_WEIGHT_SHELF_GAIN_DB, slope=K_WEIGHT_SHELF_SLOPE)
    return np.vstack((hp, shelf))


def design_band_sos(fs):
    low = butter(4, BAND_LOW_CUTOFF_HZ / (fs / 2.0), btype="lowpass", output="sos")
    mid = butter(4, [BAND_LOW_CUTOFF_HZ / (fs / 2.0), BAND_HIGH_CUTOFF_HZ / (fs / 2.0)], btype="bandpass", output="sos")
    high = butter(4, BAND_HIGH_CUTOFF_HZ / (fs / 2.0), btype="highpass", output="sos")
    return low, mid, high


# ============================================================
# ANALYSIS
# ============================================================

def measure_lufs(x, sr):
    meter = pyln.Meter(sr)
    return meter.integrated_loudness(x.astype(np.float32))


def band_features(block):
    w = np.abs(block)
    rms = np.sqrt(np.mean(np.square(block)) + EPS)
    peak = np.max(w) + EPS
    crest = db(peak / rms)
    level = db(rms)

    d = np.diff(w)
    trans = np.mean(np.maximum(d, 0.0)) if len(d) else 0.0
    return level, crest, trans


def collect_frame_metrics(mono_w, low_sig, mid_sig, high_sig, sr):
    n = len(mono_w)
    if n == 0:
        raise ValueError("Audio vuoto.")

    window = min(int(ANALYSIS_WINDOW_SEC * sr), n)
    hop = max(1, int(ANALYSIS_HOP_SEC * sr))

    frame_metrics = []
    band_metrics = {"low": [], "mid": [], "high": []}

    if n <= window:
        seg = slice(0, n)
        frame_metrics.append(band_features(mono_w[seg]))
        band_metrics["low"].append(band_features(low_sig[seg]))
        band_metrics["mid"].append(band_features(mid_sig[seg]))
        band_metrics["high"].append(band_features(high_sig[seg]))
        return frame_metrics, band_metrics, window, hop

    for i in range(0, n - window + 1, hop):
        seg = slice(i, i + window)
        frame_metrics.append(band_features(mono_w[seg]))
        band_metrics["low"].append(band_features(low_sig[seg]))
        band_metrics["mid"].append(band_features(mid_sig[seg]))
        band_metrics["high"].append(band_features(high_sig[seg]))

    if not frame_metrics:
        seg = slice(0, n)
        frame_metrics.append(band_features(mono_w[seg]))
        band_metrics["low"].append(band_features(low_sig[seg]))
        band_metrics["mid"].append(band_features(mid_sig[seg]))
        band_metrics["high"].append(band_features(high_sig[seg]))

    return frame_metrics, band_metrics, window, hop


# ============================================================
# CONTROL
# ============================================================

def build_gain_curves(analysis, target_lufs):
    frame_metrics = analysis["frame_metrics"]
    band_metrics = analysis["band_metrics"]

    base_gain_db = target_lufs - analysis["lufs"]

    global_frames = []
    band_frames = {"low": [], "mid": [], "high": []}

    for level, crest, trans in frame_metrics:
        gate = sigmoid(level, GATE_CENTER_DB, GATE_SLOPE)
        crest_f = 1.0 - sigmoid(crest, CREST_CENTER_DB, CREST_SLOPE)
        trans_f = np.exp(-trans * TRANSIENT_STRENGTH)

        g = base_gain_db * gate * crest_f * trans_f
        global_frames.append(np.clip(g, -MAX_GAIN_DB, MAX_GAIN_DB))

    for name in ("low", "mid", "high"):
        for level, crest, trans in band_metrics[name]:
            shift = BAND_SHIFT_DB[name]
            trans_scale = BAND_TRANSIENT_SCALE[name]

            gate = sigmoid(level, GATE_CENTER_DB + shift, GATE_SLOPE)
            crest_f = 1.0 - sigmoid(crest, CREST_CENTER_DB + shift, CREST_SLOPE)
            trans_f = np.exp(-trans * TRANSIENT_STRENGTH * trans_scale)

            g = base_gain_db * gate * crest_f * trans_f * BAND_BIAS[name]
            band_frames[name].append(np.clip(g, -MAX_GAIN_DB, MAX_GAIN_DB))

    global_frames = attack_release_smooth(global_frames, attack=GLOBAL_ATTACK, release=GLOBAL_RELEASE)

    for k in band_frames:
        band_frames[k] = attack_release_smooth(
            band_frames[k],
            attack=BAND_ATTACK[k],
            release=BAND_RELEASE[k]
        )

    return global_frames, band_frames


def sample_hold(frames, n_samples, hop, frame_len):
    frames = np.asarray(frames, dtype=np.float64)
    if len(frames) == 0:
        return np.zeros(n_samples, dtype=np.float64)
    if len(frames) == 1:
        return np.full(n_samples, frames[0], dtype=np.float64)

    centers = np.arange(len(frames), dtype=np.float64) * hop + (frame_len / 2.0)
    xs = np.arange(n_samples, dtype=np.float64)
    return np.interp(xs, centers, frames, left=frames[0], right=frames[-1])


# ============================================================
# LIMITER
# ============================================================

def sliding_forward_max(x, size):
    """
    For each sample i, returns max(x[i : i+size]).
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    out = np.empty(n, dtype=np.float64)
    dq = deque()

    for i in range(min(size, n)):
        val = x[i]
        while dq and dq[-1][1] <= val:
            dq.pop()
        dq.append((i, val))

    for i in range(n):
        while dq and dq[0][0] < i:
            dq.popleft()

        j = i + size - 1
        if j < n:
            val = x[j]
            while dq and dq[-1][1] <= val:
                dq.pop()
            dq.append((j, val))

        out[i] = dq[0][1] if dq else x[i]

    return out


def true_peak_limiter(x, sr, ceiling=LIMITER_CEILING, lookahead_ms=LIMITER_LOOKAHEAD_MS, up=LIMITER_OVERSAMPLE):
    mono = np.mean(x, axis=1)

    mono_os = resample_poly(mono, up, 1)
    abs_os = np.abs(mono_os)

    lookahead_os = max(1, int(sr * lookahead_ms / 1000.0 * up))
    future_peak_os = sliding_forward_max(abs_os, lookahead_os)

    n = len(mono)
    if len(future_peak_os) < n * up:
        pad = n * up - len(future_peak_os)
        future_peak_os = np.pad(future_peak_os, (0, pad), mode="edge")

    future_peak_base = future_peak_os[: n * up].reshape(n, up).max(axis=1)

    gain = np.minimum(1.0, ceiling / (future_peak_base + EPS))
    gain = attack_release_smooth(gain, attack=LIMITER_ATTACK, release=LIMITER_RELEASE)

    return x * gain[:, None]


# ============================================================
# DITHER / EXPORT
# ============================================================

def tpdf_dither(x):
    amp = 1.0 / 32768.0
    n = np.random.uniform(-amp, amp, x.shape)
    n += np.random.uniform(-amp, amp, x.shape)
    return x + n


def export_pcm16_wav(x, sr, path):
    x = audio_safety(x)
    if DITHER_ENABLED:
        x = tpdf_dither(x)
    x = audio_safety(x)
    sf.write(path, x, sr, subtype=OUTPUT_SUBTYPE, format=OUTPUT_FORMAT)


# ============================================================
# PROCESS
# ============================================================

def process(input_path, output_path, target_lufs=TARGET_LUFS_DEFAULT):
    audio, sr = sf.read(input_path, always_2d=True)
    audio = audio.astype(np.float64)

    if audio.size == 0:
        raise ValueError("File audio vuoto.")

    mx = np.max(np.abs(audio))
    if mx > MAX_INPUT_NORMALIZATION:
        audio = audio / mx

    mono = np.mean(audio, axis=1)

    # ANALYSIS
    lufs_in = measure_lufs(mono, sr)

    low_f, mid_f, high_f = design_band_sos(sr)
    kw = k_weighting_sos(sr)
    mono_w = sosfilt(kw, mono)

    low_sig = sosfilt(low_f, mono)
    mid_sig = sosfilt(mid_f, mono)
    high_sig = sosfilt(high_f, mono)

    frame_metrics, band_metrics, frame_len, hop = collect_frame_metrics(
        mono_w, low_sig, mid_sig, high_sig, sr
    )

    analysis = {
        "lufs": lufs_in,
        "frame_metrics": frame_metrics,
        "band_metrics": band_metrics,
    }

    # CONTROL
    global_frames, band_frames = build_gain_curves(analysis, target_lufs)

    global_gain = lin(sample_hold(global_frames, len(audio), hop, frame_len))
    low_gain = lin(sample_hold(band_frames["low"], len(audio), hop, frame_len))
    mid_gain = lin(sample_hold(band_frames["mid"], len(audio), hop, frame_len))
    high_gain = lin(sample_hold(band_frames["high"], len(audio), hop, frame_len))

    # RENDER
    out = np.zeros_like(audio, dtype=np.float64)

    for ch in range(audio.shape[1]):
        x = audio[:, ch]

        l = sosfilt(low_f, x) * low_gain
        m = sosfilt(mid_f, x) * mid_gain
        h = sosfilt(high_f, x) * high_gain

        out[:, ch] = (l + m + h) * global_gain

    # LIMITER
    out = true_peak_limiter(out, sr)

    # FINAL LUFS CONVERGENCE
    for _ in range(FINAL_LUFS_ITERATIONS):
        out_mono = np.mean(out, axis=1)
        lufs_out = measure_lufs(out_mono, sr)
        delta_db = target_lufs - lufs_out

        if abs(delta_db) <= FINAL_LUFS_TOLERANCE_DB:
            break

        out *= lin(delta_db)
        out = true_peak_limiter(out, sr)

    # SAFETY + EXPORT
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.clip(out, -1.0, 1.0)
    export_pcm16_wav(out, sr, output_path)

    peak = np.max(np.abs(out))
    lufs_final = measure_lufs(np.mean(out, axis=1), sr)

    print(f"input LUFS : {lufs_in:.2f}")
    print(f"final LUFS : {lufs_final:.2f}")
    print(f"final peak : {peak:.6f}")
    print("done")


# ============================================================
# MAIN
# ============================================================

def main():
    p = argparse.ArgumentParser(description="Curve-based loudness shaper / broadcast pre-processor.")
    p.add_argument("input", help="Input audio file")
    p.add_argument("output", help="Output audio file")
    p.add_argument("--target-lufs", type=float, default=TARGET_LUFS_DEFAULT, help="Target loudness level")
    args = p.parse_args()

    process(args.input, args.output, target_lufs=args.target_lufs)


if __name__ == "__main__":
    main()
