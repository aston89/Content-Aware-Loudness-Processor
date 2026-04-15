import argparse
from collections import deque

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

EPS = 1e-12


def db(x):
    return 20.0 * np.log10(np.maximum(x, EPS))


def lin(x_db):
    return 10.0 ** (x_db / 20.0)


def sigmoid(x, center, slope):
    return 1.0 / (1.0 + np.exp(-(x - center) * slope))


def biquad_sos(b0, b1, b2, a0, a1, a2):
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)


def high_shelf_sos(fs, f0=4000.0, gain_db=4.0, slope=0.7):
    # RBJ Audio EQ Cookbook
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
    hp = butter(2, 38.0 / (fs / 2.0), btype="highpass", output="sos")
    shelf = high_shelf_sos(fs, f0=4000.0, gain_db=4.0, slope=0.7)
    return np.vstack((hp, shelf))


def design_band_sos(fs):
    low = butter(4, 200.0 / (fs / 2.0), btype="lowpass", output="sos")
    mid = butter(4, [200.0 / (fs / 2.0), 4000.0 / (fs / 2.0)], btype="bandpass", output="sos")
    high = butter(4, 4000.0 / (fs / 2.0), btype="highpass", output="sos")
    return low, mid, high


def attack_release_smooth(desired, attack=0.08, release=0.015):
    desired = np.asarray(desired, dtype=np.float64)
    out = np.empty_like(desired)
    out[0] = desired[0]
    for i in range(1, len(desired)):
        coef = attack if desired[i] < out[i - 1] else release
        out[i] = coef * desired[i] + (1.0 - coef) * out[i - 1]
    return out


def sample_holds(frame_vals, n_samples, hop, frame_len):
    frame_vals = np.asarray(frame_vals, dtype=np.float64)
    if len(frame_vals) == 1:
        return np.full(n_samples, frame_vals[0], dtype=np.float64)

    centers = np.arange(len(frame_vals), dtype=np.float64) * hop + (frame_len / 2.0)
    xs = np.arange(n_samples, dtype=np.float64)
    return np.interp(xs, centers, frame_vals, left=frame_vals[0], right=frame_vals[-1])


def band_features(block):
    w = np.abs(block)
    rms = np.sqrt(np.mean(np.square(block)) + EPS)
    peak = np.max(w) + EPS
    crest = db(peak / rms)
    level = db(rms)

    d = np.diff(w)
    trans = np.mean(np.maximum(d, 0.0)) if len(d) else 0.0
    return level, crest, trans


def sliding_forward_max(x, size):
    """
    For each sample i, returns max(x[i : i+size]).
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    out = np.empty(n, dtype=np.float64)
    dq = deque()  # (index, value), decreasing by value

    # seed with first window
    for i in range(min(size, n)):
        val = x[i]
        while dq and dq[-1][1] <= val:
            dq.pop()
        dq.append((i, val))

    for i in range(n):
        # drop expired
        while dq and dq[0][0] < i:
            dq.popleft()

        # window end is i + size - 1
        j = i + size - 1
        if j < n:
            val = x[j]
            while dq and dq[-1][1] <= val:
                dq.pop()
            dq.append((j, val))

        out[i] = dq[0][1] if dq else x[i]

    return out


def process(input_path, output_path, target_lufs=-14.0):
    audio, sr = sf.read(input_path, always_2d=True)
    audio = audio.astype(np.float64)

    # normalize only if file is way out of range
    mx = np.max(np.abs(audio))
    if mx > 1.5:
        audio /= mx

    mono = np.mean(audio, axis=1)

    # analysis weighting
    kw = k_weighting_sos(sr)
    mono_w = sosfilt(kw, mono)

    # multiband split for processing
    low_sos, mid_sos, high_sos = design_band_sos(sr)
    low_m = sosfilt(low_sos, mono)
    mid_m = sosfilt(mid_sos, mono)
    high_m = sosfilt(high_sos, mono)

    # rough loudness estimate
    global_lufs = db(np.sqrt(np.mean(np.square(mono_w)) + EPS))
    base_gain_db = target_lufs - global_lufs

    window = int(3.0 * sr)
    hop = max(1, int(0.5 * sr))

    if len(mono) < window:
        raise ValueError("Audio file too short for the chosen analysis window.")

    frame_metrics = []
    band_metrics = {"low": [], "mid": [], "high": []}

    for start in range(0, len(mono) - window + 1, hop):
        end = start + window

        level, crest, trans = band_features(mono_w[start:end])
        frame_metrics.append((level, crest, trans))

        for name, sig in (("low", low_m), ("mid", mid_m), ("high", high_m)):
            blevel, bcrest, btrans = band_features(sig[start:end])
            band_metrics[name].append((blevel, bcrest, btrans))

    # curve controls
    GATE_CENTER = -50.0
    GATE_SLOPE = 0.30
    CREST_CENTER = 10.0
    CREST_SLOPE = 0.75
    TRANSIENT_STRENGTH = 7.0
    MAX_GAIN_DB = 12.0

    band_bias = {"low": 0.85, "mid": 1.00, "high": 0.92}

    global_gain_db_frames = []
    band_gain_db_frames = {"low": [], "mid": [], "high": []}

    for level, crest, trans in frame_metrics:
        gate = sigmoid(level, GATE_CENTER, GATE_SLOPE)
        crest_factor = 1.0 - sigmoid(crest, CREST_CENTER, CREST_SLOPE)
        transient_factor = np.exp(-trans * TRANSIENT_STRENGTH)

        gain_db = base_gain_db * gate * crest_factor * transient_factor
        gain_db = np.clip(gain_db, -MAX_GAIN_DB, MAX_GAIN_DB)
        global_gain_db_frames.append(gain_db)

    for band_name in ("low", "mid", "high"):
        for level, crest, trans in band_metrics[band_name]:
            if band_name == "low":
                center_shift = -1.0
                transient_scale = 0.65
            elif band_name == "mid":
                center_shift = 0.0
                transient_scale = 1.0
            else:
                center_shift = 0.8
                transient_scale = 0.9

            gate = sigmoid(level, GATE_CENTER + center_shift, GATE_SLOPE)
            crest_factor = 1.0 - sigmoid(crest, CREST_CENTER + center_shift, CREST_SLOPE)
            transient_factor = np.exp(-trans * TRANSIENT_STRENGTH * transient_scale)

            gain_db = base_gain_db * gate * crest_factor * transient_factor * band_bias[band_name]
            gain_db = np.clip(gain_db, -MAX_GAIN_DB, MAX_GAIN_DB)
            band_gain_db_frames[band_name].append(gain_db)

    global_gain_db_frames = attack_release_smooth(global_gain_db_frames, attack=0.10, release=0.03)
    for band_name in band_gain_db_frames:
        atk = 0.10 if band_name == "mid" else 0.08
        rel = 0.03 if band_name != "high" else 0.02
        band_gain_db_frames[band_name] = attack_release_smooth(
            band_gain_db_frames[band_name], attack=atk, release=rel
        )

    global_gain = lin(sample_holds(global_gain_db_frames, len(audio), hop, window))
    low_gain = lin(sample_holds(band_gain_db_frames["low"], len(audio), hop, window))
    mid_gain = lin(sample_holds(band_gain_db_frames["mid"], len(audio), hop, window))
    high_gain = lin(sample_holds(band_gain_db_frames["high"], len(audio), hop, window))

    out = np.zeros_like(audio)
    for ch in range(audio.shape[1]):
        x = audio[:, ch]
        low = sosfilt(low_sos, x) * low_gain
        mid = sosfilt(mid_sos, x) * mid_gain
        high = sosfilt(high_sos, x) * high_gain
        out[:, ch] = (low + mid + high) * global_gain

    # limiter
    peak = np.max(np.abs(out), axis=1)
    LOOKAHEAD_MS = 6.0
    lookahead = max(1, int(sr * LOOKAHEAD_MS / 1000.0))
    future_peak = sliding_forward_max(peak, lookahead)

    threshold = 0.98
    desired_gain = np.minimum(1.0, threshold / (future_peak + EPS))
    desired_gain = attack_release_smooth(desired_gain, attack=0.25, release=0.004)

    out *= desired_gain[:, None]

    # final safety
    out = np.tanh(out * 1.05) / np.tanh(1.05)

    peak_out = np.max(np.abs(out))
    if peak_out > 1.0:
        out /= peak_out

    sf.write(output_path, out, sr)
    print(f"input  : {input_path}")
    print(f"output : {output_path}")
    print(f"sr     : {sr}")
    print(f"global : {global_lufs:.2f} LUFS-ish")
    print(f"base gain applied: {base_gain_db:.2f} dB")


def main():
    p = argparse.ArgumentParser(description="Curve-based loudness shaper / ready-to-go processor.")
    p.add_argument("input", help="Input audio file")
    p.add_argument("output", help="Output audio file")
    p.add_argument("--target-lufs", type=float, default=-14.0, help="Target loudness level")
    args = p.parse_args()
    process(args.input, args.output, target_lufs=args.target_lufs)


if __name__ == "__main__":
    main()
