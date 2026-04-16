#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from collections import deque
from typing import Dict, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, resample_poly, welch
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
ANALYSIS_UPSAMPLE_SR = 48000
ANALYSIS_HF_SPLIT_HZ = 8000.0
ANALYSIS_HF_THRESHOLD = 0.12

# control curve
GATE_CENTER_DB = -50.0
GATE_SLOPE = 0.30
CREST_CENTER_DB = 10.0
CREST_SLOPE = 0.75
TRANSIENT_STRENGTH = 7.0
MAX_GAIN_DB = 15.0
CLIPPED_INPUT_GAIN_CAP_DB = 12.0

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
LIMITER_SECOND_PASS_CEILING = 0.965
LIMITER_LOOKAHEAD_MS = 6.0
LIMITER_OVERSAMPLE = 4
LIMITER_OVERSAMPLE_AUTOSENSE = 8
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
# CACHE FOR FILTERS (one set per sample-rate)
# ============================================================
_FILTER_CACHE: Dict[int, Dict[str, np.ndarray]] = {}


def get_filters(sr: int) -> Dict[str, np.ndarray]:
    """Return low, mid, high and k-weighting SOS matrices (float32) for *sr*."""
    if sr not in _FILTER_CACHE:
        low_f, mid_f, high_f = design_band_sos(sr)
        kw = k_weighting_sos(sr)

        _FILTER_CACHE[sr] = {
            "low": low_f.astype(np.float32),
            "mid": mid_f.astype(np.float32),
            "high": high_f.astype(np.float32),
            "kw": kw.astype(np.float32),
        }
    return _FILTER_CACHE[sr]


# ============================================================
# UTILITIES
# ============================================================
def db(x: np.ndarray) -> np.ndarray:
    """Convert linear amplitude to dB (avoid log(0))."""
    return 20.0 * np.log10(np.maximum(x, EPS))


def lin(x_db: float) -> float:
    """Convert dB to linear gain."""
    return 10.0 ** (x_db / 20.0)


def sigmoid(x: np.ndarray, center: float, slope: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(x - center) * slope))


def attack_release_smooth(
    desired: np.ndarray,
    attack: float = 0.08,
    release: float = 0.015,
) -> np.ndarray:
    """Simple peak-hold smoothing with different coeff. for attack / release."""
    desired = np.asarray(desired, dtype=np.float64)
    if len(desired) == 0:
        return desired

    out = np.empty_like(desired)
    out[0] = desired[0]
    for i in range(1, len(desired)):
        coef = attack if desired[i] < out[i - 1] else release
        out[i] = coef * desired[i] + (1.0 - coef) * out[i - 1]
    return out


def audio_safety(x: np.ndarray) -> np.ndarray:
    """Clamp, replace NaNs / infinities and force contiguous float32."""
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, -1.0, 1.0)
    return np.ascontiguousarray(x.astype(np.float32))


def detect_clipping(audio: np.ndarray, threshold: float = 0.999) -> Dict[str, float]:
    """Detect hard clipping-like material before analysis."""
    abs_a = np.abs(audio)
    peak = float(np.max(abs_a))
    clipped_samples = int(np.sum(abs_a >= threshold))
    total_samples = int(audio.size)
    clip_ratio = clipped_samples / max(1, total_samples)
    return {
        "peak": peak,
        "clipped_samples": clipped_samples,
        "clip_ratio": clip_ratio,
        "is_clipped": peak >= threshold or clip_ratio > 0.0,
    }


def hf_ratio_estimate(x: np.ndarray, sr: int, split_hz: float = ANALYSIS_HF_SPLIT_HZ) -> float:
    """Estimate how much energy lives above *split_hz*."""
    nperseg = min(4096, len(x)) if len(x) > 0 else 256
    f, pxx = welch(x.astype(np.float64), fs=sr, nperseg=nperseg)
    total = np.sum(pxx) + EPS
    high = np.sum(pxx[f >= split_hz])
    return float(high / total)


def maybe_upsample_for_analysis(
    audio: np.ndarray,
    sr: int,
    target_sr: int = ANALYSIS_UPSAMPLE_SR,
    hf_threshold: float = ANALYSIS_HF_THRESHOLD,
) -> Tuple[np.ndarray, int, bool]:
    """Upsample only for analysis when low SR and bright content make it useful."""
    if sr >= target_sr:
        return audio, sr, False

    mono = np.mean(audio, axis=1)
    hf_ratio = hf_ratio_estimate(mono, sr)

    if hf_ratio < hf_threshold:
        return audio, sr, False

    up = resample_poly(audio, target_sr, sr, axis=0)
    return np.ascontiguousarray(up.astype(np.float32)), target_sr, True


def choose_limiter_oversample(x: np.ndarray, sr: int) -> int:
    """Autosense oversampling for the true-peak limiter."""
    peak = float(np.max(np.abs(x)))
    mono = np.mean(x, axis=1)
    rms = float(np.sqrt(np.mean(np.square(mono)) + EPS))
    crest = float(db(np.array([(peak + EPS) / (rms + EPS)], dtype=np.float64))[0])

    if peak > 0.985 or crest > 14.0:
        return LIMITER_OVERSAMPLE_AUTOSENSE
    if sr < 48000 and peak > 0.95:
        return LIMITER_OVERSAMPLE_AUTOSENSE

    return LIMITER_OVERSAMPLE


# ============================================================
# FILTER DESIGN
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
    shelf = high_shelf_sos(
        fs,
        f0=K_WEIGHT_SHELF_HZ,
        gain_db=K_WEIGHT_SHELF_GAIN_DB,
        slope=K_WEIGHT_SHELF_SLOPE,
    )
    return np.vstack((hp, shelf))


def design_band_sos(fs):
    low = butter(4, BAND_LOW_CUTOFF_HZ / (fs / 2.0), btype="lowpass", output="sos")
    mid = butter(
        4,
        [BAND_LOW_CUTOFF_HZ / (fs / 2.0), BAND_HIGH_CUTOFF_HZ / (fs / 2.0)],
        btype="bandpass",
        output="sos",
    )
    high = butter(4, BAND_HIGH_CUTOFF_HZ / (fs / 2.0), btype="highpass", output="sos")
    return low, mid, high


# ============================================================
# ANALYSIS
# ============================================================
def measure_lufs(x: np.ndarray, sr: int) -> float:
    meter = pyln.Meter(sr)
    return meter.integrated_loudness(x.astype(np.float32))


def band_features(block: np.ndarray):
    w = np.abs(block)
    rms = np.sqrt(np.mean(np.square(block)) + EPS)
    peak = np.max(w) + EPS
    crest = db(np.array([peak / rms], dtype=np.float64))[0]
    level = db(np.array([rms], dtype=np.float64))[0]

    d = np.diff(w)
    trans = np.mean(np.maximum(d, 0.0)) if len(d) else 0.0
    return float(level), float(crest), float(trans)


def collect_frame_metrics(
    mono_w: np.ndarray, low_sig: np.ndarray, mid_sig: np.ndarray, high_sig: np.ndarray, sr: int
):
    n = len(mono_w)
    if n == 0:
        raise ValueError("Audio vuoto.")

    window = min(int(ANALYSIS_WINDOW_SEC * sr), n)
    hop = max(1, int(ANALYSIS_HOP_SEC * sr))

    frame_times_sec = []
    frame_metrics = []
    band_metrics = {"low": [], "mid": [], "high": []}

    if n <= window:
        seg = slice(0, n)
        center_sec = (n / 2.0) / sr
        frame_times_sec.append(center_sec)
        frame_metrics.append(band_features(mono_w[seg]))
        band_metrics["low"].append(band_features(low_sig[seg]))
        band_metrics["mid"].append(band_features(mid_sig[seg]))
        band_metrics["high"].append(band_features(high_sig[seg]))
        return frame_times_sec, frame_metrics, band_metrics

    for i in range(0, n - window + 1, hop):
        seg = slice(i, i + window)
        frame_times_sec.append((i + window / 2.0) / sr)
        frame_metrics.append(band_features(mono_w[seg]))
        band_metrics["low"].append(band_features(low_sig[seg]))
        band_metrics["mid"].append(band_features(mid_sig[seg]))
        band_metrics["high"].append(band_features(high_sig[seg]))

    if not frame_metrics:
        seg = slice(0, n)
        center_sec = (n / 2.0) / sr
        frame_times_sec.append(center_sec)
        frame_metrics.append(band_features(mono_w[seg]))
        band_metrics["low"].append(band_features(low_sig[seg]))
        band_metrics["mid"].append(band_features(mid_sig[seg]))
        band_metrics["high"].append(band_features(high_sig[seg]))

    return frame_times_sec, frame_metrics, band_metrics


# ============================================================
# CONTROL
# ============================================================
def build_gain_curves(analysis: dict, target_lufs: float, max_gain_db: float):
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
        global_frames.append(np.clip(g, -max_gain_db, max_gain_db))

    for name in ("low", "mid", "high"):
        for level, crest, trans in band_metrics[name]:
            shift = BAND_SHIFT_DB[name]
            trans_scale = BAND_TRANSIENT_SCALE[name]

            gate = sigmoid(level, GATE_CENTER_DB + shift, GATE_SLOPE)
            crest_f = 1.0 - sigmoid(crest, CREST_CENTER_DB + shift, CREST_SLOPE)
            trans_f = np.exp(-trans * TRANSIENT_STRENGTH * trans_scale)

            g = base_gain_db * gate * crest_f * trans_f * BAND_BIAS[name]
            band_frames[name].append(np.clip(g, -max_gain_db, max_gain_db))

    global_frames = attack_release_smooth(global_frames, attack=GLOBAL_ATTACK, release=GLOBAL_RELEASE)

    for k in band_frames:
        band_frames[k] = attack_release_smooth(
            band_frames[k],
            attack=BAND_ATTACK[k],
            release=BAND_RELEASE[k],
        )

    return global_frames, band_frames


def sample_hold_from_times(frame_times_sec, frames, n_samples, sr):
    frames = np.asarray(frames, dtype=np.float64)
    frame_times_sec = np.asarray(frame_times_sec, dtype=np.float64)

    if len(frames) == 0:
        return np.zeros(n_samples, dtype=np.float64)
    if len(frames) == 1:
        return np.full(n_samples, frames[0], dtype=np.float64)

    xs = np.arange(n_samples, dtype=np.float64) / float(sr)
    return np.interp(xs, frame_times_sec, frames, left=frames[0], right=frames[-1])


# ============================================================
# LIMITER
# ============================================================
def sliding_forward_max(x, size):
    """For each sample i, returns max(x[i : i+size])."""
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


def _pad_or_trim_os(x_os: np.ndarray, target_len: int) -> np.ndarray:
    """Make oversampled signals match an exact target length."""
    n = len(x_os)
    if n == target_len:
        return x_os
    if n < target_len:
        return np.pad(x_os, (0, target_len - n), mode="edge")
    return x_os[:target_len]


def _future_peak_gain_from_signal(signal_os: np.ndarray, n: int, up: int, ceiling: float, lookahead_ms: float, sr: int):
    """Build a per-sample gain envelope from an oversampled peak trace."""
    abs_os = np.abs(signal_os)
    lookahead_os = max(1, int(sr * lookahead_ms / 1000.0 * up))
    future_peak_os = sliding_forward_max(abs_os, lookahead_os)
    future_peak_os = _pad_or_trim_os(future_peak_os, n * up)
    future_peak_base = future_peak_os.reshape(n, up).max(axis=1)

    gain = np.minimum(1.0, ceiling / (future_peak_base + EPS))
    gain = attack_release_smooth(gain, attack=LIMITER_ATTACK, release=LIMITER_RELEASE)
    return gain


def true_peak_limiter(
    x,
    sr,
    ceiling=LIMITER_CEILING,
    lookahead_ms=LIMITER_LOOKAHEAD_MS,
    up=None,
):
    """True-peak limiter with stereo mid-side peak detection when possible."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("Expected a 2D array shaped (samples, channels).")

    n, ch = x.shape
    if n == 0:
        return x

    if up is None:
        up = choose_limiter_oversample(x, sr)

    if ch == 1:
        mono = x[:, 0]
        mono_os = resample_poly(mono, up, 1)
        mono_os = _pad_or_trim_os(mono_os, n * up)
        gain = _future_peak_gain_from_signal(mono_os, n, up, ceiling, lookahead_ms, sr)
        return x * gain[:, None]

    if ch == 2:
        mid = 0.5 * (x[:, 0] + x[:, 1])
        side = 0.5 * (x[:, 0] - x[:, 1])

        mid_os = resample_poly(mid, up, 1)
        side_os = resample_poly(side, up, 1)
        mid_os = _pad_or_trim_os(mid_os, n * up)
        side_os = _pad_or_trim_os(side_os, n * up)

        left_os = mid_os + side_os
        right_os = mid_os - side_os
        peak_os = np.maximum(np.abs(left_os), np.abs(right_os))

        gain = _future_peak_gain_from_signal(peak_os, n, up, ceiling, lookahead_ms, sr)
        return x * gain[:, None]

    # Multichannel fallback: protect the loudest channel at any moment.
    os_channels = []
    for ch_idx in range(ch):
        ch_os = resample_poly(x[:, ch_idx], up, 1)
        os_channels.append(_pad_or_trim_os(ch_os, n * up))

    peak_os = np.max(np.abs(np.stack(os_channels, axis=1)), axis=1)
    gain = _future_peak_gain_from_signal(peak_os, n, up, ceiling, lookahead_ms, sr)
    return x * gain[:, None]


def final_lufs_convergence(out: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    # Double precision for the final trim.
    out = out.astype(np.float64)

    for _ in range(FINAL_LUFS_ITERATIONS):
        lufs_out = measure_lufs(np.mean(out, axis=1), sr)
        delta_db = target_lufs - lufs_out

        if abs(delta_db) <= FINAL_LUFS_TOLERANCE_DB:
            break

        out *= lin(delta_db)
        out = true_peak_limiter(
            out,
            sr,
            ceiling=LIMITER_SECOND_PASS_CEILING,
            up=choose_limiter_oversample(out, sr),
        )

    return out.astype(np.float32)


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
# MAIN PROCESSING PIPELINE
# ============================================================
def process(input_path: str, output_path: str, target_lufs: float = TARGET_LUFS_DEFAULT, verbose: bool = False) -> None:
    # ------------------- load -------------------------------------------------
    audio, sr = sf.read(input_path, always_2d=True)
    audio = audio.astype(np.float32)

    if audio.size == 0:
        raise ValueError("File audio vuoto.")

    mx = np.max(np.abs(audio))
    if mx > MAX_INPUT_NORMALIZATION:
        audio = audio / mx

    clip_info = detect_clipping(audio)
    if verbose and clip_info["is_clipped"]:
        print(
            f"warning: pre-analysis clipping detected "
            f"(peak={clip_info['peak']:.4f}, ratio={clip_info['clip_ratio']:.6f})"
        )

    effective_max_gain_db = MAX_GAIN_DB
    if clip_info["is_clipped"]:
        effective_max_gain_db = min(effective_max_gain_db, CLIPPED_INPUT_GAIN_CAP_DB)

    # Upsample only for analysis when the source is low SR and bright.
    analysis_audio, analysis_sr, upsampled_for_analysis = maybe_upsample_for_analysis(audio, sr)
    if verbose and upsampled_for_analysis:
        print(f"analysis upsampled to {analysis_sr} Hz for more stable HF / K-weighting")

    mono_analysis = np.mean(analysis_audio, axis=1)

    # ------------------- analysis ---------------------------------------------
    lufs_in = measure_lufs(mono_analysis, analysis_sr)

    filters_analysis = get_filters(analysis_sr)
    low_f_a, mid_f_a, high_f_a, kw_a = (
        filters_analysis["low"], filters_analysis["mid"], filters_analysis["high"], filters_analysis["kw"]
    )

    mono_w = sosfilt(kw_a, mono_analysis)
    low_sig = sosfilt(low_f_a, mono_analysis)
    mid_sig = sosfilt(mid_f_a, mono_analysis)
    high_sig = sosfilt(high_f_a, mono_analysis)

    frame_times_sec, frame_metrics, band_metrics = collect_frame_metrics(
        mono_w, low_sig, mid_sig, high_sig, analysis_sr
    )

    analysis = {
        "lufs": lufs_in,
        "frame_times_sec": frame_times_sec,
        "frame_metrics": frame_metrics,
        "band_metrics": band_metrics,
    }

    # ------------------- control ----------------------------------------------
    global_frames, band_frames = build_gain_curves(
        analysis,
        target_lufs,
        max_gain_db=effective_max_gain_db,
    )

    global_gain = lin(sample_hold_from_times(frame_times_sec, global_frames, len(audio), sr)).astype(np.float32)
    low_gain = lin(sample_hold_from_times(frame_times_sec, band_frames["low"], len(audio), sr)).astype(np.float32)
    mid_gain = lin(sample_hold_from_times(frame_times_sec, band_frames["mid"], len(audio), sr)).astype(np.float32)
    high_gain = lin(sample_hold_from_times(frame_times_sec, band_frames["high"], len(audio), sr)).astype(np.float32)

    # ------------------- render -----------------------------------------------
    filters_render = get_filters(sr)
    low_f, mid_f, high_f = filters_render["low"], filters_render["mid"], filters_render["high"]

    out = np.zeros_like(audio, dtype=np.float32)

    for ch in range(audio.shape[1]):
        x = audio[:, ch]

        l = sosfilt(low_f, x) * low_gain
        m = sosfilt(mid_f, x) * mid_gain
        h = sosfilt(high_f, x) * high_gain

        out[:, ch] = (l + m + h) * global_gain

    # ------------------- limiter ----------------------------------------------
    out = true_peak_limiter(out, sr, ceiling=LIMITER_CEILING, up=choose_limiter_oversample(out, sr))

    # ------------------- final LUFS convergence --------------------------------
    out = final_lufs_convergence(out, sr, target_lufs)

    # ------------------- safety & export ---------------------------------------
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.clip(out, -1.0, 1.0)

    export_pcm16_wav(out, sr, output_path)

    if verbose:
        peak = np.max(np.abs(out))
        lufs_final = measure_lufs(np.mean(out, axis=1), sr)
        print(f"input LUFS  : {lufs_in:.2f}")
        print(f"final LUFS  : {lufs_final:.2f}")
        print(f"final peak  : {peak:.6f}")
        print("done")


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Curve-based loudness shaper / broadcast pre-processor."
    )
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("output", help="Output audio file")
    parser.add_argument(
        "--target-lufs",
        type=float,
        default=TARGET_LUFS_DEFAULT,
        help="Target loudness level (LUFS)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print processing details (input/final LUFS, peak, ...)",
    )
    args = parser.parse_args()

    try:
        process(
            args.input,
            args.output,
            target_lufs=args.target_lufs,
            verbose=args.verbose,
        )
    except Exception as e:
        sys.exit(f"Errore durante l'elaborazione: {e}")


if __name__ == "__main__":
    main()
