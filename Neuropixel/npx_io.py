"""Loading helpers for the Neuropixel open-field dataset (Kilosort/Phy output + overhead video).

Files expected in DATA_DIR:
    spike_times.npy     int64 (n_spikes,)  sample index of every spike, 30 kHz
    spike_clusters.npy  int32 (n_spikes,)  cluster id of every spike
    cluster_info.tsv    one row per cluster: Kilosort + Bombcell + manual labels
    cluster_rf.tsv      cluster_id -> manual curation label (duplicated in cluster_info.rf_label)
    video_compressed.mkv overhead camera, ~60 fps
"""
from pathlib import Path
import numpy as np
import pandas as pd

FS = 30_000.0  # Neuropixels AP-band sampling rate (Hz)

LABEL_ORDER = ["good", "mua", "nonso", "noise"]


def load_cluster_info(data_dir):
    """cluster_info.tsv as a DataFrame indexed by cluster_id, with a tidy `label` column."""
    ci = pd.read_csv(Path(data_dir) / "cluster_info.tsv", sep="\t")
    ci = ci.set_index("cluster_id").sort_index()
    ci["label"] = pd.Categorical(ci["rf_label"], categories=LABEL_ORDER)
    ci["fr_hz"] = ci["fr"]
    ci["depth_um"] = ci["depth"]
    ci["duration_ms"] = ci["waveformDuration_peakTrough"] / 1000.0  # given in µs
    return ci


def load_spikes(data_dir, fs=FS):
    """Return (spike_times_s, spike_clusters) as float64 seconds and int32 ids."""
    d = Path(data_dir)
    st = np.load(d / "spike_times.npy").astype(np.float64) / fs
    sc = np.load(d / "spike_clusters.npy").astype(np.int32)
    return st, sc


def spikes_by_unit(st, sc, unit_ids):
    """Dict {cluster_id: sorted spike times (s)} for the requested units."""
    order = np.argsort(sc, kind="stable")
    sc_sorted, st_sorted = sc[order], st[order]
    bounds = np.searchsorted(sc_sorted, np.asarray(unit_ids), side="left")
    ends = np.searchsorted(sc_sorted, np.asarray(unit_ids), side="right")
    return {u: st_sorted[a:b] for u, a, b in zip(unit_ids, bounds, ends)}


def bin_spikes(st, sc, unit_ids, bin_s, t_max=None):
    """Binned spike counts, shape (n_units, n_bins), plus bin edges (s)."""
    unit_ids = np.asarray(unit_ids)
    t_max = st.max() if t_max is None else t_max
    edges = np.arange(0, t_max + bin_s, bin_s)
    lut = -np.ones(sc.max() + 1, dtype=np.int64)
    lut[unit_ids] = np.arange(len(unit_ids))
    row = lut[sc]
    keep = (row >= 0) & (st < edges[-1])          # drop spikes beyond the last edge
    col = (st[keep] / bin_s).astype(np.int64)
    counts = np.zeros((len(unit_ids), len(edges) - 1), dtype=np.int32)
    np.add.at(counts, (row[keep], col), 1)
    return counts, edges


def autocorrelogram(t, bin_ms=1.0, window_ms=50.0):
    """Spike-time autocorrelogram (counts per lag bin), lags in ms. Zero-lag bin is removed."""
    t_ms = t * 1000.0
    lags = []
    hi = np.searchsorted(t_ms, t_ms + window_ms, side="right")
    for k, (a, b) in enumerate(zip(np.arange(len(t_ms)) + 1, hi)):
        if b > a:
            lags.append(t_ms[a:b] - t_ms[k])
    lags = np.concatenate(lags) if lags else np.array([])
    edges = np.arange(0, window_ms + bin_ms, bin_ms)
    h, _ = np.histogram(lags, edges)
    full = np.concatenate([h[::-1], h])
    centers = np.concatenate([-edges[1:][::-1] + bin_ms / 2, edges[:-1] + bin_ms / 2])
    return centers, full


def video_metadata(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    meta = dict(
        fps=cap.get(cv2.CAP_PROP_FPS),
        n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    meta["duration_s"] = meta["n_frames"] / meta["fps"]
    cap.release()
    return meta


def read_frame(path, frame_idx, gray=True):
    import cv2
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, fr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {frame_idx}")
    return cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if gray else fr


def coarse_video_scan(path, step_s=10.0, cache=None, downsample=8):
    """Mean brightness and frame-to-frame motion energy every `step_s` seconds.

    Returns array shape (3, n): [time_s, mean_intensity, motion_energy]. Slow (~2 min per
    700 samples on a 2K video) so pass `cache` to save/load a .npy.
    """
    import cv2
    if cache is not None and Path(cache).exists():
        return np.load(cache)
    cap = cv2.VideoCapture(str(path))
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    idxs = np.arange(0, n, int(round(fps * step_s)))
    prev, mean, motion = None, [], []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok:
            mean.append(np.nan); motion.append(np.nan); continue
        g = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (w // downsample, h // downsample)).astype(np.float32)
        mean.append(g.mean())
        motion.append(np.nan if prev is None else np.abs(g - prev).mean())
        prev = g
    cap.release()
    out = np.stack([idxs / fps, np.array(mean), np.array(motion)])
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, out)
    return out


def smooth_rows(x, k):
    """Boxcar-smooth each row of a 2-D array with a window of k bins (same length)."""
    if k <= 1:
        return np.asarray(x, dtype=float)
    ker = np.ones(k) / k
    return np.apply_along_axis(lambda r: np.convolve(r, ker, mode="same"), 1, np.asarray(x, dtype=float))


def zscore_rows(x, eps=1e-9):
    x = np.asarray(x, dtype=float)
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + eps)


def rate_autocorr_timescale(counts, bin_s, max_lag_s=1.0):
    """Intrinsic timescale per unit (Murray et al. 2014 style).

    Pearson autocorrelation of spike counts (bins of `bin_s`, e.g. 50 ms) at lags 1..max_lag,
    then a least-squares fit of A*exp(-lag/tau) + B. The zero-lag bin is excluded so the
    Poisson delta does not dominate. Returns tau (s); NaN where the fit fails or tau hits bounds.
    """
    from scipy.optimize import curve_fit
    n_lag = int(round(max_lag_s / bin_s))
    lags = np.arange(1, n_lag + 1) * bin_s
    out = np.full(counts.shape[0], np.nan)
    f = lambda x, a, tau, b: a * np.exp(-x / tau) + b
    for i, c in enumerate(counts.astype(float)):
        if c.std() == 0:
            continue
        c = c - c.mean()
        n = len(c)
        ac = np.array([np.dot(c[:-k], c[k:]) / (n - k) for k in range(1, n_lag + 1)]) / c.var()
        try:
            p, _ = curve_fit(f, lags, ac, p0=[ac[0], 0.1, ac[-1]],
                             bounds=([0, bin_s / 2, -1], [2, 5 * max_lag_s, 1]), maxfev=2000)
            if bin_s / 2 < p[1] < 5 * max_lag_s * 0.99:
                out[i] = p[1]
        except Exception:
            pass
    return out


def find_epochs(mask, t, min_len_s=0.0):
    """Contiguous True runs of a boolean mask -> list of (t_start, t_end)."""
    mask = np.asarray(mask, bool)
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    dt = t[1] - t[0]
    return [(t[s], t[e - 1] + dt) for s, e in zip(starts, ends) if (e - s) * dt >= min_len_s]


# ---------------------------------------------------------------------------
# Video <-> ephys alignment
# Measured 2026-09-09 by cross-correlating video motion energy with the good-unit population
# rate: the two streams start together but the video timeline (constant 59.637 fps) runs 0.6%
# fast relative to the ephys clock, i.e. the effective frame rate is ~59.28 fps (dropped frames
# or a wrong nominal rate). Linear fit t_ephys = a + (1 + b) * t_video, stored in cache/video_warp.npy.
# ---------------------------------------------------------------------------
def load_video_warp(cache_dir):
    a, b = np.load(Path(cache_dir) / "video_warp.npy")
    return float(a), float(b)


def video_to_ephys_time(t_video_s, warp):
    """Map container video time (frame_idx / nominal fps) to ephys seconds."""
    a, b = warp
    return a + (1.0 + b) * np.asarray(t_video_s, dtype=float)


def frame_to_ephys_time(frame_idx, warp, fps_nominal=59.63700234192037):
    return video_to_ephys_time(np.asarray(frame_idx, dtype=float) / fps_nominal, warp)


def ephys_to_frame(t_ephys_s, warp, fps_nominal=59.63700234192037):
    a, b = warp
    return np.round((np.asarray(t_ephys_s, dtype=float) - a) / (1.0 + b) * fps_nominal).astype(int)


def load_motion_energy(cache_dir, warp=None):
    """Per-frame motion energy (mean |frame diff| on a 268x258 downsample). Returns (t_s, motion).
    If warp is given, times are in ephys seconds; otherwise container video seconds."""
    ts, mot = np.load(Path(cache_dir) / "motion_energy_fullrate.npy")
    ok = ~np.isnan(mot)
    ts, mot = ts[ok], mot[ok]
    if warp is not None:
        ts = video_to_ephys_time(ts, warp)
    return ts, mot


def bin_signal(t, x, edges):
    """Mean of x per bin (NaN where empty)."""
    idx = np.searchsorted(edges, t, side="right") - 1
    ok = (idx >= 0) & (idx < len(edges) - 1)
    n = np.bincount(idx[ok], minlength=len(edges) - 1)
    s = np.bincount(idx[ok], weights=x[ok], minlength=len(edges) - 1)
    with np.errstate(invalid="ignore"):
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)
