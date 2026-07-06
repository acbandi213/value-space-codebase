"""
pupil_pipeline.py
=================
DeepLabCut-based pupillometry for the GRIN/2P rig.

Why DLC: on this rig the pupil is NOT reliably the darkest region — when the
visual-stimulus screen is bright the pupil appears as a *light* disc inside a
darker iris ring, so the pupil/background contrast inverts between frames.
Intensity thresholding (and FaceMap's saturation algorithm) fail on that. A
DeepLabCut model trained on pupil-edge keypoints tracks the boundary regardless
of whether the pupil is lighter or darker than its surround.

Workflow
--------
1. `extract_labeling_frames`  -> diverse frames (bright + dim) for labeling
2. DLC (on a GPU machine): create project, label the pupil-edge points, train,
   analyze the videos  -> per-frame keypoints (x, y, likelihood)
3. `measure`  -> robust ellipse fit to the pupil-edge keypoints per frame
                 -> area, diameter, center (pixels) + normalized traces
                 -> blink / low-confidence flags + QC plots

Steps 1 and 3 run anywhere (only numpy/scipy/pandas/opencv). Step 2 calls the
`deeplabcut` package and needs a GPU; the thin wrappers here just forward to it.

Default keypoint layout (8 points clockwise around the pupil edge):
    pupil_N, pupil_NE, pupil_E, pupil_SE, pupil_S, pupil_SW, pupil_W, pupil_NW
Optionally also label: reflection (the IR glint), canthus_medial, canthus_lateral.
"""

from __future__ import annotations
import os
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

PUPIL_PARTS = ["pupil_N", "pupil_NE", "pupil_E", "pupil_SE",
               "pupil_S", "pupil_SW", "pupil_W", "pupil_NW"]
EXTRA_PARTS = ["reflection", "canthus_medial", "canthus_lateral"]


# =============================================================================
# Robust ellipse fitting (Halir & Flusser 1998, numerically stable direct LSQ)
# =============================================================================
def fit_ellipse(x, y):
    """Direct least-squares ellipse fit to boundary points.

    Returns dict(cx, cy, major, minor, angle_deg, area) or None if degenerate.
    major/minor are FULL axis lengths (diameters), angle_deg is the major-axis
    orientation.  Needs >= 5 points.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 5:
        return None
    # ISOTROPIC normalisation (single scale) so the ellipse only scales -> exact
    # de-normalisation of centre and axis lengths.
    mx, my = x.mean(), y.mean()
    s = max(np.sqrt(((x - mx)**2 + (y - my)**2).mean()), 1e-6)
    xn, yn = (x - mx) / s, (y - my) / s

    D1 = np.column_stack([xn**2, xn*yn, yn**2])       # quadratic part
    D2 = np.column_stack([xn, yn, np.ones_like(xn)])  # linear part
    S1 = D1.T @ D1; S2 = D1.T @ D2; S3 = D2.T @ D2
    try:
        T = -np.linalg.inv(S3) @ S2.T
    except np.linalg.LinAlgError:
        return None
    M = S1 + S2 @ T
    M = np.array([M[2] / 2, -M[1], M[0] / 2])
    evals, evec = np.linalg.eig(M)
    cond = 4 * evec[0] * evec[2] - evec[1]**2         # ellipse constraint > 0
    a1 = evec[:, cond > 0]
    if a1.shape[1] == 0:
        return None
    a1 = a1[:, 0]
    A, Bxy, C = a1
    D, E, F = T @ a1                                   # conic in normalised coords
    B = Bxy
    den = B*B - 4*A*C                                  # < 0 for an ellipse
    if abs(den) < 1e-12:
        return None
    xc = (2*C*D - B*E) / den
    yc = (2*A*E - B*D) / den
    # semi-axis lengths (Wikipedia general-ellipse formula)
    root = np.sqrt((A - C)**2 + B*B)
    common = 2*(A*E**2 + C*D**2 + F*B**2 - B*D*E - 4*A*C*F)
    t1 = common * ((A + C) + root)
    t2 = common * ((A + C) - root)
    if t1 <= 0 or t2 <= 0:
        return None
    ax_a = -np.sqrt(t1) / den
    ax_b = -np.sqrt(t2) / den
    theta = 0.5 * np.arctan2(-B, C - A)               # major-axis angle
    # de-normalise (isotropic)
    cx = xc * s + mx; cy = yc * s + my
    semi_major = max(ax_a, ax_b) * s
    semi_minor = min(ax_a, ax_b) * s
    major, minor = 2 * semi_major, 2 * semi_minor
    area = np.pi * semi_major * semi_minor
    return dict(cx=float(cx), cy=float(cy), major=float(major),
                minor=float(minor), angle_deg=float(np.degrees(theta)),
                area=float(area))


def _ellipse_residual(fit, x, y):
    """Mean absolute algebraic distance of points to the fitted ellipse (px-ish)."""
    if fit is None:
        return np.inf
    t = np.radians(fit["angle_deg"]); ct, st = np.cos(t), np.sin(t)
    xr = (x - fit["cx"]) * ct + (y - fit["cy"]) * st
    yr = -(x - fit["cx"]) * st + (y - fit["cy"]) * ct
    a, b = fit["major"] / 2, fit["minor"] / 2
    r = (xr / max(a, 1e-6))**2 + (yr / max(b, 1e-6))**2
    return float(np.mean(np.abs(r - 1)))


def _point_residuals(fit, x, y):
    """Per-point |r-1| where r is the normalised ellipse radius of each point."""
    t = np.radians(fit["angle_deg"]); ct, st = np.cos(t), np.sin(t)
    xr = (x - fit["cx"]) * ct + (y - fit["cy"]) * st
    yr = -(x - fit["cx"]) * st + (y - fit["cy"]) * ct
    r = (xr / max(fit["major"]/2, 1e-6))**2 + (yr / max(fit["minor"]/2, 1e-6))**2
    return np.abs(np.sqrt(np.maximum(r, 0)) - 1)


def fit_ellipse_robust(x, y, w=None, max_drop=2, resid_thr=0.15, outlier_k=3.0):
    """Ellipse fit that drops only *clear* outlier keypoints.

    A point is dropped only if its individual residual is both above ``resid_thr``
    AND more than ``outlier_k`` times the median residual of the other points, so
    uniform localisation noise does not trigger rejection.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    keep = np.ones(len(x), bool)
    best = fit_ellipse(x, y)
    for _ in range(max_drop):
        if best is None or keep.sum() <= 5:
            break
        idx = np.where(keep)[0]
        res = _point_residuals(best, x[idx], y[idx])
        worst = int(np.argmax(res))
        others = np.delete(res, worst)
        med = np.median(others) if len(others) else 0.0
        if res[worst] > max(resid_thr, outlier_k * med):
            keep[idx[worst]] = False
            best = fit_ellipse(x[keep], y[keep])
        else:
            break
    if best is not None:
        best["residual"] = _ellipse_residual(best, x[keep], y[keep])
        best["n_used"] = int(keep.sum())
    return best


# =============================================================================
# Main pipeline
# =============================================================================
class PupilPipeline:
    def __init__(self, video_paths, project_dir, pupil_parts=None,
                 camera_fps=10.0, verbose=True):
        self.video_paths = [str(p) for p in np.atleast_1d(video_paths)]
        self.project_dir = Path(project_dir)
        self.pupil_parts = list(pupil_parts) if pupil_parts else list(PUPIL_PARTS)
        self.camera_fps = camera_fps
        self.verbose = verbose
        self.dlc_config = None  # set after create/locate DLC project

    def _log(self, *a):
        if self.verbose:
            print(*a)

    # ---------------------------------------------------------------- ffprobe
    @staticmethod
    def video_info(path):
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,duration",
             "-of", "default=noprint_wrappers=1:nokey=0", str(path)],
            capture_output=True, text=True).stdout
        info = {}
        for line in out.strip().splitlines():
            k, _, v = line.partition("=")
            info[k] = v
        return info

    # --------------------------------------------------- 1. labeling frames
    def extract_labeling_frames(self, out_dir, n_per_video=50, t_start=2.0,
                                t_end=None, diversify=True):
        """Sample frames spread across each video (optionally re-ranked for
        appearance diversity) and save PNGs for DLC labeling.

        Uses ffmpeg seeks (streams from disk, never loads the whole video), so it
        is safe on the multi-GB raw AVIs.
        """
        import cv2, tempfile, shutil
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        for vp in self.video_paths:
            info = self.video_info(vp)
            dur = float(info.get("duration", 0)) or 600.0
            hi = t_end or (dur - 2.0)
            n_probe = n_per_video * (3 if diversify else 1)
            times = np.linspace(t_start, hi, n_probe)
            stem = Path(vp).stem
            # probe into a LOCAL temp dir (network/mounted dirs may forbid unlink)
            tmp = Path(tempfile.mkdtemp(prefix="pupilprobe_"))
            cand = []
            for t in times:
                png = tmp / f"{stem}_t{t:07.2f}.png"
                subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error",
                                "-ss", f"{t:.3f}", "-i", vp, "-frames:v", "1",
                                "-pix_fmt", "gray", str(png)],
                               capture_output=True)
                if png.exists():
                    cand.append(png)
            if diversify and len(cand) > n_per_video:
                feats = np.array([cv2.resize(cv2.imread(str(p), 0), (8, 8)).flatten().astype(float)
                                  for p in cand])
                keep = sorted(set(_kmeans_pick(feats, n_per_video)))
                cand = [cand[i] for i in keep]
            # copy the selected frames to the output folder
            for p in cand:
                dst = out_dir / p.name
                shutil.copyfile(p, dst)
                saved.append(dst)
            shutil.rmtree(tmp, ignore_errors=True)
            self._log(f"[frames] {stem}: kept {len(cand)} labeling frames")
        self._log(f"[frames] total {len(saved)} frames -> {out_dir}")
        return saved

    # --------------------------------------------------- 2. DLC glue (GPU box)
    def create_dlc_project(self, name, experimenter, extra_parts=("reflection",)):
        import deeplabcut
        cfg = deeplabcut.create_new_project(
            name, experimenter, self.video_paths,
            working_directory=str(self.project_dir), copy_videos=False)
        bodyparts = self.pupil_parts + list(extra_parts)
        deeplabcut.auxiliaryfunctions.edit_config(
            cfg, {"bodyparts": bodyparts, "dotsize": 5})
        self.dlc_config = cfg
        self._log(f"[dlc] project created: {cfg}")
        self._log(f"[dlc] bodyparts set to: {bodyparts}")
        self._log("[dlc] now: deeplabcut.label_frames(cfg) on a machine with a display")
        return cfg

    def train(self, **kw):
        import deeplabcut
        deeplabcut.create_training_dataset(self.dlc_config)
        deeplabcut.train_network(self.dlc_config, **kw)

    def analyze(self, videos=None, **kw):
        import deeplabcut
        vids = videos or self.video_paths
        deeplabcut.analyze_videos(self.dlc_config, vids, save_as_csv=True, **kw)
        return vids

    # --------------------------------------------------- 3. post-processing
    @staticmethod
    def load_dlc(path):
        """Load a DLC .h5/.csv into a tidy dict {part: (x, y, likelihood arrays)}."""
        path = str(path)
        if path.endswith(".h5"):
            df = pd.read_hdf(path)
        else:
            df = pd.read_csv(path, header=[0, 1, 2], index_col=0)
        df.columns = df.columns.droplevel(0)  # drop scorer level
        parts = df.columns.get_level_values(0).unique().tolist()
        data = {p: (df[(p, "x")].to_numpy(),
                    df[(p, "y")].to_numpy(),
                    df[(p, "likelihood")].to_numpy()) for p in parts}
        return data, len(df)

    def measure(self, dlc, likelihood_thr=0.6, min_points=5,
                resid_thr=0.25, smooth_win=5):
        """Per-frame ellipse fit -> pupil area / diameter / center DataFrame.

        `dlc` is the dict from `load_dlc`. Frames with too few confident
        pupil-edge points, or a poor fit, are flagged as blink/low-confidence
        (NaN measurements). Returns a DataFrame with raw + normalized traces.
        """
        parts = [p for p in self.pupil_parts if p in dlc]
        n = len(next(iter(dlc.values()))[0])
        rows = []
        for f in range(n):
            xs, ys, ws = [], [], []
            for p in parts:
                x, y, l = dlc[p]
                if l[f] >= likelihood_thr:
                    xs.append(x[f]); ys.append(y[f]); ws.append(l[f])
            rec = dict(frame=f, n_valid=len(xs))
            if len(xs) >= min_points:
                fit = fit_ellipse_robust(xs, ys, resid_thr=resid_thr)
            else:
                fit = None
            if fit and fit.get("residual", np.inf) <= resid_thr * 4:
                rec.update(cx=fit["cx"], cy=fit["cy"], major=fit["major"],
                           minor=fit["minor"], angle=fit["angle_deg"],
                           area=fit["area"], diameter=np.sqrt(fit["major"]*fit["minor"]),
                           residual=fit["residual"], blink=0)
            else:
                rec.update(cx=np.nan, cy=np.nan, major=np.nan, minor=np.nan,
                           angle=np.nan, area=np.nan, diameter=np.nan,
                           residual=np.nan, blink=1)
            rows.append(rec)
        df = pd.DataFrame(rows)
        df["time_s"] = df["frame"] / self.camera_fps
        df = self._postprocess(df, smooth_win)
        self._log(f"[measure] {n} frames, {int((df.blink==0).sum())} good, "
                  f"{int(df.blink.sum())} blink/low-conf "
                  f"({100*df.blink.mean():.1f}%)")
        return df

    def _postprocess(self, df, smooth_win):
        # temporal outlier rejection on area (robust z on good frames)
        a = df["area"].to_numpy(float)
        good = ~np.isnan(a)
        if good.sum() > 10:
            med = np.nanmedian(a); mad = np.nanmedian(np.abs(a - med)) + 1e-9
            outlier = good & (np.abs(a - med) > 6 * 1.4826 * mad)
            df.loc[outlier, ["area", "diameter", "major", "minor", "cx", "cy"]] = np.nan
            df.loc[outlier, "blink"] = 1
        # interpolate short gaps for a clean trace (keep raw in *_raw)
        for c in ["area", "diameter", "cx", "cy", "major", "minor"]:
            df[c + "_raw"] = df[c]
            s = df[c].interpolate(limit=int(self.camera_fps * 1), limit_area="inside")
            if smooth_win and smooth_win > 1:
                s = s.rolling(smooth_win, center=True, min_periods=1).median()
            df[c] = s
        # normalized pupil size
        base = np.nanpercentile(df["area"], 98)
        df["area_fracmax"] = df["area"] / base if base else np.nan
        m, sd = np.nanmean(df["area"]), np.nanstd(df["area"])
        df["area_z"] = (df["area"] - m) / (sd + 1e-9)
        return df

    # --------------------------------------------------- QC / save
    @staticmethod
    def save(df, path):
        path = str(path)
        (df.to_hdf(path, key="pupil") if path.endswith(".h5")
         else df.to_csv(path, index=False))
        print(f"[save] {path}")

    def plot_trace(self, df, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(13, 3))
        ax.plot(df["time_s"], df["area"], lw=0.8, label="pupil area (px²)")
        bl = df[df["blink"] == 1]
        ax.plot(bl["time_s"], np.full(len(bl), np.nanmin(df["area"])), "r|",
                ms=8, label="blink/low-conf")
        ax.set(xlabel="time (s)", ylabel="pupil area (px²)",
               title=f"pupil area — {int((df.blink==0).sum())}/{len(df)} good frames")
        ax.legend(loc="upper right", fontsize=8)
        return ax

    def plot_overlay(self, video_path, dlc, frame_idx, out_png=None):
        """Draw the fitted ellipse + keypoints on a single video frame (QC)."""
        import cv2
        info = self.video_info(video_path)
        fps = eval(info.get("avg_frame_rate", "10/1"))
        t = frame_idx / fps
        tmp = "/tmp/_ovl.png"
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-ss",
                        f"{t:.3f}", "-i", str(video_path), "-frames:v", "1",
                        "-pix_fmt", "gray", tmp], capture_output=True)
        g = cv2.imread(tmp, 0); vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        parts = [p for p in self.pupil_parts if p in dlc]
        xs, ys = [], []
        for p in parts:
            x, y, l = dlc[p]
            if l[frame_idx] >= 0.6:
                xs.append(x[frame_idx]); ys.append(y[frame_idx])
                cv2.circle(vis, (int(x[frame_idx]), int(y[frame_idx])), 4, (0, 255, 0), -1)
        fit = fit_ellipse_robust(xs, ys) if len(xs) >= 5 else None
        if fit:
            cv2.ellipse(vis, (int(fit["cx"]), int(fit["cy"])),
                        (int(fit["major"]/2), int(fit["minor"]/2)),
                        fit["angle_deg"], 0, 360, (0, 0, 255), 2)
        if out_png:
            cv2.imwrite(str(out_png), vis)
        return vis


# ------------------------------------------------------------------ helpers
def _kmeans_pick(feats, k, iters=25, seed=0):
    """Tiny k-means; return one representative (nearest-to-centroid) index/cluster."""
    rng = np.random.default_rng(seed)
    c = feats[rng.choice(len(feats), k, replace=False)]
    for _ in range(iters):
        d = ((feats[:, None, :] - c[None]) ** 2).sum(-1)
        lab = d.argmin(1)
        for j in range(k):
            if (lab == j).any():
                c[j] = feats[lab == j].mean(0)
    # representative = closest point to each centroid
    picks = []
    for j in range(k):
        idx = np.where(lab == j)[0]
        if len(idx):
            picks.append(idx[np.argmin(((feats[idx] - c[j])**2).sum(1))])
    return picks
