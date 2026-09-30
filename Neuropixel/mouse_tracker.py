"""Classical single-mouse tracker for the IR open-field video (cropped/downscaled chunks in cache/video_small).

Per frame: |frame - background| -> threshold -> morphology -> largest blob inside the arena = mouse.
Body centroid and axis from image moments; head end = the end of the body axis closer to the brightest
foreground cluster (the headstage). Emits a DLC-like table with nose / neck / mouse_center / tail_base
so `approach_events.py` can consume it. Likelihood columns encode a quality score in [0, 1].
"""
import cv2, numpy as np, pandas as pd, time
from pathlib import Path

BG_PATH = "cache/background_median_small.npy"   # whole-session median (robust to corner sitting)

PARAMS = dict(
    bg_samples=60,          # frames sampled across a chunk for the median background
    thr=10,                 # trunk = pixels darker than background by more than this (CLAHE video)
    thr_low=10,             # lower threshold used to grow the selected blob (hysteresis)
    grow_px=25,             # hysteresis growth is limited to this neighbourhood of the seed blob
    fill_k=15,              # dilation used to merge trunk pieces split by a narrow waist
    bright_thr=45,          # frame - bg threshold for headstage / bright fur
    open_k=5, close_k=9,    # morphology kernels (px at 0.4 scale)
    body_k=13,              # closing that bridges gaps in the dark trunk
    min_area=600,           # dark trunk blob (px at 0.4 scale); trunk is typically 1500-3500
    headstage_min=25,       # px: compact bright blob taken as the headstage
    motion_min=5.0,         # mean |g_t - g_(t-motion_lag)| inside the blob to count as moving
    motion_lag=15,          # frames (~0.25 s)
    max_jump=120,           # px: prefer the blob nearest the previous position when several candidates
    nose_offset=0.15,       # nose = head end of the axis + this fraction of body length
)


def rolling_backgrounds(path, sample_s=5.0, half_win_s=180.0):
    """Median background per minute of video from frames sampled every `sample_s` within +/- half_win_s.
    Adapts to moved objects within a few minutes; a mouse resting > half_win_s vanishes (handled by position hold)."""
    cap = cv2.VideoCapture(str(path)); N = int(cap.get(7)); fps = cap.get(5)
    idx = np.arange(0, N, int(round(sample_s * fps))); fr = []
    for i in idx:
        cap.set(1, int(i)); ok, f = cap.read()
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if ok else fr[-1])
    cap.release(); fr = np.stack(fr)
    minutes = np.arange(0, N, int(round(60 * fps))); bgs = []
    for m in minutes:
        sel = np.abs(idx - m) <= half_win_s * fps
        bgs.append(np.median(fr[sel], 0).astype(np.float32))
    return minutes, bgs


def chunk_background(path, n=60):
    cap = cv2.VideoCapture(str(path)); N = int(cap.get(7)); fr = []
    for i in np.linspace(0, N - 1, n).astype(int):
        cap.set(1, int(i)); ok, f = cap.read()
        if ok: fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    return np.median(np.stack(fr), 0).astype(np.float32)


def arena_mask(shape, rim_small, shrink_px=6):
    m = np.zeros(shape, np.uint8)
    c = rim_small.mean(0); poly = c + (rim_small - c) * (1 - shrink_px / np.linalg.norm(rim_small[0] - c))
    cv2.fillPoly(m, [poly.astype(np.int32)], 1)
    return m.astype(bool)


def track_frame(g, bg, amask, prev, P, k_open, k_close, k_body, k_fill, k_grow, g_prev=None):
    diff = g.astype(np.float32) - bg
    # trunk of a black mouse = pixels darker than background. Tail, tether, paws, pellets are bright -> excluded.
    dark = ((diff < -P["thr"]) & amask).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k_open)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k_body)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(dark, 8)
    cands, motion, dist = [], [], []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < P["min_area"]:
            continue
        sel = lab == i
        mo = float(np.abs(g.astype(np.float32) - g_prev)[sel].mean()) if g_prev is not None else 0.0
        cands.append(i); motion.append(mo)
        dist.append(np.linalg.norm(cents[i] - prev) if prev is not None else 0.0)
    if not cands:
        return None
    motion, dist, areas = np.array(motion), np.array(dist), np.array([stats[i, cv2.CC_STAT_AREA] for i in cands])
    if prev is not None and dist.min() < P["max_jump"]:
        near = dist < P["max_jump"]
        i = cands[int(np.argmax(np.where(near, areas, -1)))]          # largest dark blob near the previous position
    elif motion.max() >= P["motion_min"]:
        i = cands[int(np.argmax(motion * np.sqrt(areas)))]          # (re)acquire: the moving blob
    else:
        i = cands[int(np.argmax(areas))]
    m = lab == i
    # merge trunk pieces split by a narrow waist: other dark blobs within touching distance of the chosen one
    dil = cv2.dilate(m.astype(np.uint8), k_fill).astype(bool)
    for j in cands:
        if j != i and (dil & (lab == j)).any():
            m |= lab == j
    if m.sum() < 150:
        return None
    mom = cv2.moments(m.astype(np.uint8), binaryImage=True)
    cx, cy = mom["m10"] / mom["m00"], mom["m01"] / mom["m00"]
    mu20, mu02, mu11 = mom["mu20"] / mom["m00"], mom["mu02"] / mom["m00"], mom["mu11"] / mom["m00"]
    ang = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    lam = 0.5 * (mu20 + mu02) + 0.5 * np.sqrt(4 * mu11 ** 2 + (mu20 - mu02) ** 2)
    half_len = 2.0 * np.sqrt(max(lam, 1e-6))
    ax = np.array([np.cos(ang), np.sin(ang)])
    # shape cue: rump wider than head
    ys_, xs_ = np.nonzero(m); rel = np.stack([xs_ - cx, ys_ - cy], 1)
    along = rel @ ax; across = rel @ np.array([-ax[1], ax[0]])
    lo, hi = np.percentile(along, [15, 85])
    w_neg = np.std(across[along < lo]) if (along < lo).sum() > 10 else np.nan
    w_pos = np.std(across[along > hi]) if (along > hi).sum() > 10 else np.nan
    shape_sign = 0.0 if np.isnan(w_neg) or np.isnan(w_pos) else float(np.sign(w_neg - w_pos))
    shape_q = 0.0 if shape_sign == 0 else float(min(1.0, abs(w_neg - w_pos) / (0.15 * (w_neg + w_pos) + 1e-6)))
    # headstage cue: compact bright blob next to the trunk (tail is thin -> removed by the opening)
    ring = cv2.dilate(m.astype(np.uint8), k_close, iterations=2).astype(bool)
    bright = ((diff > P["bright_thr"]) & ring).astype(np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, k_open)
    nb, labb, statb, centb = cv2.connectedComponentsWithStats(bright, 8)
    sgn, head_q = 1.0, 0.0
    if nb > 1:
        areas_b = statb[1:, cv2.CC_STAT_AREA]; jb = 1 + int(np.argmax(areas_b))
        if areas_b.max() >= P["headstage_min"]:
            v = centb[jb] - [cx, cy]; proj = np.dot(v, ax)
            if abs(proj) > 0.3 * half_len:
                sgn = float(np.sign(proj)); head_q = min(1.0, areas_b.max() / 80)
    if head_q < 0.5 and shape_sign != 0 and shape_q > 0.5:
        sgn = shape_sign
    head_dir = ax * sgn
    head = np.array([cx, cy]) + head_dir * half_len
    nose = head + head_dir * P["nose_offset"] * 2 * half_len
    neck = np.array([cx, cy]) + head_dir * 0.5 * half_len
    tail = np.array([cx, cy]) - head_dir * half_len
    area = int(m.sum())
    q = float(np.clip((area - 300) / 800, 0.2, 1.0))
    return dict(cx=cx, cy=cy, nose=nose, neck=neck, tail=tail, dir=head_dir, half_len=half_len, area=area,
                q_body=q, q_head=max(head_q, shape_q), n_cands=len(cands))


prev_dir = np.array([1.0, 0.0])


def track_video(path, rim_small, P=PARAMS, out=None, progress_every=5000, max_frames=None, start=0):
    global prev_dir
    path = Path(path)
    minutes, bgs = rolling_backgrounds(path); bg = bgs[0]
    cap = cv2.VideoCapture(str(path)); N = int(cap.get(7))
    if start: cap.set(1, int(start))
    N = min(N - start, max_frames) if max_frames else N - start
    amask = arena_mask(bg.shape, rim_small)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (P["open_k"],) * 2)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (P["close_k"],) * 2)
    k_body = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (P["body_k"],) * 2)
    k_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (P["fill_k"],) * 2)
    k_grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (P["grow_px"],) * 2)
    rows = np.full((N, 14), np.nan, np.float32); prev = None; last = None; t0 = time.time()
    from collections import deque
    buf = deque(maxlen=P["motion_lag"] + 1)
    for f in range(N):
        ok, fr = cap.read()
        if not ok: break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        bg = bgs[min(int(np.searchsorted(minutes, f + start, side="right") - 1), len(bgs) - 1)]
        buf.append(g.astype(np.float32)); g_prev = buf[0] if len(buf) > P["motion_lag"] else None
        r = track_frame(g, bg, amask, prev, P, k_open, k_close, k_body, k_fill, k_grow, g_prev)
        if r is not None:
            prev = np.array([r["cx"], r["cy"]]); prev_dir = r["dir"]; last = r
            rows[f] = [r["cx"], r["cy"], *r["nose"], *r["neck"], *r["tail"], *r["dir"], r["half_len"], r["area"], r["q_body"], r["q_head"]]
        elif prev is not None:                      # nothing found: hold last position (resting mouse absorbed by background)
            r = last
            rows[f] = [r["cx"], r["cy"], *r["nose"], *r["neck"], *r["tail"], *r["dir"], r["half_len"], 0, 0.1, r["q_head"]]
        if progress_every and f % progress_every == 0 and f:
            print(f"{path.name}: {f}/{N} {f/(time.time()-t0):.0f} fps", flush=True)
    cap.release()
    df = pd.DataFrame(rows, columns=["mouse_center_x", "mouse_center_y", "nose_x", "nose_y", "neck_x", "neck_y",
                                     "tail_base_x", "tail_base_y", "dir_x", "dir_y", "half_len", "area", "q_body", "q_head"])
    df["mouse_center_p"] = df["q_body"]; df["neck_p"] = df["q_body"]; df["tail_base_p"] = df["q_body"]
    df["nose_p"] = df["q_body"] * (0.5 + 0.5 * df["q_head"])
    df = refine_heading(df)
    if out: df.to_parquet(out) if str(out).endswith(".parquet") else df.to_csv(out, index=False)
    return df


def refine_heading(df, win=61, v_min=1.5):
    """Resolve the 180-degree axis ambiguity offline.

    1. Make the body axis temporally continuous (flip sign to match the previous frame).
    2. Per-frame head evidence = sign from the bright headstage cluster (weight q_head) plus the direction
       of motion when the mouse is moving faster than v_min px/frame (mice walk head first).
    3. Median-filter that evidence over `win` frames and rebuild nose / neck / tail from the axis.
    """
    from scipy.signal import medfilt
    ax = df[["dir_x", "dir_y"]].values.copy(); ok = ~np.isnan(ax[:, 0])
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return df
    cont = ax.copy()
    for a, b in zip(idx[:-1], idx[1:]):
        if np.dot(cont[a], cont[b]) < 0: cont[b] = -cont[b]
    raw_sign = np.sign((ax * cont).sum(1))                     # +1 where the raw head guess agrees with the continuous axis
    c = df[["mouse_center_x", "mouse_center_y"]].values
    vel = np.gradient(np.where(np.isnan(c), np.nan, c), axis=0)
    speed = np.linalg.norm(vel, axis=1)
    v_sign = np.sign((vel * cont).sum(1)); v_sign[~(speed > v_min)] = 0
    evidence = 0.5 * raw_sign * df["q_head"].values + 2.0 * v_sign   # motion direction dominates when moving
    evidence = np.nan_to_num(evidence)
    sm = medfilt(evidence, win)
    sm2 = np.convolve(evidence, np.ones(win) / win, mode="same")
    sign = np.where(sm != 0, np.sign(sm), np.sign(sm2 + 1e-9))
    head_dir = cont * sign[:, None]
    hl = df["half_len"].values[:, None]
    df["dir_x"], df["dir_y"] = head_dir.T
    head = c + head_dir * hl
    df[["nose_x", "nose_y"]] = head + head_dir * PARAMS["nose_offset"] * 2 * hl
    df[["neck_x", "neck_y"]] = c + head_dir * 0.5 * hl
    df[["tail_base_x", "tail_base_y"]] = c - head_dir * hl
    df["head_conf"] = np.abs(sm2)
    return df


def _work(args):
    path, rim_small, out = args
    df = track_video(path, rim_small, out=out, progress_every=0)
    return path, int(df["mouse_center_x"].notna().sum()), len(df)


if __name__ == "__main__":
    import sys, glob
    from multiprocessing import Pool
    from arena import to_small, RIM_ORIG
    rim = to_small(RIM_ORIG)
    files = sorted(glob.glob("cache/video_small/chunk_*.mp4"))
    jobs = [(f, rim, f.replace(".mp4", "_track.csv")) for f in files]
    with Pool(len(jobs)) as p:
        for path, n_ok, n in p.imap_unordered(_work, jobs):
            print(f"{Path(path).name}: tracked {n_ok}/{n} frames", flush=True)
    print("ALL_DONE", flush=True)
