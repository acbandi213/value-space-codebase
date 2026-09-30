"""Approach-event detector for the open-field social arena, following
`planning /Self_Paced_Approach_Neural_Analysis_Framework.docx`.

Pipeline
  1. keypoints (DLC SuperAnimal TopViewMouse) -> confidence mask, short-gap interpolation, zero-phase smoothing
  2. kinematics: body centroid, speed, heading (neck -> nose); per wall: snout distance d_j, inward velocity v_par_j,
     head angle theta_j, target-support score S_j = w_v * v_par + w_h * cos(theta) - w_d * d
  3. events per episode: approach onset (from an eligible open-arena state), target commitment (one wall wins by a
     margin for a sustained interval, with hysteresis), investigation (snout in a wall's interaction zone),
     withdrawal, and an outcome label. Commitment is detected prospectively; outcome is assigned afterwards.

All thresholds live in DEFAULTS and were chosen from behavior only. Do not tune them on neural data.
"""
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

KP = dict(nose="nose", neck="neck", center="mouse_center", tail="tail_base",
          l_ear="left_ear", r_ear="right_ear", mid_back="mid_back")

DEFAULTS = dict(
    pcutoff=0.4,            # DLC likelihood below which a keypoint is treated as missing
    max_gap_s=0.25,         # interpolate gaps up to this length; longer gaps stay NaN
    smooth_hz=4.0,          # low-pass corner for position smoothing (zero phase)
    vel_win_s=0.2,          # window for the central-difference velocity (suppresses tracking jitter)
    w_v=1.0, w_h=3.0, w_d=0.3,   # target-support weights (cm/s, unitless, cm)
    v_min=4.0,              # cm/s: overall speed for approach onset
    v_in_min=2.5,           # cm/s: inward speed for approach onset / commitment entry
    v_in_keep=1.0,          # cm/s: looser continuation threshold (hysteresis)
    theta_max_deg=60.0,     # head angle limit for commitment entry
    theta_keep_deg=90.0,    # looser continuation limit
    persist_s=0.25,         # commitment must hold this long
    margin=2.0,             # S_target must exceed the best other wall by this margin at entry
    d_zone_cm=3.0,          # snout distance defining the interaction zone
    d_elig_cm=8.0,          # snout must be at least this far from every interaction segment to start an episode
    lockout_s=1.0,          # after withdrawal/investigation before a new episode may start
    latency_max_s=3.0,      # commitment -> investigation must occur within this window else 'aborted'
    withdraw_v=3.0,         # cm/s outward speed for withdrawal onset
    withdraw_persist_s=0.3,
    min_bout_gap_s=0.5,     # merge threshold interruptions shorter than this
)


# ----------------------------------------------------------------------------- preprocessing
def load_dlc(path):
    """Read a DLC .h5/.csv into a tidy frame: columns <bodypart>_x/_y/_p for a single animal."""
    df = pd.read_hdf(path) if str(path).endswith(".h5") else pd.read_csv(path, header=[0, 1, 2, 3] if "csv" in str(path) else None, index_col=0)
    cols = df.columns
    if cols.nlevels == 4:       # scorer, individual, bodypart, coord
        df = df.droplevel(0, axis=1)
        indiv = df.columns.get_level_values(0).unique()[0]
        df = df[indiv]
    elif cols.nlevels == 3:
        df = df.droplevel(0, axis=1)
    out = pd.DataFrame(index=df.index)
    for bp in df.columns.get_level_values(0).unique():
        out[f"{bp}_x"] = df[(bp, "x")].values
        out[f"{bp}_y"] = df[(bp, "y")].values
        out[f"{bp}_p"] = df[(bp, "likelihood")].values
    return out


def clean_track(xy, p, fps, pcutoff, max_gap_s, smooth_hz):
    """Mask low-confidence samples, interpolate short gaps, zero-phase low-pass. xy: (N,2)."""
    xy = np.asarray(xy, float).copy()
    xy[np.asarray(p) < pcutoff] = np.nan
    bad = np.isnan(xy).any(1)
    max_gap = int(round(max_gap_s * fps))
    filled = pd.DataFrame(xy).interpolate(limit_area="inside").values
    # restore NaN inside gaps longer than max_gap
    d = np.diff(np.r_[0, bad.astype(int), 0]); starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    for s_, e_ in zip(starts, ends):
        if e_ - s_ > max_gap:
            filled[s_:e_] = np.nan
    out = filled
    if med_win := int(round(0.25 * fps)) | 1:
        from scipy.ndimage import median_filter
        ok0 = ~np.isnan(out).any(1)
        tmp = pd.DataFrame(out).interpolate(limit_direction="both").values
        tmp = median_filter(tmp, size=(med_win, 1), mode="nearest")
        out = np.where(ok0[:, None], tmp, np.nan)
    if smooth_hz:
        b, a = butter(2, smooth_hz / (fps / 2))
        ok = ~np.isnan(out).any(1)
        edges = np.diff(np.r_[0, ok.astype(int), 0])
        for s_, e_ in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
            if e_ - s_ > 12:
                out[s_:e_] = filtfilt(b, a, out[s_:e_], axis=0)
    return out


# ----------------------------------------------------------------------------- kinematics
def kinematics(dlc, arena, fps, params=None):
    P = {**DEFAULTS, **(params or {})}
    pts = {}
    for key, bp in KP.items():
        if f"{bp}_x" in dlc:
            pts[key] = clean_track(dlc[[f"{bp}_x", f"{bp}_y"]].values, dlc[f"{bp}_p"].values, fps, P["pcutoff"], P["max_gap_s"], P["smooth_hz"])
    nose, neck = pts["nose"], pts["neck"]
    body = pts.get("center", pts.get("mid_back", neck))
    N = len(nose)
    k = pd.DataFrame(index=np.arange(N))
    k["nose_x"], k["nose_y"] = nose.T
    k["body_x"], k["body_y"] = body.T
    h = max(1, int(round(P["vel_win_s"] * fps / 2)))                    # central difference over vel_win_s
    vel = np.full_like(body, np.nan)
    vel[h:-h] = (body[2 * h:] - body[:-2 * h]) / (2 * h) * fps / arena.px_per_cm   # cm/s (image y down)
    k["speed"] = np.linalg.norm(vel, axis=1)
    head = nose - neck
    k["heading"] = np.arctan2(head[:, 1], head[:, 0])
    bc = arena.cm(body); k["x_cm"], k["y_cm"] = bc.T
    for w in arena.walls:
        d, c = arena.dist_to_wall(nose, w)
        to_c = c - body; nrm = np.linalg.norm(to_c, axis=1) + 1e-9
        v_par = (vel * (to_c / nrm[:, None])).sum(1)
        cos_th = (head * to_c).sum(1) / (np.linalg.norm(head, axis=1) * nrm + 1e-9)
        k[f"d_{w}"] = d
        k[f"vin_{w}"] = v_par
        k[f"theta_{w}"] = np.degrees(np.arccos(np.clip(cos_th, -1, 1)))
        k[f"S_{w}"] = P["w_v"] * v_par + P["w_h"] * cos_th - P["w_d"] * d
    k["valid"] = ~np.isnan(nose).any(1) & ~np.isnan(body).any(1)
    return k


# ----------------------------------------------------------------------------- events
def _runs(mask):
    m = np.asarray(mask, bool); d = np.diff(np.r_[0, m.astype(int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def detect_events(k, arena, fps, params=None):
    """Return an events DataFrame; one row per episode. Frames are indices into `k`."""
    P = {**DEFAULTS, **(params or {})}
    walls = list(arena.walls)
    D = np.stack([k[f"d_{w}"].values for w in walls], 1)
    VIN = np.stack([k[f"vin_{w}"].values for w in walls], 1)
    TH = np.stack([k[f"theta_{w}"].values for w in walls], 1)
    S = np.stack([k[f"S_{w}"].values for w in walls], 1)
    speed = k["speed"].values; valid = k["valid"].values
    N = len(k); persist = int(round(P["persist_s"] * fps)); lock = int(round(P["lockout_s"] * fps))
    in_zone = D < P["d_zone_cm"]                                  # (N, walls)
    eligible = valid & (D.min(1) > P["d_elig_cm"])

    # commitment candidate per frame (prospective): best wall by S, with entry/continuation thresholds
    best = np.nanargmax(np.where(np.isnan(S), -np.inf, S), 1)
    S_sorted = np.sort(np.nan_to_num(S, nan=-np.inf), 1)
    margin_ok = (S_sorted[:, -1] - S_sorted[:, -2]) > P["margin"]
    idx = np.arange(N)
    entry = valid & margin_ok & (speed > P["v_min"]) & (VIN[idx, best] > P["v_in_min"]) & (TH[idx, best] < P["theta_max_deg"])
    keep = valid & (VIN[idx, best] > P["v_in_keep"]) & (TH[idx, best] < P["theta_keep_deg"])

    events = []; i = 0; last_end = -lock
    while i < N:
        if not eligible[i] or i < last_end + lock:
            i += 1; continue
        # approach onset: first frame of sustained movement toward any wall
        j = i
        while j < N and not (valid[j] and speed[j] > P["v_min"] and VIN[j].max() > P["v_in_min"]):
            if not valid[j] and j - i > lock: break
            if in_zone[j].any(): break
            j += 1
        if j >= N or in_zone[j].any() or not valid[j]:
            i = j + 1; continue
        onset = j
        # follow the episode until investigation, withdrawal-to-eligible, or timeout
        commit_f, commit_w, invest_f, invest_w, switch = None, None, None, None, False
        t = onset; tmax = min(N, onset + int(30 * fps))
        while t < tmax:
            if commit_f is None and entry[t] and (t + persist <= N) and keep[t:t + persist].all() and (best[t:t + persist] == best[t]).all():
                commit_f, commit_w = t, walls[best[t]]
            elif commit_f is not None and entry[t] and walls[best[t]] != commit_w and keep[t:t + persist].all() and (best[t:t + persist] == best[t]).all():
                switch = True; commit_w = walls[best[t]]; commit_f = t
            if in_zone[t].any():
                invest_f, invest_w = t, walls[int(np.argmin(D[t]))]; break
            if t > onset + 5 * fps and eligible[t] and speed[t] < P["v_min"]:
                break                                             # gave up, back in the open
            t += 1
        # withdrawal after investigation: outward speed sustained
        withdraw_f = None
        if invest_f is not None:
            wi = walls.index(invest_w); u = invest_f
            while u < N and (in_zone[u, wi] or not valid[u]): u += 1
            wp = int(round(P["withdraw_persist_s"] * fps))
            for v in range(u, min(N - wp, u + int(20 * fps))):
                if (VIN[v:v + wp, wi] < -P["withdraw_v"]).all():
                    withdraw_f = v; break
        # outcome
        if invest_f is not None and commit_f is not None and invest_w == commit_w and (invest_f - commit_f) <= P["latency_max_s"] * fps:
            outcome = "switch_success" if switch else "success"
        elif invest_f is not None and commit_f is None:
            outcome = "investigate_no_commit"
        elif invest_f is not None and invest_w != commit_w:
            outcome = "switch"
        elif commit_f is not None:
            outcome = "committed_aborted"
        else:
            outcome = "approach_no_commit"
        events.append(dict(onset=onset, commit=commit_f, target=commit_w, invest=invest_f, invest_wall=invest_w,
                           withdraw=withdraw_f, outcome=outcome))
        end = max(x for x in [onset, commit_f, invest_f, withdraw_f] if x is not None)
        last_end = end; i = end + 1
    ev = pd.DataFrame(events)
    for c in ["onset", "commit", "invest", "withdraw"]:
        if c in ev: ev[f"{c}_s"] = ev[c] / fps
    return ev


def run(dlc, arena, fps, params=None):
    k = kinematics(dlc, arena, fps, params)
    return k, detect_events(k, arena, fps, params)
