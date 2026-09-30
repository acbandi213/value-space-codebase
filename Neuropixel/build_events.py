"""Combine chunk tracks -> kinematics on the ephys clock -> approach events.

Outputs (cache/):
  tracking_session.parquet   one row per video frame: frame, t_ephys, positions (small px), quality
  kinematics_session.parquet one row per frame: speed, heading, per-wall distance / inward speed / angle / support score
  approach_events.csv        one row per episode with onset / commit / invest / withdraw frames, ephys times, outcome
"""
import glob, re, sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import npx_io as io, approach_events as ae
from arena import Arena

FPS = 59.63700234192037
CACHE = Path(__file__).parent / "cache"


def load_tracks(pattern="cache/video_small/chunk_*_track.csv"):
    parts = []
    for f in sorted(glob.glob(str(Path(__file__).parent / pattern))):
        f0 = int(re.search(r"_f(\d+)_track", f).group(1))
        d = pd.read_csv(f); d.insert(0, "frame", f0 + np.arange(len(d))); parts.append(d)
    df = pd.concat(parts, ignore_index=True).sort_values("frame").reset_index(drop=True)
    warp = io.load_video_warp(CACHE)
    df.insert(1, "t_ephys", io.frame_to_ephys_time(df["frame"].values, warp))
    return df


def main(track_pattern="cache/video_small/chunk_*_track.csv", t_max=None, params=None, tag="session"):
    df = load_tracks(track_pattern)
    if t_max is not None:
        df = df[df.t_ephys <= t_max]
    arena = Arena(space="small").fit_boundaries(df[["nose_x", "nose_y"]].values)   # empirical wall interfaces
    # stimulus cages along the LEFT wall (user, 2026-09-09): male / fake mouse / female from top to bottom.
    # The left wall polyline runs from bottom-left (a) to top-left (b), so list bottom -> top.
    arena.split_wall("left", ["female", "fake", "male"])
    kin, ev = ae.run(df, arena, FPS, params)
    json.dump({w: arena.walls[w]["poly"].tolist() for w in arena.walls if "poly" in arena.walls[w]}, open(CACHE / f"wall_boundaries_{tag}.json", "w"))
    import pickle; pickle.dump(arena, open(CACHE / f"arena_{tag}.pkl", "wb"))
    kin.insert(0, "frame", df["frame"].values); kin.insert(1, "t_ephys", df["t_ephys"].values)
    for c in ["onset", "commit", "invest", "withdraw"]:
        if c in ev:
            ev[f"{c}_t"] = np.where(ev[c].notna(), np.interp(ev[c].fillna(0).values, np.arange(len(df)), df["t_ephys"].values), np.nan)
    df.to_parquet(CACHE / f"tracking_{tag}.parquet"); kin.to_parquet(CACHE / f"kinematics_{tag}.parquet")
    ev.to_csv(CACHE / f"approach_events_{tag}.csv", index=False)
    return df, kin, ev, arena


if __name__ == "__main__":
    df, kin, ev, arena = main(t_max=5707.0)
    print(f"frames {len(df)}, tracked {df.mouse_center_x.notna().mean()*100:.1f}%, ephys span {df.t_ephys.min():.0f}-{df.t_ephys.max():.0f} s")
    print(f"px/cm {arena.px_per_cm:.2f}; median speed {kin.speed.median():.1f} cm/s; frac speed>4 {np.nanmean(kin.speed>4):.2f}")
    print(f"{len(ev)} episodes"); print(ev.outcome.value_counts().to_string()); print(ev.invest_wall.value_counts(dropna=False).to_string())
