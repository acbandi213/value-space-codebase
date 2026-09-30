"""
abf_imaging_alignment.py
========================
Align suite2p neural data with pClamp (.abf) behavioral / imaging-clock signals.

Designed for a 2-photon / GRIN-lens rig where each session is recorded as one or
more acquisitions. For every acquisition there is:
  * a suite2p run (concatenated across acquisitions in ``suite2p/plane0``)
  * a Clampex ``.abf`` file containing, on separate analog channels:
        - galvo sawtooth  -> imaging frame clock (one ramp per frame)
        - stimulus boxcar -> TTL high while a stimulus is on screen
        - roll / pitch / yaw -> mouse locomotion on a ball (continuous analog)

The core idea (same as the Runyan-lab Virmen pipeline, generalised):
  1. Detect each imaging frame in abf time from the galvo flyback.
  2. Detect behavioural events (stimulus on/off) in abf time.
  3. Map every abf event to the imaging frame it belongs to, accounting for the
     frame offset of earlier acquisitions (suite2p ``frames_per_folder``).
  4. Resample continuous behaviour (ball motion) onto the imaging frame grid.
  5. Emit trial-by-trial dicts and fixed-window, event-locked tensors.

Typical notebook use
--------------------
>>> from abf_imaging_alignment import AlignmentSession
>>> s = AlignmentSession(
...     suite2p_path="/Volumes/Runyan5/Naomi/GRINPL1/.../suite2p/plane0",
...     abf_paths=["...0000.abf", "...0001.abf", "...0002.abf"],
... )
>>> s.load_abf().detect_frames().detect_stim().load_suite2p().compute_dff()
>>> s.validate()                       # frame counts vs suite2p
>>> trials = s.build_trials(pre_s=1.0, post_s=3.0)
>>> X = s.epoch("dff", anchor="stim_onset", pre_s=1.0, post_s=3.0)  # (trials, neurons, time)

Author: built with Akhil, Value Space Project.
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np

try:
    from neo.io import AxonIO
except Exception:  # pragma: no cover
    AxonIO = None

from scipy.signal import find_peaks
from scipy.ndimage import percentile_filter


# ----------------------------------------------------------------------------
# Default channel map for the GRINPL rig.  Channels are addressed BY NAME or
# INDEX so the code is robust to neo packing them into one multichannel
# AnalogSignal.  Override any of these in the constructor if a session differs.
# ----------------------------------------------------------------------------
DEFAULT_CHANNELS = {
    "galvo": 6,          # IN6  – sawtooth frame clock
    "stim": 4,           # IN4  – stimulus boxcar (TTL high while stim present)
    "roll": 0,           # IN0
    "pitch": 1,          # I_MTest1
    "yaw": 2,            # I_MTest2
}


class AlignmentSession:
    """One imaging session = one suite2p run + an ordered list of .abf files."""

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        suite2p_path: str | None = None,
        abf_paths: list[str] | None = None,
        channels: dict | None = None,
        neuropil_coeff: float = 0.7,
        baseline_pct: int = 8,
        verbose: bool = True,
    ):
        self.suite2p_path = Path(suite2p_path) if suite2p_path else None
        self.abf_paths = [str(p) for p in (abf_paths or [])]
        self.channels = {**DEFAULT_CHANNELS, **(channels or {})}
        self.neuropil_coeff = neuropil_coeff
        self.baseline_pct = baseline_pct
        self.verbose = verbose

        # populated as the pipeline runs
        self.abf = []            # list of per-acquisition dicts
        self.frames = []         # list of per-acquisition dicts (frame_samples, ...)
        self.stim = []           # list of per-acquisition stim event dicts
        self.suite2p = {}        # F, Fneu, iscell, spks, ops, fs, frames_per_folder
        self.dff = None          # (n_neurons, n_frames_total)
        self.light = None        # additive light-bleed trace (n_frames_total,), if corrected
        self.F0 = None           # time-varying baseline (method="maximin")
        self.dark_offset = 0.0
        self.z_dff = None
        self.deconv = None
        self.frame_offsets = None  # cumulative neural-frame offset per acquisition

    # --------------------------------------------------------------- logging
    def _log(self, *a):
        if self.verbose:
            print(*a)

    # =====================================================================
    # 1. ABF LOADING
    # =====================================================================
    @staticmethod
    def _read_abf(path):
        """Return (data[N, n_ch], fs, channel_names) using neo AxonIO."""
        if AxonIO is None:
            raise ImportError("neo is required: pip install neo")
        reader = AxonIO(str(path))
        block = reader.read_block()
        seg = block.segments[0]
        sigs = seg.analogsignals
        if len(sigs) == 1:
            data = np.asarray(sigs[0].magnitude)
        else:  # older Clampex files may split channels across signals
            data = np.concatenate([np.asarray(s.magnitude) for s in sigs], axis=1)
        fs = float(sigs[0].sampling_rate)
        names = [str(n) for n in reader.header["signal_channels"]["name"]]
        return data, fs, names

    def _chan(self, acq, key):
        """Pull one named/indexed channel's trace from a loaded acquisition."""
        ref = self.channels[key]
        names = acq["names"]
        if isinstance(ref, str) and ref in names:
            idx = names.index(ref)
        elif isinstance(ref, (int, np.integer)):
            idx = int(ref)
        else:
            raise KeyError(f"channel '{ref}' not in {names}")
        return acq["data"][:, idx]

    def load_abf(self):
        """Read every .abf; store raw arrays, sampling rate and channel names."""
        self.abf = []
        for p in self.abf_paths:
            data, fs, names = self._read_abf(p)
            self.abf.append({
                "path": str(p),
                "name": os.path.basename(p),
                "data": data,
                "fs": fs,
                "names": names,
                "n_samples": data.shape[0],
                "duration_s": data.shape[0] / fs,
            })
            self._log(f"[abf] {os.path.basename(p)}: {data.shape[1]} ch, "
                      f"{fs:.0f} Hz, {data.shape[0]/fs:.1f} s")
        return self

    # =====================================================================
    # 2. FRAME-CLOCK DETECTION  (galvo sawtooth flyback)
    # =====================================================================
    @staticmethod
    def detect_galvo_frames(galvo, fs, min_isi_s=0.04, method="peak", flyback_frac=0.25):
        """Frame sample indices from a sawtooth galvo signal.

        ``method="peak"`` (recommended): one marker at the TOP of each ramp
        (the sawtooth peak). This gives exactly one detection per scanned frame
        and, unlike flyback detection, does NOT count the park->scan startup
        transient as a spurious frame. ``method="flyback"``: one marker just
        after each flyback (ramp start).

        The galvo often scans a few WARM-UP cycles beyond the number of frames
        the scope actually saves; reconcile with the true count via
        ``detect_frames(frames_per_tseries=...)`` or ``validate()`` (suite2p).
        """
        galvo = np.asarray(galvo, dtype=float)
        amp = np.percentile(galvo, 90) - np.percentile(galvo, 10)  # robust p2p amplitude
        if method == "peak":
            pk, _ = find_peaks(galvo, prominence=0.5 * amp, distance=int(min_isi_s * fs))
            return pk
        # flyback
        nd = -np.diff(galvo)
        topk = np.sort(nd)[-500:] if nd.size > 500 else nd
        ref = np.median(topk[topk > 0]) if np.any(topk > 0) else np.max(nd)
        fly, _ = find_peaks(nd, height=flyback_frac * ref, distance=int(min_isi_s * fs))
        return fly + 1

    def detect_frames(self, min_isi_s=0.04, method="peak",
                      frames_per_tseries=None, trim_side="start"):
        """Detect imaging frames per acquisition (default: sawtooth peaks).

        If ``frames_per_tseries`` is given (the number of frames the scope saved
        per t-series, e.g. 10000), any extra warm-up cycles are trimmed to that
        count. ``trim_side="start"`` keeps the LAST N frames (drops leading
        warm-up cycles — the usual case, since the galvo parks-then-scans and the
        acquisition ends when the galvo stops); ``"end"`` keeps the first N.
        """
        self.frames = []
        for acq in self.abf:
            galvo = self._chan(acq, "galvo")
            f_samp = self.detect_galvo_frames(galvo, acq["fs"], min_isi_s, method)
            n_raw = len(f_samp); n_trim = 0
            if frames_per_tseries is not None and n_raw > frames_per_tseries:
                n_trim = n_raw - frames_per_tseries
                f_samp = f_samp[n_trim:] if trim_side == "start" else f_samp[:frames_per_tseries]
            isi = np.diff(f_samp) / acq["fs"]
            self.frames.append({
                "frame_samples": f_samp,
                "frame_times_s": f_samp / acq["fs"],
                "n_frames": len(f_samp),
                "n_detected": n_raw,
                "n_trimmed": n_trim,
                "rate_hz": len(f_samp) / acq["duration_s"],
                "isi_cv": float(np.std(isi) / np.mean(isi)) if len(isi) else np.nan,
            })
            extra = f" (detected {n_raw}, trimmed {n_trim} from {trim_side})" if n_trim else ""
            self._log(f"[frames] {acq['name']}: {len(f_samp)} frames{extra}, "
                      f"ISI CV={self.frames[-1]['isi_cv']:.3f}")
        self._update_offsets()
        return self

    def _update_offsets(self):
        counts = [f["n_frames"] for f in self.frames]
        self.frame_offsets = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(int) \
            if counts else np.array([], dtype=int)

    # =====================================================================
    # 3. STIMULUS / EVENT DETECTION  (boxcar TTL)
    # =====================================================================
    @staticmethod
    def detect_boxcars(sig, fs, thr=None, min_dur_s=0.005):
        """Return onset/offset sample indices of TTL boxcars (paired)."""
        sig = np.asarray(sig, dtype=float)
        if thr is None:
            thr = sig.min() + 0.5 * (sig.max() - sig.min())
        above = sig > thr
        onsets = np.where((~above[:-1]) & (above[1:]))[0] + 1
        offsets = np.where((above[:-1]) & (~above[1:]))[0] + 1
        on, off = [], []
        for o in onsets:
            later = offsets[offsets > o]
            if len(later):
                d = (later[0] - o) / fs
                if d >= min_dur_s:
                    on.append(o)
                    off.append(later[0])
        return np.array(on, int), np.array(off, int)

    def detect_stim(self, thr=None, min_dur_s=0.005):
        """Detect stimulus boxcars per acquisition (samples, seconds, durations)."""
        self.stim = []
        for acq in self.abf:
            sig = self._chan(acq, "stim")
            on, off = self.detect_boxcars(sig, acq["fs"], thr, min_dur_s)
            self.stim.append({
                "onset_samples": on,
                "offset_samples": off,
                "onset_s": on / acq["fs"],
                "offset_s": off / acq["fs"],
                "duration_s": (off - on) / acq["fs"],
                "n": len(on),
            })
            if len(on):
                self._log(f"[stim] {acq['name']}: {len(on)} raw pulses, "
                          f"median dur={np.median((off-on)/acq['fs'])*1000:.0f} ms")
            else:
                self._log(f"[stim] {acq['name']}: 0 pulses")
        return self

    # ------------------------------------------------ 3b. entry/present/exit
    def parse_stim(self, long_thresh_s=2.0, entry_gap_s=1.5, estimate_missing=True,
                   merge_gap_s=1.0):
        """Parse IN4 boxcars into entry / presentation / exit TRIALS.

        Each stimulus creates: a short ENTRY pulse, a long (~7-10 s) PRESENTATION
        pulse, and a short EXIT pulse. Rarely the entry is a *double-click*
        (two short pulses close together) — this does NOT affect the presentation,
        only the entry-marker count. Short pulses that surround no presentation
        (a stimulus that entered/exited but whose long pulse is missing) become an
        ABORTED trial.

        If ``estimate_missing`` is True, an aborted trial that still has an entry
        marker gets an ESTIMATED presentation onset = entry + the median
        entry->presentation delay of the good trials (``kind="estimated"``,
        ``present_estimated=True``), so it is still included in tensors.

        Two consecutive long pulses separated by less than ``merge_gap_s`` are a
        single presentation split by a TTL dropout and are merged (set 0/None to
        disable).

        Trials are anchored on the PRESENTATION onset. Sets ``self.stim_trials``:
        a per-acquisition list of trial dicts ordered in time.
        """
        if not self.stim:
            self.detect_stim()
        self.stim_trials = []
        for a, st in enumerate(self.stim):
            fs = self.abf[a]["fs"]
            on = st["onset_samples"].astype(float)
            off = st["offset_samples"].astype(float)
            # merge long pulses split by a brief TTL dropout: a presentation can't
            # directly follow a presentation (there must be an exit + entry between)
            if merge_gap_s and len(on):
                long0 = (off - on) / fs > long_thresh_s
                keep_on, keep_off, keep_long = [on[0]], [off[0]], [long0[0]]
                for i in range(1, len(on)):
                    gap = (on[i] - keep_off[-1]) / fs
                    if keep_long[-1] and long0[i] and gap < merge_gap_s:
                        self._log(f"[parse] {self.abf[a]['name']}: merged split presentation "
                                  f"at {keep_on[-1]/fs:.2f}s (gap {gap*1000:.0f} ms)")
                        keep_off[-1] = off[i]
                    else:
                        keep_on.append(on[i]); keep_off.append(off[i]); keep_long.append(long0[i])
                if len(keep_on) < len(on):
                    on, off = np.array(keep_on), np.array(keep_off)
                    st["onset_samples"], st["offset_samples"] = on.astype(int), off.astype(int)
                    st["onset_s"], st["offset_s"] = on / fs, off / fs
                    st["duration_s"] = (off - on) / fs
                    st["n"] = len(on)
            dur = (off - on) / fs
            is_long = dur > long_thresh_s
            n = len(on)
            pres = [i for i in range(n) if is_long[i]]
            # pass 1: exit = first short pulse after each presentation
            exits = {}
            for k, pi in enumerate(pres):
                hi = pres[k + 1] if k + 1 < len(pres) else n
                for j in range(pi + 1, hi):
                    if not is_long[j]:
                        exits[pi] = j
                        break
            exit_set = set(exits.values())
            # pass 2: entry = last contiguous short cluster before presentation
            trials = []; used = set()
            for k, pi in enumerate(pres):
                lo = pres[k - 1] if k > 0 else -1
                entry = []; j = pi - 1
                while j > lo and not is_long[j] and j not in exit_set:
                    if not entry or (on[entry[0]] - off[j]) / fs <= entry_gap_s:
                        entry.insert(0, j); j -= 1
                    else:
                        break
                ex = exits.get(pi)
                trials.append(self._mk_trial(a, "presentation", entry, pi, ex))
                used.update(entry); used.add(pi)
                if ex is not None:
                    used.add(ex)
            # leftover shorts -> aborted trials (one per inter-presentation gap)
            orphans = [i for i in range(n) if i not in used and not is_long[i]]
            for grp in self._group_between_presentations(orphans, pres):
                trials.append(self._mk_trial(a, "aborted", grp, None, None))
            # estimate a presentation onset for aborted trials that have an entry
            if estimate_missing:
                delays = [t["present_onset_s"] - t["entry_onset_s"] for t in trials
                          if t["kind"] == "presentation" and t["entry_onset_samples"]]
                durs = [t["present_dur_s"] for t in trials if t["kind"] == "presentation"]
                med_delay = float(np.median(delays)) if delays else np.nan
                med_dur = float(np.median(durs)) if durs else np.nan
                if not np.isnan(med_delay):
                    for t in trials:
                        if t["kind"] == "aborted" and t["entry_onset_samples"]:
                            es = int(t["entry_onset_samples"][0] + med_delay * fs)
                            t["present_onset_sample"] = es
                            t["present_onset_s"] = es / fs
                            t["present_dur_s"] = med_dur
                            t["present_offset_s"] = es / fs + med_dur
                            t["kind"] = "estimated"
                            t["present_estimated"] = True
            trials.sort(key=lambda t: t["order_time_s"])
            self.stim_trials.append(trials)
            npr = sum(t["kind"] == "presentation" for t in trials)
            nes = sum(t["kind"] == "estimated" for t in trials)
            nab = sum(t["kind"] == "aborted" for t in trials)
            ndc = sum(t["n_entry_clicks"] >= 2 for t in trials
                      if t["kind"] == "presentation")
            self._log(f"[parse] {self.abf[a]['name']}: {len(trials)} trials "
                      f"({npr} presentations, {nes} estimated, {nab} aborted, "
                      f"{ndc} double-click entries)")
        return self

    def _mk_trial(self, a, kind, entry_idx, pres_i, exit_i):
        st = self.stim[a]; fs = self.abf[a]["fs"]
        on = st["onset_samples"]; off = st["offset_samples"]
        d = {"acquisition": a, "kind": kind, "n_entry_clicks": len(entry_idx),
             "entry_onset_samples": [int(on[i]) for i in entry_idx]}
        d["entry_onset_s"] = float(on[entry_idx[0]] / fs) if entry_idx else np.nan
        if pres_i is not None:
            d["present_onset_sample"] = int(on[pres_i])
            d["present_onset_s"] = float(on[pres_i] / fs)
            d["present_offset_s"] = float(off[pres_i] / fs)
            d["present_dur_s"] = float((off[pres_i] - on[pres_i]) / fs)
        else:
            d["present_onset_sample"] = None
            d["present_onset_s"] = d["present_offset_s"] = d["present_dur_s"] = np.nan
        d["present_estimated"] = False
        d["exit_onset_s"] = float(on[exit_i] / fs) if exit_i is not None else np.nan
        # time used to order trials within an acquisition
        if entry_idx:
            d["order_time_s"] = float(on[entry_idx[0]] / fs)
        elif pres_i is not None:
            d["order_time_s"] = float(on[pres_i] / fs)
        else:
            d["order_time_s"] = np.nan
        return d

    @staticmethod
    def _group_between_presentations(orphans, pres):
        """Group orphan short pulses by which inter-presentation gap they fall in;
        each group becomes one aborted trial."""
        groups = {}
        for i in orphans:
            key = sum(1 for p in pres if p < i)
            groups.setdefault(key, []).append(i)
        return [sorted(g) for g in groups.values()]

    def attach_stim_log(self, csv_path, tseries_col="T-series",
                        stim_col="Stim", trial_col="Trial", notes_col="Notes", strict=True):
        """Attach stimulus identities from a logstimuli.csv.

        Rows are grouped by ``T-series`` (assumed to match acquisition order) and
        assigned to the parsed trials in time order. Aborted trials still consume
        a stimulus slot so the ordering stays correct. If the number of detected
        trials does not match the number of log rows for an acquisition, raises
        ValueError (labels would be shifted); pass ``strict=False`` to only warn.
        """
        import pandas as pd
        if not self.stim_trials:
            self.parse_stim()
        df = pd.read_csv(csv_path)
        labels_by_ts = {}
        for _, row in df.iterrows():
            ts = int(row[tseries_col])
            labels_by_ts.setdefault(ts, []).append({
                "stim": row[stim_col],
                "log_trial": row.get(trial_col),
                "notes": row.get(notes_col) if notes_col in df.columns else None,
            })
        ts_keys = sorted(labels_by_ts)
        self.stim_log_report = []
        for a, trials in enumerate(self.stim_trials):
            ts = ts_keys[a] if a < len(ts_keys) else None
            labels = labels_by_ts.get(ts, [])
            if len(labels) != len(trials):
                msg = (f"acq{a} (T-series {ts}): {len(trials)} detected trials vs "
                       f"{len(labels)} log rows — check alignment near aborted trials")
                self.stim_log_report.append(msg)
                if strict:
                    raise ValueError(msg + " — labels would be shifted; fix parsing or pass strict=False")
                self._log("[stimlog] !!", msg)
            for t, lab in zip(trials, labels):
                t["stim"] = lab["stim"]; t["log_trial"] = lab["log_trial"]
                t["notes"] = lab["notes"]
            for t in trials[len(labels):]:
                t.setdefault("stim", None)
            self._log(f"[stimlog] acq{a} (T-series {ts}): "
                      f"labeled {min(len(labels), len(trials))}/{len(trials)} trials")
        return self

    # =====================================================================
    # 4. SUITE2P LOADING + DFF
    # =====================================================================
    def load_suite2p(self, use_iscell=True):
        """Load F, Fneu, spks, iscell, ops from a suite2p/plane0 directory."""
        if self.suite2p_path is None:
            raise ValueError("suite2p_path not set")
        d = self.suite2p_path
        F = np.load(d / "F.npy")
        Fneu = np.load(d / "Fneu.npy")
        ops = np.load(d / "ops.npy", allow_pickle=True).item()
        iscell = np.load(d / "iscell.npy") if (d / "iscell.npy").exists() else None
        spks = np.load(d / "spks.npy") if (d / "spks.npy").exists() else None

        cells = np.where(iscell[:, 0] == 1)[0] if (use_iscell and iscell is not None) \
            else np.arange(F.shape[0])

        nframes = int(ops.get("nframes", F.shape[1]))
        fpf = ops.get("frames_per_folder", None)
        fpf = np.asarray(fpf) if fpf is not None else None
        # frames_per_folder can be degenerate (e.g. [nframes, 0, 0] when suite2p ran
        # all t-series as one folder). Derive the true per-acquisition counts from
        # the filelist so offsets / validate / detrend are correct.
        if (fpf is None or np.any(fpf == 0)) and ops.get("filelist") is not None:
            true_fpf = self._frames_per_folder_from_filelist(ops["filelist"])
            if true_fpf is not None and int(true_fpf.sum()) == nframes:
                self._log(f"[suite2p] frames_per_folder {None if fpf is None else fpf.tolist()} "
                          f"degenerate; derived {true_fpf.tolist()} from filelist")
                fpf = true_fpf
        self.suite2p = {
            "F": F[cells], "Fneu": Fneu[cells],
            "F_all": F, "Fneu_all": Fneu,          # every ROI — used to estimate light bleed
            "spks": spks[cells] if spks is not None else None,
            "iscell": iscell, "cell_idx": cells, "ops": ops,
            "fs": float(ops.get("fs", np.nan)),
            "nframes": nframes,
            "frames_per_folder": fpf,
        }
        self._log(f"[suite2p] {len(cells)} cells x {F.shape[1]} frames, "
                  f"fs={self.suite2p['fs']:.2f} Hz, "
                  f"frames_per_folder={self.suite2p['frames_per_folder']}")
        return self

    @staticmethod
    def _frames_per_folder_from_filelist(filelist):
        """Per-acquisition frame counts from suite2p ops['filelist'] (contiguous
        TSeries folder blocks)."""
        import os
        try:
            fol = [os.path.dirname(str(f).replace("\\", "/")) for f in filelist]
        except Exception:
            return None
        if not fol:
            return None
        counts = []; cur = fol[0]; c = 0
        for x in fol:
            if x == cur:
                c += 1
            else:
                counts.append(c); cur = x; c = 1
        counts.append(c)
        return np.asarray(counts)

    def _detrend_per_acquisition(self, arr, deg=3):
        """Remove a per-acquisition low-order polynomial trend (photobleaching)
        from each neuron, preserving the acquisition mean. Boundaries come from
        ``frames_per_folder`` (falls back to the whole array)."""
        T = arr.shape[1]
        fpf = self.suite2p.get("frames_per_folder")
        if fpf is not None and int(np.asarray(fpf).sum()) == T:
            e = np.concatenate([[0], np.cumsum(np.asarray(fpf))]).astype(int)
            bounds = [(int(e[i]), int(e[i + 1])) for i in range(len(e) - 1) if e[i + 1] > e[i]]
        else:
            bounds = [(0, T)]
        out = arr.astype(np.float32).copy()
        for lo, hi in bounds:
            seg = arr[:, lo:hi].astype(np.float64)
            x = np.arange(hi - lo)
            V = np.vander(x, deg + 1)
            co, _, _, _ = np.linalg.lstsq(V, seg.T, rcond=None)
            out[:, lo:hi] = (seg - (V @ co).T + seg.mean(1, keepdims=True)).astype(np.float32)
        return out

    def estimate_light(self):
        """Additive light-bleed trace (raw F units, median ~0), one value per frame.

        Stray light adds the SAME absolute fluorescence to every pixel, so it shows
        up identically in every ROI's neuropil trace, while neural signals differ
        between neuropil masks. The frame-wise median across all ROIs' neuropil
        traces (each minus its own session median) therefore tracks the light and
        rejects local neural activity. Sets and returns ``self.light``."""
        R = self.suite2p["Fneu_all"].astype(float)
        self.light = np.median(R - np.median(R, axis=1, keepdims=True), axis=0).astype(np.float32)
        return self.light

    def resolve_dark_offset(self, dark_offset):
        """Dark / PMT offset in raw F units. None -> 0 (no subtraction);
        "auto" -> 1st percentile of suite2p meanImg (the darkest pixels, i.e. no
        fluorescence); a number is used as given."""
        if dark_offset is None:
            return 0.0
        if isinstance(dark_offset, str):
            if dark_offset != "auto":
                raise ValueError('dark_offset must be None, "auto" or a number')
            return float(np.percentile(self.suite2p["ops"]["meanImg"], 1))
        return float(dark_offset)

    def _acq_bounds(self, T):
        fpf = self.suite2p.get("frames_per_folder")
        if fpf is not None and int(np.asarray(fpf).sum()) == T:
            e = np.concatenate([[0], np.cumsum(np.asarray(fpf))]).astype(int)
            return [(int(e[i]), int(e[i + 1])) for i in range(len(e) - 1) if e[i + 1] > e[i]]
        return [(0, T)]

    def _maximin_baseline(self, f, win_s=30.0, sigma_s=1.0):
        """Time-varying F0 (suite2p-style "maximin"), per acquisition: gaussian
        smooth (``sigma_s``), then a running min and a running max over ``win_s``.
        Follows monotonic trends exactly (bleaching, the fast decay after each
        acquisition starts) while ignoring positive transients shorter than
        ``win_s``. Never crosses acquisition boundaries."""
        from scipy.ndimage import gaussian_filter1d, minimum_filter1d, maximum_filter1d
        fs = self.suite2p["fs"]
        w = max(3, int(round(win_s * fs)))
        F0 = np.empty_like(f)
        for lo, hi in self._acq_bounds(f.shape[1]):
            seg = gaussian_filter1d(f[:, lo:hi], sigma_s * fs, axis=1, mode="nearest")
            seg = minimum_filter1d(seg, w, axis=1, mode="nearest")
            F0[:, lo:hi] = maximum_filter1d(seg, w, axis=1, mode="nearest")
        return F0

    def compute_dff(self, size=225, method="session", neuropil_coeff=None,
                    detrend=False, detrend_deg=3, deconvolve=False, light_correct=False,
                    dark_offset=None, baseline_win_s=30.0, baseline_sigma_s=1.0):
        """Compute dF/F and Z-scored dF/F.

        Order: (1) subtract ``dark_offset`` from F and Fneu, (2) subtract the
        additive light-bleed trace (``light_correct``), (3) neuropil subtraction
        f = F - c*Fneu, (4) baseline F0 and dF/F = (f - F0) / F0.

        method="maximin" (recommended for GRIN / drifting data): F0(t) is a
            running baseline (``_maximin_baseline``: smooth ``baseline_sigma_s``,
            min then max over ``baseline_win_s``), per acquisition. Removes
            bleaching and slow brightness drift; ``detrend`` is not needed.
        method="session" (Willmore et al. 2023): F0 = mean(f) over the whole
            session (optionally after a per-acquisition polynomial ``detrend``).
        method="rolling": legacy pydff — rolling ``baseline_pct``-percentile.

        ``dark_offset``: the detector's zero level is inside raw F; left in, it
        inflates F0 and shrinks dF/F by a cell-dependent factor. None / "auto" /
        number (see ``resolve_dark_offset``).
        ``light_correct=True`` subtracts the additive light-bleed trace
        (``estimate_light``) from F and Fneu before neuropil subtraction —
        neuropil subtraction alone leaves (1 - c) of the stray light in every cell.
        z_dff = dF/F z-scored per neuron over the session (kept for compatibility).
        """
        c = self.neuropil_coeff if neuropil_coeff is None else neuropil_coeff
        F = self.suite2p["F"].astype(float)
        Fneu = self.suite2p["Fneu"].astype(float)
        dark = self.resolve_dark_offset(dark_offset)
        if dark:
            F, Fneu = F - dark, Fneu - dark
            self._log(f"[dff] dark offset: subtracted {dark:.1f} F units from F and Fneu")
        self.dark_offset = dark
        if light_correct:
            L = self.estimate_light()
            F, Fneu = F - L, Fneu - L
            self._log(f"[dff] light-bleed correction: removed additive trace "
                      f"(1-99th pct {np.percentile(L, 1):.1f} to {np.percentile(L, 99):.1f} F units)")
        f_corr = F - c * Fneu
        if method == "maximin":
            F0 = self._maximin_baseline(f_corr, baseline_win_s, baseline_sigma_s)
            if np.any(F0 <= 0):
                bad = np.where((F0 <= 0).any(1))[0].tolist()
                self._log(f"[dff] !! F0 <= 0 for neurons {bad} — check dark offset / neuropil coeff")
            with np.errstate(divide="ignore", invalid="ignore"):
                self.dff = (f_corr - F0) / F0
            self.F0 = F0.astype(np.float32)
            self._log(f"[dff] maximin dF/F: {self.dff.shape} (F0 window {baseline_win_s:g} s, "
                      f"smooth {baseline_sigma_s:g} s, neuropil c={c})")
        else:
            if detrend:
                f_corr = self._detrend_per_acquisition(f_corr, detrend_deg)
                self._log(f"[dff] per-acquisition bleaching detrend (deg {detrend_deg})")
            if method == "session":
                F0 = f_corr.mean(1, keepdims=True)
                self.dff = (f_corr - F0) / F0
                self._log(f"[dff] Willmore session dF/F: {self.dff.shape} (neuropil c={c})")
            else:
                win = 2 * int(size) + 1
                base_corr = percentile_filter(f_corr, self.baseline_pct, size=(1, win), mode="reflect")
                base_raw = percentile_filter(F, self.baseline_pct, size=(1, win), mode="reflect")
                with np.errstate(divide="ignore", invalid="ignore"):
                    dff = (f_corr - base_corr) / base_raw
                self.dff = np.nan_to_num(dff, nan=0.0, posinf=0.0, neginf=0.0)
                self._log(f"[dff] rolling dF/F: {self.dff.shape}, win={win} frames (neuropil c={c})")
        self.z_dff = (self.dff - self.dff.mean(1, keepdims=True)) / (self.dff.std(1, keepdims=True) + 1e-12)
        if deconvolve:
            self.deconv = self._oasis_deconv(self.dff)
        return self

    def compute_activity(self, cache_path=None, size=225, method="session", neuropil_coeff=None,
                         detrend=True, detrend_deg=3, deconvolve=True, force=False,
                         light_correct=False, dark_offset=None, baseline_win_s=30.0,
                         baseline_sigma_s=1.0):
        """Load cached dff/z_dff/deconv if ``cache_path`` exists and was computed
        with the same parameters, else compute + save. Pass ``force=True`` to
        recompute. Arguments are passed to ``compute_dff``."""
        import os, pickle
        c = self.neuropil_coeff if neuropil_coeff is None else neuropil_coeff
        params = {"method": method, "neuropil_coeff": c,
                  "detrend": detrend if method != "maximin" else None,
                  "light_correct": light_correct,
                  "dark_offset": self.resolve_dark_offset(dark_offset),
                  "baseline_win_s": baseline_win_s if method == "maximin" else None,
                  "baseline_sigma_s": baseline_sigma_s if method == "maximin" else None,
                  # identify the source data so a cache from another session is never reused
                  "suite2p_path": str(self.suite2p_path),
                  "cell_idx": [int(i) for i in self.suite2p["cell_idx"]],
                  "nframes": int(self.suite2p["nframes"])}
        if cache_path and not force and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                act = pickle.load(f)
            if act.get("params") != params:
                self._log(f"[activity] cache params {act.get('params')} != requested {params} — recomputing")
            else:
                self.dff = np.asarray(act["dff"])
                self.light = None if act.get("light") is None else np.asarray(act["light"])
                self.F0 = None if act.get("F0") is None else np.asarray(act["F0"])
                self.dark_offset = params["dark_offset"]
                self.z_dff = None if act.get("z_dff") is None else np.asarray(act["z_dff"])
                self.deconv = None if act.get("deconv") is None else np.asarray(act["deconv"])
                if self.suite2p and act.get("celi") is not None:
                    self.suite2p["cell_idx"] = np.asarray(act["celi"])
                if self.suite2p and self.suite2p.get("nframes") and self.dff.shape[1] != self.suite2p["nframes"]:
                    self._log(f"[activity] !! cached dff has {self.dff.shape[1]} frames but suite2p "
                              f"nframes={self.suite2p['nframes']} — cache may be stale (use force=True)")
                self._log(f"[activity] loaded cache {os.path.basename(cache_path)} ({params}): "
                          f"dff {self.dff.shape}" + ("" if self.deconv is None else f", deconv {self.deconv.shape}"))
                return self
        self.compute_dff(size=size, method=method, neuropil_coeff=neuropil_coeff,
                         detrend=detrend, detrend_deg=detrend_deg, light_correct=light_correct,
                         dark_offset=dark_offset, baseline_win_s=baseline_win_s,
                         baseline_sigma_s=baseline_sigma_s)
        if deconvolve:
            try:
                self.deconv = self._oasis_deconv(self.dff)
                self._log(f"[deconv] {self.deconv.shape}")
            except Exception as e:
                self._log(f"[deconv] skipped: {e}")
        if cache_path:
            celi = self.suite2p.get("cell_idx") if self.suite2p else None
            self.save_pickle({"dff": self.dff, "z_dff": self.z_dff, "deconv": self.deconv,
                              "celi": celi, "params": params,
                              "light": self.light if light_correct else None,
                              "F0": getattr(self, "F0", None) if method == "maximin" else None},
                             cache_path)
        return self

    def cell_qc(self, min_f0=1.0, max_noise=None):
        """Per-cell quality table from the computed dF/F.

        A cell fails if its baseline F0 ever drops to <= ``min_f0`` raw F units
        (dF/F undefined / exploding: the neuropil-corrected signal sits at the
        dark level) or, if ``max_noise`` is set, its frame-to-frame dF/F noise
        (robust SD) exceeds it. Returns a DataFrame (neuron, roi, F0_median,
        F0_min, dff_noise, keep, reason)."""
        import pandas as pd
        F0 = self.F0 if self.F0 is not None else np.broadcast_to(
            np.nanmean(self.dff, axis=1, keepdims=True), self.dff.shape)
        noise = 1.4826 * np.nanmedian(np.abs(np.diff(self.dff, axis=1)), axis=1) / np.sqrt(2)
        rows = []
        for k in range(self.dff.shape[0]):
            reason = []
            if np.nanmin(F0[k]) <= min_f0:
                reason.append(f"F0 min {np.nanmin(F0[k]):.2f} <= {min_f0}")
            if max_noise is not None and noise[k] > max_noise:
                reason.append(f"noise {noise[k]:.3f} > {max_noise}")
            rows.append({"neuron": k, "roi": int(self.suite2p["cell_idx"][k]),
                         "F0_median": float(np.nanmedian(F0[k])), "F0_min": float(np.nanmin(F0[k])),
                         "dff_noise": float(noise[k]), "keep": not reason, "reason": "; ".join(reason)})
        return pd.DataFrame(rows)

    def apply_cell_qc(self, keep):
        """Keep only cells where ``keep`` is True, in every per-cell array
        (dff, z_dff, deconv, F0, suite2p F/Fneu/spks/cell_idx)."""
        keep = np.asarray(keep, bool)
        for name in ("dff", "z_dff", "deconv", "F0"):
            arr = getattr(self, name, None)
            if arr is not None:
                setattr(self, name, np.asarray(arr)[keep])
        for key in ("F", "Fneu", "spks", "cell_idx"):
            if self.suite2p.get(key) is not None:
                self.suite2p[key] = np.asarray(self.suite2p[key])[keep]
        dropped = int((~keep).sum())
        self._log(f"[qc] kept {int(keep.sum())}/{len(keep)} cells" + (f" (dropped {dropped})" if dropped else ""))
        return self

    @staticmethod
    def _oasis_deconv(dff):
        from oasis.functions import deconvolve, estimate_parameters
        out = np.zeros_like(dff)
        for i in range(dff.shape[0]):
            g = estimate_parameters(dff[i], 1)[0]
            c, s, b, g, lam = deconvolve(dff[i], g=g, penalty=1)
            out[i] = s
        return out

    # =====================================================================
    # 5. VALIDATION
    # =====================================================================
    def validate(self, cap=True, trim_side="start"):
        """Cross-check detected galvo frames against suite2p frame counts.

        With ``cap=True`` any surplus detected frames are trimmed to the suite2p
        ``frames_per_folder`` count. ``trim_side="start"`` drops leading warm-up
        frames (keeps the last N); ``"end"`` keeps the first N.
        """
        det = [f["n_frames"] for f in self.frames]
        total_det = int(sum(det))
        s2p = self.suite2p or {}
        fpf = s2p.get("frames_per_folder")
        fpf = None if fpf is None else np.asarray(fpf)
        nframes = s2p.get("nframes")
        report = {"detected_per_acq": det, "detected_total": total_det,
                  "suite2p_total": nframes,
                  "frames_per_folder": None if fpf is None else fpf.tolist()}

        msgs = []
        # Only trim per-acquisition if frames_per_folder actually enumerates the
        # acquisitions (one positive entry per abf). A single concatenated suite2p
        # run gives e.g. [30000, 0, 0] -> NOT per-acquisition; using it would zero
        # out acq1/acq2. In that case reconcile against the TOTAL instead.
        fpf_per_acq = (fpf is not None and len(fpf) == len(det) and bool(np.all(fpf > 0)))
        if fpf_per_acq:
            for i, (a, b) in enumerate(zip(det, fpf)):
                b = int(b)
                if a != b:
                    msgs.append(f"   acq{i}: {a} detected vs {b} suite2p frames (diff {a-b})")
                    if cap and a > b:
                        keep = slice(a - b, None) if trim_side == "start" else slice(0, b)
                        self.frames[i]["frame_samples"] = self.frames[i]["frame_samples"][keep]
                        self.frames[i]["frame_times_s"] = self.frames[i]["frame_times_s"][keep]
                        self.frames[i]["n_frames"] = b
                        msgs.append(f"      -> trimmed acq{i} to {b} (dropped {a-b} from {trim_side})")
            self._update_offsets()
            msgs = msgs or ["frame counts consistent (per-folder)"]
        else:
            if nframes is None:
                msgs.append(f"no suite2p nframes; using detected split {det} (total {total_det})")
            elif total_det == nframes:
                msgs.append(f"total detected {total_det} == suite2p nframes {nframes}; "
                            f"per-acquisition split {det} taken from the abf frame clock "
                            f"(frames_per_folder={report['frames_per_folder']} is a single "
                            f"concatenated folder, not per-acquisition)")
            else:
                msgs.append(f"!! total detected {total_det} != suite2p nframes {nframes} "
                            f"(diff {total_det - nframes}). Set frames_per_tseries so each "
                            f"acquisition matches, or check the galvo channel.")
            self._update_offsets()   # keep per-acq split from the detected abf frames
        report["messages"] = msgs
        for m in msgs:
            self._log("[validate]", m)
        return report

    # =====================================================================
    # 6. EVENT -> FRAME MAPPING + CONTINUOUS RESAMPLING
    # =====================================================================
    def event_to_global_frame(self, acq_idx, sample, mode="next"):
        """Map an abf sample (in acquisition ``acq_idx``) to a global frame index."""
        ft = self.frames[acq_idx]["frame_samples"]
        if mode == "next":
            j = int(np.searchsorted(ft, sample, side="left"))
            j = min(j, len(ft) - 1)
        else:  # nearest
            j = int(np.argmin(np.abs(ft - sample)))
        return self.frame_offsets[acq_idx] + j

    def ball_on_frames(self, acq_idx):
        """Resample roll/pitch/yaw onto this acquisition's frame grid (per-frame mean)."""
        acq = self.abf[acq_idx]
        f_samp = self.frames[acq_idx]["frame_samples"]
        edges = np.concatenate([f_samp, [acq["n_samples"]]])
        out = {}
        for key in ("roll", "pitch", "yaw"):
            sig = self._chan(acq, key)
            out[key] = np.array([sig[edges[i]:edges[i + 1]].mean()
                                 for i in range(len(f_samp))])
        return out

    def ball_session(self):
        """Concatenate ball signals across acquisitions into session-length arrays.

        Returns dict with 'roll', 'pitch', 'yaw' each of shape (n_total_frames,),
        aligned to the neural data (dff / z_dff / deconv).
        """
        arrs = {k: [] for k in ("roll", "pitch", "yaw")}
        for a in range(len(self.abf)):
            b = self.ball_on_frames(a)
            for k in arrs:
                arrs[k].append(b[k])
        return {k: np.concatenate(v) for k, v in arrs.items()}

    # =====================================================================
    # 7. TRIAL-BY-TRIAL DICTS  (one trial = one stimulus presentation)
    # =====================================================================
    def build_trials(self, n_frames=45, pre_frames=0, signals=("dff", "z_dff", "deconv")):
        """One entry per stimulus PRESENTATION, anchored on the presentation frame.

        Neural data is a window that starts at the presentation-onset frame and
        runs FORWARD ``n_frames`` (optionally with ``pre_frames`` before onset).
        Each trial carries its entry/exit times, double-click count, and (if a
        log was attached) its ``stim`` label. Aborted trials are skipped here but
        remain in ``self.stim_trials``.
        """
        if not self.stim_trials:
            self.parse_stim()
        if self.dff is None and self.suite2p and any(s in signals for s in ("dff", "z_dff")):
            self._log("[trials] dff not computed yet — running compute_dff()")
            self.compute_dff()
        trials = {}; t = 0
        for a, tr_list in enumerate(self.stim_trials):
            ball = self.ball_on_frames(a)
            acq_bounds = (int(self.frame_offsets[a]),
                          int(self.frame_offsets[a]) + self.frames[a]["n_frames"])
            for tr in tr_list:
                if tr["kind"] not in ("presentation", "estimated"):
                    continue
                t += 1
                onset_g = self.event_to_global_frame(a, tr["present_onset_sample"])
                onset_local = onset_g - self.frame_offsets[a]
                d = dict(tr)
                d["onset_frame"] = int(onset_g)
                d["ball"] = {kk: vv[max(0, onset_local):onset_local + n_frames]
                             for kk, vv in ball.items()}
                for sig_name in signals:
                    sig = getattr(self, sig_name)
                    if sig is None:
                        continue
                    d[sig_name] = self._window_forward(sig, onset_g, n_frames, pre_frames, bounds=acq_bounds)
                trials[f"trial_{t}"] = d
        self.trials = trials
        self._log(f"[trials] built {len(trials)} presentation trials "
                  f"(window = {pre_frames} pre + {n_frames} forward frames)")
        return trials

    def _window_forward(self, sig, onset_frame, n_frames, pre_frames=0, bounds=None):
        """Forward window [onset-pre, onset+n_frames); edges padded with NaN.

        ``bounds=(lo, hi)`` restricts the window to one acquisition's frame range
        so it never bleeds into a neighbouring acquisition (frames outside the
        acquisition are left NaN).
        """
        n = sig.shape[1]
        clo, chi = (0, n) if bounds is None else (max(0, bounds[0]), min(n, bounds[1]))
        lo, hi = onset_frame - pre_frames, onset_frame + n_frames
        out = np.full((sig.shape[0], pre_frames + n_frames), np.nan)
        sa, sb = max(lo, clo), min(hi, chi)
        if sb > sa:
            out[:, (sa - lo):(sb - lo)] = sig[:, sa:sb]
        return out

    # =====================================================================
    # 8. EVENT-LOCKED TENSOR  (trials x neurons x time), aligned to presentation
    # =====================================================================
    def epoch(self, signal="dff", n_frames=45, pre_frames=0):
        """Stack forward windows from each PRESENTATION onset.

        Returns ``(n_trials, n_neurons, pre_frames + n_frames)`` and stores the
        per-trial stimulus labels in ``self.epoch_labels``.
        """
        sig = getattr(self, signal, None)
        if sig is None and signal in ("dff", "z_dff") and self.suite2p:
            self._log(f"[epoch] '{signal}' not computed yet — running compute_dff()")
            self.compute_dff()
            sig = getattr(self, signal, None)
        if sig is None:
            raise ValueError(
                f"signal '{signal}' not computed. Run s.load_suite2p() then "
                f"s.compute_dff() on THIS session object (re-running the config "
                f"cell that builds `s` resets it).")
        if not self.stim_trials:
            self.parse_stim()
        mats = []; labels = []; anchors = []
        for a, tr_list in enumerate(self.stim_trials):
            acq_bounds = (int(self.frame_offsets[a]),
                          int(self.frame_offsets[a]) + self.frames[a]["n_frames"])
            for tr in tr_list:
                if tr["kind"] not in ("presentation", "estimated"):
                    continue
                f = self.event_to_global_frame(a, tr["present_onset_sample"])
                anchors.append(int(f))
                mats.append(self._window_forward(sig, f, n_frames, pre_frames, bounds=acq_bounds))
                labels.append(tr.get("stim"))
        if anchors and len(set(anchors)) < 0.9 * len(anchors):
            self._log(f"[epoch] !! only {len(set(anchors))}/{len(anchors)} unique onset frames — "
                      f"trials are collapsing to the same frame (check validate / frames_per_folder / "
                      f"that each acquisition has frame times)")
        self.epoch_labels = labels
        return np.stack(mats, axis=0) if mats else np.empty((0,))

    # =====================================================================
    # 9. SAVE / SUMMARY / QC PLOTS
    # =====================================================================
    def save_pickle(self, obj, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        self._log(f"[save] {path}")

    def trials_table(self, include_aborted=True):
        """One row per trial with all timing/label info (no neural arrays).

        ``tensor_index`` is the 0-based row in the ``epoch`` tensor for
        presentation/estimated trials (NaN for aborted); rows are in tensor order,
        so ``df[df.tensor_index.notna()]`` lines up 1:1 with ``X = self.epoch(...)``.

        Columns: tensor_index, acquisition, log_trial, stim, kind,
        present_estimated, present_onset_s, present_offset_s, present_dur_s,
        onset_frame, entry_onset_s, exit_onset_s, n_entry_clicks, notes.
        """
        import pandas as pd
        rows = []; ti = 0
        for a, tr_list in enumerate(self.stim_trials):
            for tr in tr_list:
                in_tensor = tr["kind"] in ("presentation", "estimated")
                if not in_tensor and not include_aborted:
                    continue
                onset_frame = np.nan
                if in_tensor and self.frames:
                    onset_frame = int(self.event_to_global_frame(a, tr["present_onset_sample"]))
                rows.append({
                    "tensor_index": ti if in_tensor else np.nan,
                    "acquisition": a,
                    "log_trial": tr.get("log_trial"),
                    "stim": tr.get("stim"),
                    "kind": tr["kind"],
                    "present_estimated": tr.get("present_estimated", False),
                    "present_onset_s": tr.get("present_onset_s", np.nan),
                    "present_offset_s": tr.get("present_offset_s", np.nan),
                    "present_dur_s": tr.get("present_dur_s", np.nan),
                    "onset_frame": onset_frame,
                    "entry_onset_s": tr.get("entry_onset_s", np.nan),
                    "exit_onset_s": tr.get("exit_onset_s", np.nan),
                    "n_entry_clicks": tr.get("n_entry_clicks"),
                    "notes": tr.get("notes"),
                })
                if in_tensor:
                    ti += 1
        return pd.DataFrame(rows)

    def save_trials_csv(self, path, include_aborted=True):
        """Write the per-trial info table to CSV; returns the DataFrame."""
        df = self.trials_table(include_aborted=include_aborted)
        df.to_csv(path, index=False)
        self._log(f"[save] {path} ({len(df)} rows)")
        return df

    @staticmethod
    def neuron_regions(roi_indices_by_region, cell_idx, default="none"):
        """Map anatomical-region labels onto the tensor's neuron axis.

        roi_indices_by_region : segmentation-notebook dict {region: ROI indices}.
        cell_idx : self.suite2p['cell_idx'] (the neuron order of dff / the tensor).
        Returns region labels aligned to cell_idx; neurons in no region -> default.
            regions = s.neuron_regions(roi_dict, s.suite2p['cell_idx'])
            X_v1 = X[:, regions == 'V1', :]
        """
        lookup = {}
        for region, idxs in roi_indices_by_region.items():
            for i in np.asarray(idxs).ravel():
                lookup[int(i)] = region
        return np.array([lookup.get(int(c), default) for c in cell_idx])

    def summary(self):
        """Human-readable one-shot status of the loaded session."""
        print("=" * 60)
        print(f"acquisitions : {len(self.abf)}")
        for i, acq in enumerate(self.abf):
            nf = self.frames[i]["n_frames"] if i < len(self.frames) else "-"
            if getattr(self, "stim_trials", None) and i < len(self.stim_trials):
                tr = self.stim_trials[i]
                npr = sum(t["kind"] == "presentation" for t in tr)
                nes = sum(t["kind"] == "estimated" for t in tr)
                nab = sum(t["kind"] == "aborted" for t in tr)
                s_str = f"trials={len(tr)} (pres={npr}, est={nes}, aborted={nab})"
            else:
                s_str = f"pulses={self.stim[i]['n']}" if i < len(self.stim) else ""
            print(f"  acq{i}: {acq['name']:<24} {acq['duration_s']:7.1f}s  "
                  f"frames={nf}  {s_str}")
        if self.suite2p:
            print(f"suite2p      : {len(self.suite2p.get('cell_idx', []))} cells x "
                  f"{self.suite2p.get('nframes')} frames @ {self.suite2p.get('fs'):.2f} Hz")
        if self.dff is not None:
            print(f"dff          : {self.dff.shape}")
        print("=" * 60)

    def plot_frame_qc(self, acq=0, window_s=(100, 103), ax=None):
        """Overlay detected frames on the galvo signal for a short window."""
        import matplotlib.pyplot as plt
        a = self.abf[acq]; fs = a["fs"]
        g = self._chan(a, "galvo")
        i0, i1 = int(window_s[0] * fs), int(window_s[1] * fs)
        fr = self.frames[acq]["frame_samples"]
        fr = fr[(fr >= i0) & (fr < i1)]
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 3))
        ax.plot(np.arange(i0, i1) / fs, g[i0:i1], lw=0.6, label="galvo")
        ax.plot(fr / fs, g[fr], "r.", ms=6, label="detected frames")
        ax.set(xlabel="time (s)", ylabel="galvo (V)",
               title=f"acq{acq} frame detection ({self.frames[acq]['n_frames']} frames, "
                     f"{self.frames[acq]['rate_hz']:.2f} Hz)")
        ax.legend(loc="upper right", fontsize=8)
        return ax

    def plot_stim_raster(self, ax=None):
        """One row per acquisition; bars mark stimulus on-periods."""
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 2 + 0.4 * len(self.stim)))
        for a, st in enumerate(self.stim):
            for on, off in zip(st["onset_s"], st["offset_s"]):
                ax.plot([on, off], [a, a], lw=4, solid_capstyle="butt")
        ax.set(xlabel="time within acquisition (s)", ylabel="acquisition",
               yticks=range(len(self.stim)), title="stimulus boxcars")
        return ax

    def plot_trials(self, acq=0, window_s=None, ax=None):
        """Show IN4 with parsed entry (green) / presentation (blue) / exit (red)
        and the stimulus label per trial."""
        import matplotlib.pyplot as plt
        a = self.abf[acq]; fs = a["fs"]; sig = self._chan(a, "stim")
        if window_s is None:
            window_s = (0, a["duration_s"])
        i0, i1 = int(window_s[0] * fs), int(window_s[1] * fs)
        if ax is None:
            _, ax = plt.subplots(figsize=(13, 3))
        ax.plot(np.arange(i0, i1) / fs, sig[i0:i1], lw=0.5, color="0.6")
        for tr in self.stim_trials[acq]:
            if not np.isnan(tr["present_onset_s"]):
                est = tr.get("present_estimated", False)
                col = "tab:orange" if est else "tab:blue"
                ax.axvspan(tr["present_onset_s"], tr["present_offset_s"],
                           color=col, alpha=0.15)
                ax.axvline(tr["present_onset_s"], color=col, lw=0.9,
                           ls="--" if est else "-")
            if not np.isnan(tr["entry_onset_s"]):
                ax.axvline(tr["entry_onset_s"], color="tab:green", lw=0.7, ls=":")
            if not np.isnan(tr["exit_onset_s"]):
                ax.axvline(tr["exit_onset_s"], color="tab:red", lw=0.7, ls=":")
            if tr.get("stim") is not None and window_s[0] <= tr["order_time_s"] <= window_s[1]:
                suffix = {"aborted": "*", "estimated": "~"}.get(tr["kind"], "")
                ax.text(tr["order_time_s"], sig.max() * 1.03, str(tr["stim"]) + suffix,
                        fontsize=6, rotation=90, va="bottom")
        ax.set(xlabel="time (s)", ylabel="IN4 (V)", xlim=window_s,
               title=f"acq{acq} trials — presentation=blue, estimated=orange(dashed), "
                     f"entry=green, exit=red  (~ estimated, * aborted-no-entry)")
        return ax

    def plot_psth(self, X, pre_s=1.0, post_s=3.0, ax=None, label=None):
        """Population-mean PSTH from an epoch tensor (trials x neurons x time)."""
        import matplotlib.pyplot as plt
        fs = self.suite2p["fs"]
        t = (np.arange(X.shape[2]) - int(round(pre_s * fs))) / fs
        m = np.nanmean(X, axis=(0, 1))
        sem = np.nanstd(np.nanmean(X, axis=1), axis=0) / np.sqrt(X.shape[0])
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))
        ax.plot(t, m, label=label)
        ax.fill_between(t, m - sem, m + sem, alpha=0.25)
        ax.axvline(0, color="k", ls="--", lw=0.8)
        ax.set(xlabel="time from stimulus onset (s)", ylabel="mean activity",
               title="population PSTH")
        if label:
            ax.legend()
        return ax


def offset_local_clip(vv, offset_g, off_a):
    """Helper: clip an offset frame to the length of a per-acq ball vector."""
    return min(len(vv), offset_g - off_a)
