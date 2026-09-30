"""
amygdala_analysis.py
====================
Stimulus responses and single-neuron selectivity for the SocialPresentation
GRIN (amygdala) sessions: few neurons, many repeats of 8 stimuli.

Inputs (written by ``processing_main_abf.ipynb``)
------------------------------------------------
tensor_<signal>_raw.npy (n_trials, n_neurons, n_frames)  presentation-locked,
                     NOT baseline-corrected (baseline is chosen here)
tensor_labels.npy    (n_trials,)                      stimulus per trial
tensor_meta.npy      dict(pre_frames, n_frames_forward, fs, baseline_win_s)
trials_info.csv      one row per trial (block = ``log_trial``)
session_<signal>.npy (n_neurons, n_frames_total)      only for the baseline-by-position check
suite2p plane0/      ops.npy / stat.npy / iscell.npy  only for the FOV map

Design
------
* All analysis choices (baseline window, response window, alpha, shuffles,
  classification rule) are set once via ``set_windows`` / arguments, so they can
  be fixed in the notebook's config cell before any result is looked at.
* Traces / PSTHs (``Xb``) are baseline-corrected per trial to the BASELINE
  window (default -2 to 0 s: the flat stretch just before onset).
* Statistics use ``resp`` = mean RESPONSE-window activity minus the same
  trial's BASELINE-window mean (``trial_baseline=True``, default). This is
  required here: raw activity falls steadily across each block (and decays after
  each acquisition starts), so with the fixed stimulus order, raw levels differ
  by block position; the per-trial baseline removes that.
* Responsiveness = stimulus vs the EMPTY cage, paired by block (Wilcoxon), so
  the non-specific rise after onset (seen for Empty too) doesn't count.
* Selectivity uses ROC AUC (A vs B trials) with a label-permutation null that
  shuffles labels WITHIN each block, so slow drift across the session can't
  create selectivity. BH-FDR is applied across neurons within each contrast.

Caveat: every block presents the stimuli in the same fixed order, so stimulus
identity is confounded with position in the block. The control analyses
(Male1 vs Male2, split-half, drift, baseline carry-over) test pieces of this but
cannot remove it.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import matplotlib.patheffects as pe
from scipy import stats
from scipy.ndimage import gaussian_filter1d

# --- experiment-specific stimulus table (edit here if names change) ----------
SIGNAL_LABELS = {"dff": "ΔF/F", "zdff": "z-ΔF/F", "deconv": "deconv. activity"}

STIM_ORDER = ["Empty", "FamFood", "NewFood", "NewObject",
              "Male1", "Female1", "Male2", "Female2"]
STIM_LABELS = {"Empty": "empty", "FamFood": "familiar food", "NewFood": "novel food",
               "NewObject": "novel object", "Male1": "male 1", "Female1": "female 1",
               "Male2": "male 2", "Female2": "female 2"}
STIM_COLORS = {"Empty": "#8C8C8C", "FamFood": "#F7941D", "NewFood": "#B5541B",
               "NewObject": "#2E9E6A", "Male1": "#27AAE1", "Male2": "#1B5E9A",
               "Female1": "#EC008C", "Female2": "#9C0060"}

CATEGORIES = {"control": ["Empty"],
              "food":    ["FamFood", "NewFood"],
              "object":  ["NewObject"],
              "social":  ["Male1", "Female1", "Male2", "Female2"]}
CATEGORY_COLORS = {"control": "#8C8C8C", "food": "#F7941D",
                   "object": "#2E9E6A", "social": "#7B3FA0"}

# name -> (group A, group B); AUC > 0.5 means higher activity for A
CONTRASTS = {
    # category preference (used for classification)
    "social_vs_rest":   (CATEGORIES["social"], CATEGORIES["food"] + CATEGORIES["object"]),
    "food_vs_rest":     (CATEGORIES["food"], CATEGORIES["social"] + CATEGORIES["object"]),
    "object_vs_rest":   (CATEGORIES["object"], CATEGORIES["social"] + CATEGORIES["food"]),
    # pairwise category contrasts
    "social_vs_food":   (CATEGORIES["social"], CATEGORIES["food"]),
    "social_vs_object": (CATEGORIES["social"], CATEGORIES["object"]),
    "food_vs_object":   (CATEGORIES["food"], CATEGORIES["object"]),
    # vs the empty-cage control
    "social_vs_empty":  (CATEGORIES["social"], ["Empty"]),
    "food_vs_empty":    (CATEGORIES["food"], ["Empty"]),
    "object_vs_empty":  (CATEGORIES["object"], ["Empty"]),
    # within category
    "male_vs_female":   (["Male1", "Male2"], ["Female1", "Female2"]),
    "famfood_vs_newfood": (["FamFood"], ["NewFood"]),
    # identity controls (same category, different animal / block position)
    "male1_vs_male2":   (["Male1"], ["Male2"]),
    "female1_vs_female2": (["Female1"], ["Female2"]),
}
CLASS_CONTRASTS = {"social": "social_vs_rest", "food": "food_vs_rest",
                   "object": "object_vs_rest"}
CLASS_COLORS = {**{k: CATEGORY_COLORS[k] for k in CLASS_CONTRASTS},
                "non-selective": "#D9D9D9", "non-responsive": "none"}


def fdr_bh(p):
    """Benjamini-Hochberg q-values (NaNs are passed through)."""
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if not ok.any():
        return q
    pv = p[ok]
    order = np.argsort(pv)
    n = len(pv)
    ranked = pv[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q


def _smooth(x, sigma):
    return gaussian_filter1d(x, sigma=sigma, axis=-1) if sigma and sigma > 0 else x


def _nanmean_sem(x, axis=0):
    n = np.sum(~np.isnan(x), axis=axis)
    m = np.nanmean(x, axis=axis)
    s = np.nanstd(x, axis=axis, ddof=1) / np.sqrt(np.maximum(n, 1))
    return m, s


class StimSession:
    """One imaging session: presentation-locked tensor + trial table."""

    def __init__(self, X, labels, trials, fs, pre_frames, data_dir=None, signal="dff"):
        self.X = np.asarray(X, float)
        self.labels = np.asarray(labels).astype(str)
        self.trials = trials.reset_index(drop=True)
        self.fs = float(fs)
        self.pre = int(pre_frames)
        self.data_dir = data_dir
        self.signal = signal
        self.ylab = SIGNAL_LABELS.get(signal, signal)
        self.n_trials, self.n_neurons, self.n_frames = self.X.shape
        self.time = (np.arange(self.n_frames) - self.pre) / self.fs   # 0 = onset
        self.block = self.trials["log_trial"].to_numpy()
        self.stims = [s for s in STIM_ORDER if s in set(self.labels)]
        extra = sorted(set(self.labels) - set(STIM_ORDER))
        if extra:
            raise ValueError(f"stimuli not in STIM_ORDER: {extra} — add them to the stimulus table")
        # median event times relative to presentation onset (for plot markers)
        # (entry marker only if most trials have an entry pulse — 09-03 has none)
        ent = (self.trials["entry_onset_s"] - self.trials["present_onset_s"]).to_numpy(float)
        self.entry_t = float(np.nanmedian(ent)) if np.mean(~np.isnan(ent)) >= 0.5 else None
        self.offset_t = float(np.nanmedian(self.trials["present_dur_s"]))
        self.fov = None
        self.plot_win = (-2.0, 8.0)
        self.set_windows()

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, data_dir, signal="dff"):
        path = os.path.join(data_dir, f"tensor_{signal}_raw.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing — re-run processing_main_abf.ipynb "
                                    "(it now saves uncorrected tensors)")
        X = np.load(path)
        labels = np.load(os.path.join(data_dir, "tensor_labels.npy"), allow_pickle=True)
        meta = np.load(os.path.join(data_dir, "tensor_meta.npy"), allow_pickle=True).item()
        trials = pd.read_csv(os.path.join(data_dir, "trials_info.csv"))
        trials = trials[trials["tensor_index"].notna()].sort_values("tensor_index")
        if len(trials) != len(labels) or not np.array_equal(trials["stim"].astype(str).to_numpy(),
                                                             labels.astype(str)):
            raise ValueError("trials_info.csv and tensor_labels.npy disagree — re-run the alignment notebook")
        if X.shape[2] != meta["pre_frames"] + meta["n_frames_forward"]:
            raise ValueError("tensor length does not match tensor_meta.npy")
        obj = cls(X, labels, trials, meta["fs"], meta["pre_frames"], data_dir=data_dir, signal=signal)
        obj.meta = meta
        return obj

    # --------------------------------------------------------------- windows
    def _frames(self, win):
        a = self.pre + int(round(win[0] * self.fs))
        b = self.pre + int(round(win[1] * self.fs))
        return max(0, a), min(self.n_frames, b)

    def set_windows(self, baseline=(-2.0, 0.0), response=(0.0, 4.0), trial_baseline=True):
        """Set baseline / response windows (s rel. presentation onset).

        ``Xb``   : tensor baseline-corrected per trial (used for all traces).
        ``resp`` : (n_trials, n_neurons) mean response-window activity used for
                   statistics — baseline-subtracted per trial if
                   ``trial_baseline`` is True (default), raw if False (raw levels
                   carry the within-block trend; use only for comparison).
        Clears cached stats."""
        self.baseline_win, self.response_win = tuple(baseline), tuple(response)
        self.trial_baseline = bool(trial_baseline)
        ba, bb = self._frames(baseline)
        ra, rb = self._frames(response)
        self.Xb = self.X - np.nanmean(self.X[:, :, ba:bb], axis=2, keepdims=True)
        src = self.Xb if trial_baseline else self.X
        self.resp = np.nanmean(src[:, :, ra:rb], axis=2)
        self.valid = ~np.isnan(self.resp).any(axis=1)          # trials usable for stats
        self._cache = {}
        return self

    # -------------------------------------------------------------- helpers
    def trials_of(self, stims):
        stims = [stims] if isinstance(stims, str) else list(stims)
        return np.isin(self.labels, stims)

    def _group(self, g):
        """A group name -> list of stimuli (category name or stimulus name)."""
        return CATEGORIES.get(g, [g]) if isinstance(g, str) else list(g)

    def neuron_mean(self, stims):
        """Trial-averaged Xb per neuron: (n_neurons, n_frames)."""
        return np.nanmean(self.Xb[self.trials_of(self._group(stims))], axis=0)

    def mean_resp_matrix(self, stims=None):
        """(n_neurons, n_stims) mean window response."""
        stims = stims or self.stims
        return np.column_stack([np.nanmean(self.resp[self.trials_of(s) & self.valid], axis=0)
                                for s in stims])

    def rel_matrix(self, ref="Empty"):
        """(n_neurons, n_stims-without-ref) mean response minus the mean response
        to ``ref``. Removes each neuron's offset so neurons are comparable."""
        stims = [s for s in self.stims if s != ref]
        return self.mean_resp_matrix(stims) - self.mean_resp_matrix([ref]), stims

    def set_plot_window(self, win=(-2.0, 8.0)):
        """Time range (s rel. onset) shown on every time-axis plot. Display only:
        smoothing is done on the full trace before cropping, and statistics
        always use the full tensor."""
        self.plot_win = tuple(win)
        return self

    def _ps(self):
        """(frame slice, time vector) for the plot window."""
        a = max(0, self.pre + int(round(self.plot_win[0] * self.fs)))
        b = min(self.n_frames, self.pre + int(round(self.plot_win[1] * self.fs)) + 1)
        return slice(a, b), self.time[a:b]

    def _mark_events(self, ax, lw=0.6):
        lo, hi = self.plot_win
        ax.axvspan(max(lo, self.response_win[0]), min(hi, self.response_win[1]),
                   color="0.92", zorder=0, lw=0)
        for x, c, ls in ((self.entry_t, "0.6", ":"), (0, "k", "--"), (self.offset_t, "0.6", "--")):
            if x is not None and lo <= x <= hi:
                ax.axvline(x, color=c, ls=ls, lw=lw)
        ax.axhline(0, color="0.75", lw=0.4)
        ax.set_xlim(lo, hi)

    # =================================================== 1. population PSTHs
    def plot_population_psth(self, groups=None, colors=None, smooth_sigma=2, ax=None,
                             figsize=(3, 2), dpi=200, legend=True):
        """Mean over neurons of each neuron's trial-averaged trace, ±SEM across
        neurons. ``groups``: stimulus names and/or category names."""
        groups = groups or self.stims
        colors = colors or {**STIM_COLORS, **CATEGORY_COLORS}
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        self._mark_events(ax)
        for g in groups:
            per_neuron = _smooth(self.neuron_mean(g), smooth_sigma)   # (neurons, frames)
            sl, t = self._ps()
            m, s = _nanmean_sem(per_neuron[:, sl], axis=0)
            n_tr = int(self.trials_of(self._group(g)).sum())
            ax.plot(t, m, color=colors.get(g, None), lw=1.1,
                    label=f"{STIM_LABELS.get(g, g)} ({n_tr})")
            ax.fill_between(t, m - s, m + s, color=colors.get(g, None), alpha=0.18, lw=0)
        ax.set(xlabel="time from presentation onset (s)", ylabel=f"{self.ylab} (baseline-sub.)")
        if legend:
            ax.legend(frameon=False, fontsize="small", loc="upper left", bbox_to_anchor=(1, 1))
        return ax.figure

    def plot_heatmaps(self, stims=None, sort_by="all", clim=None, smooth_sigma=1,
                      cmap="RdBu_r", figsize=None, dpi=200):
        """Neurons x time per stimulus. All panels share one neuron order, sorted
        by mean response-window activity to ``sort_by`` (a stimulus, a category,
        or "all")."""
        stims = stims or self.stims
        ref = self.stims if sort_by == "all" else self._group(sort_by)
        ra, rb = self._frames(self.response_win)
        order = np.argsort(-np.nanmean(self.neuron_mean(ref)[:, ra:rb], axis=1))
        sl, t = self._ps()
        mats = [_smooth(self.neuron_mean(s), smooth_sigma)[order][:, sl] for s in stims]
        if clim is None:
            clim = np.nanpercentile(np.abs(np.concatenate(mats)), 98)
        figsize = figsize or (1.3 * len(stims) + 0.6, 2.2)
        fig, axs = plt.subplots(1, len(stims), figsize=figsize, dpi=dpi, sharey=True)
        axs = np.atleast_1d(axs)
        for ax, s, M in zip(axs, stims, mats):
            im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=-clim, vmax=clim, interpolation="none",
                           extent=[t[0], t[-1], len(order) - 0.5, -0.5])
            ax.axvline(0, color="k", ls="--", lw=0.5)
            ax.set_title(STIM_LABELS.get(s, s), color=STIM_COLORS.get(s, "k"))
            ax.set_xlabel("time (s)")
        axs[0].set_yticks(range(len(order)))
        axs[0].set_yticklabels(order, fontsize="x-small")
        axs[0].set_ylabel(f"neuron (sorted by {sort_by})")
        fig.colorbar(im, ax=axs, shrink=0.8, pad=0.01, label=self.ylab)
        return fig

    # ================================================== 2. single neurons
    def plot_neuron_grid(self, groups=None, neurons=None, ncols=4, smooth_sigma=2,
                         sharey=False, figsize=None, dpi=200, titles=None):
        """One panel per neuron: trial-averaged trace per stimulus/category ±SEM
        across trials. ``titles``: optional {neuron: extra text}."""
        groups = groups or self.stims
        neurons = list(range(self.n_neurons)) if neurons is None else list(neurons)
        colors = {**STIM_COLORS, **CATEGORY_COLORS}
        nrows = int(np.ceil(len(neurons) / ncols))
        figsize = figsize or (1.7 * ncols + 1.0, 1.3 * nrows)
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi, sharex=True,
                                sharey=sharey, squeeze=False)
        for k, ax in enumerate(axs.flat):
            if k >= len(neurons):
                ax.axis("off")
                continue
            nid = neurons[k]
            self._mark_events(ax, lw=0.4)
            for g in groups:
                tr = self.Xb[self.trials_of(self._group(g)), nid, :]
                sl, t = self._ps()
                m, s = _nanmean_sem(_smooth(tr, smooth_sigma)[:, sl], axis=0)
                ax.plot(t, m, color=colors.get(g), lw=0.8, label=STIM_LABELS.get(g, g))
                ax.fill_between(t, m - s, m + s, color=colors.get(g), alpha=0.15, lw=0)
            extra = "" if not titles else f"  {titles.get(nid, '')}"
            ax.set_title(f"n{nid}{extra}", fontsize="small", pad=2)
            if k % ncols == 0:
                ax.set_ylabel(self.ylab)
            if k >= (nrows - 1) * ncols:
                ax.set_xlabel("time (s)")
        handles, labs = axs.flat[0].get_legend_handles_labels()
        fig.legend(handles, labs, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
        fig.tight_layout()
        return fig

    def plot_neuron_raster(self, nid, stims=None, clim=None, smooth_sigma=1, ax=None,
                           figsize=(2.4, 3), dpi=200, cmap="RdBu_r"):
        """Single-trial heatmap for one neuron, trials grouped by stimulus
        (in session order within each group)."""
        stims = stims or self.stims
        idx = np.concatenate([np.where(self.trials_of(s))[0] for s in stims])
        sl, t = self._ps()
        M = _smooth(self.Xb[idx, nid, :], smooth_sigma)[:, sl]
        clim = clim or np.nanpercentile(np.abs(M), 98)
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.imshow(M, aspect="auto", cmap=cmap, vmin=-clim, vmax=clim, interpolation="none",
                  extent=[t[0], t[-1], len(idx), 0])
        ax.axvline(0, color="k", ls="--", lw=0.5)
        edges = np.cumsum([0] + [int(self.trials_of(s).sum()) for s in stims])
        for s, a, b in zip(stims, edges[:-1], edges[1:]):
            ax.axhline(b, color="k", lw=0.4)
            ax.plot([t[0]] * 2, [a, b], color=STIM_COLORS[s], lw=4,
                    solid_capstyle="butt", clip_on=False)
        ax.set_yticks((edges[:-1] + edges[1:]) / 2)
        ax.set_yticklabels([STIM_LABELS.get(s, s) for s in stims], fontsize="x-small")
        ax.tick_params(axis="y", length=0, pad=6)
        ax.set(xlabel="time (s)", title=f"neuron {nid}")
        return ax.figure

    # ================================================ 3. selectivity stats
    def responsiveness(self, alpha=0.05, reference="Empty"):
        """Is each neuron driven by each stimulus? BH-FDR across all tests.

        reference="Empty"   : stimulus vs empty cage, paired by block (per-block
                              means; Wilcoxon signed-rank on the differences).
                              Default — the post-onset rise seen for Empty too
                              (cage moving, recovery from the pre-onset dip) is
                              not counted as a stimulus response.
        reference="baseline": window minus per-trial baseline vs 0 (Wilcoxon).
        Returns a long DataFrame (neuron, stim, mean, n, p, q, sig, sign); ``mean``
        is the mean difference from the reference."""
        rows = []
        if reference == "baseline":
            ba, bb = self._frames(self.baseline_win)
            ra, rb = self._frames(self.response_win)
            d = np.nanmean(self.Xb[:, :, ra:rb], axis=2)
            for st in self.stims:
                tr = self.trials_of(st) & self.valid
                for n in range(self.n_neurons):
                    x = d[tr, n]
                    p = stats.wilcoxon(x).pvalue if np.any(x != 0) else 1.0
                    rows.append({"neuron": n, "stim": st, "mean": float(np.mean(x)),
                                 "n": int(tr.sum()), "p": float(p)})
        else:
            blocks = np.unique(self.block)
            def block_means(stim):          # (n_blocks, n_neurons), NaN where absent
                out = np.full((len(blocks), self.n_neurons), np.nan)
                for i, b in enumerate(blocks):
                    m = self.trials_of(stim) & self.valid & (self.block == b)
                    if m.any():
                        out[i] = self.resp[m].mean(0)
                return out
            ref = block_means(reference)
            for st in self.stims:
                if st == reference:
                    continue
                diff = block_means(st) - ref
                ok = ~np.isnan(diff[:, 0])
                for n in range(self.n_neurons):
                    x = diff[ok, n]
                    p = stats.wilcoxon(x).pvalue if np.any(x != 0) else 1.0
                    rows.append({"neuron": n, "stim": st, "mean": float(np.mean(x)),
                                 "n": int(ok.sum()), "p": float(p)})
        df = pd.DataFrame(rows)
        df["q"] = fdr_bh(df["p"])
        df["sig"] = df["q"] < alpha
        df["sign"] = np.where(df["sig"], np.sign(df["mean"]), 0).astype(int)
        df.attrs["reference"] = reference
        self._cache["resp"] = df
        return df

    def tuning(self, ref="Empty"):
        """Per neuron: Kruskal-Wallis across all stimuli, preferred stimulus and
        category, and lifetime sparseness (0 = equal drive to all, 1 = one
        stimulus) — all on responses relative to ``ref`` (rectified for sparseness)."""
        R, stims = self.rel_matrix(ref)
        cats = [c for c in CATEGORIES if c != "control"]
        C = np.column_stack([np.nanmean(self.resp[self.trials_of(CATEGORIES[c]) & self.valid], axis=0)
                             for c in cats])
        rows = []
        for n in range(self.n_neurons):
            groups = [self.resp[self.trials_of(st) & self.valid, n] for st in self.stims]
            kw = stats.kruskal(*groups).pvalue
            r = np.clip(R[n], 0, None)
            k = len(r)
            ls = ((1 - (r.sum() / k) ** 2 / ((r ** 2).sum() / k)) / (1 - 1 / k)
                  if (r ** 2).sum() > 0 else np.nan)
            rows.append({"neuron": n, "kw_p": float(kw),
                         "pref_stim": stims[int(np.argmax(R[n]))],
                         "pref_category": cats[int(np.argmax(C[n]))],
                         f"max_vs_{ref}": float(R[n].max()), f"min_vs_{ref}": float(R[n].min()),
                         "sparseness": float(ls)})
        df = pd.DataFrame(rows)
        df["kw_q"] = fdr_bh(df["kw_p"])
        return df

    def _perm_masks(self, idx, is_a, n_shuffle, rng, within_block=True):
        """(n_shuffle, len(idx)) boolean masks: labels permuted within blocks."""
        out = np.empty((n_shuffle, len(idx)), bool)
        groups = ([np.where(self.block[idx] == b)[0] for b in np.unique(self.block[idx])]
                  if within_block else [np.arange(len(idx))])
        for k in range(n_shuffle):
            m = is_a.copy()
            for g in groups:
                m[g] = rng.permutation(is_a[g])
            out[k] = m
        return out

    def contrast_auc(self, contrasts=None, n_shuffle=1000, within_block=True, seed=0,
                     alpha=0.05):
        """ROC AUC per neuron for each contrast (A vs B trials) with a
        permutation null. Two-sided p; BH-FDR across neurons within a contrast.
        Returns a long DataFrame (neuron, contrast, auc, p, q, sig)."""
        contrasts = contrasts or CONTRASTS
        key = (tuple(contrasts), n_shuffle, within_block, seed)
        if self._cache.get("auc_key") == key:
            return self._cache["auc"]
        rng = np.random.default_rng(seed)
        rows = []
        for name, (A, B) in contrasts.items():
            a_m, b_m = self.trials_of(A) & self.valid, self.trials_of(B) & self.valid
            idx = np.where(a_m | b_m)[0]
            is_a = a_m[idx]
            nA, nB = int(is_a.sum()), int((~is_a).sum())
            if nA == 0 or nB == 0:
                continue
            ranks = stats.rankdata(self.resp[idx], axis=0)          # (trials, neurons)
            to_auc = lambda M: (M.astype(float) @ ranks - nA * (nA + 1) / 2) / (nA * nB)
            obs = to_auc(is_a[None])[0]                               # (neurons,)
            null = to_auc(self._perm_masks(idx, is_a, n_shuffle, rng, within_block))
            p = (1 + (np.abs(null - 0.5) >= np.abs(obs - 0.5) - 1e-12).sum(0)) / (1 + n_shuffle)
            q = fdr_bh(p)
            for n in range(self.n_neurons):
                rows.append({"neuron": n, "contrast": name, "auc": float(obs[n]),
                             "p": float(p[n]), "q": float(q[n]), "n_a": nA, "n_b": nB})
        df = pd.DataFrame(rows)
        df["sig"] = df["q"] < alpha
        self._cache.update(auc=df, auc_key=key)
        return df

    def classify(self, alpha=0.05, n_shuffle=1000, within_block=True, seed=0, reference="Empty"):
        """One class per neuron:
        * <category>     : its "<category>_vs_rest" AUC > 0.5 with q < alpha
                           (if several, the highest AUC wins)
        * non-selective  : responsive (any stimulus differs from ``reference``,
                           q < alpha) but no category preference
        * non-responsive : neither
        Preference is RELATIVE (higher activity than the other categories), which
        can include "less suppressed"; check the tuning matrix for the sign."""
        resp = self.responsiveness(alpha, reference=reference)
        auc = self.contrast_auc(CONTRASTS, n_shuffle, within_block, seed, alpha)
        tun = self.tuning()
        responsive = resp.groupby("neuron")["sig"].any()
        rows = []
        for n in range(self.n_neurons):
            best, best_auc = None, 0.5
            for cls, cname in CLASS_CONTRASTS.items():
                r = auc[(auc.neuron == n) & (auc.contrast == cname)].iloc[0]
                if r.sig and r.auc > best_auc:
                    best, best_auc = cls, r.auc
            label = best or ("non-selective" if responsive[n] else "non-responsive")
            rows.append({"neuron": n, "class": label, "responsive": bool(responsive[n])})
        df = pd.DataFrame(rows).merge(tun, on="neuron")
        wide = auc.pivot(index="neuron", columns="contrast", values="auc").add_prefix("auc_")
        wq = auc.pivot(index="neuron", columns="contrast", values="q").add_prefix("q_")
        df = df.merge(wide, left_on="neuron", right_index=True).merge(wq, left_on="neuron", right_index=True)
        self.classes = df
        return df

    # -------------------------------------------------- selectivity figures
    def plot_tuning_matrix(self, classes=None, alpha=0.05, clim=None, figsize=(3.2, 3), dpi=200,
                           cmap="RdBu_r"):
        """Neurons x stimuli mean response MINUS the Empty response; rows grouped
        by class then preferred stimulus; * = significant in ``responsiveness``
        (vs Empty by default, FDR)."""
        classes = self.classes if classes is None else classes
        resp = self._cache.get("resp")
        if resp is None:
            resp = self.responsiveness(alpha)
        R, stims = self.rel_matrix("Empty")
        cls_order = list(CLASS_COLORS)
        key = [(cls_order.index(classes.loc[classes.neuron == n, "class"].iloc[0]),
                stims.index(classes.loc[classes.neuron == n, "pref_stim"].iloc[0]))
               for n in range(self.n_neurons)]
        order = sorted(range(self.n_neurons), key=lambda n: key[n])
        clim = clim or np.nanmax(np.abs(R))
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        im = ax.imshow(R[order], cmap=cmap, vmin=-clim, vmax=clim, aspect="auto")
        sig = (resp.pivot(index="neuron", columns="stim", values="sig")
               .reindex(columns=stims, fill_value=False).to_numpy())
        for i, n in enumerate(order):
            for j in range(len(stims)):
                if sig[n, j]:
                    ax.text(j, i, "*", ha="center", va="center", fontsize="small")
            c = CLASS_COLORS[classes.loc[classes.neuron == n, "class"].iloc[0]]
            ax.add_patch(plt.Rectangle((-1.1, i - 0.5), 0.45, 1, color=c if c != "none" else "white",
                                       ec="0.6", lw=0.3, clip_on=False))
        ax.set_xticks(range(len(stims)))
        ax.set_xticklabels([STIM_LABELS[s] for s in stims], rotation=60, ha="right")
        for t, s in zip(ax.get_xticklabels(), stims):
            t.set_color(STIM_COLORS[s])
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([f"n{n}" for n in order], fontsize="x-small")
        ax.tick_params(axis="y", pad=12)
        fig.colorbar(im, ax=ax, shrink=0.8,
                     label=f"{self.ylab} vs empty ({self.response_win[0]:g}-{self.response_win[1]:g} s)")
        return fig

    def plot_category_scatter(self, x="food", y="social", classes=None, ax=None,
                              figsize=(2.2, 2.2), dpi=200):
        """Per-neuron mean response to category x vs category y (each minus the
        Empty response, so neuron offsets cancel), colored by class."""
        classes = self.classes if classes is None else classes
        e = np.nanmean(self.resp[self.trials_of("Empty") & self.valid], axis=0)
        mx = np.nanmean(self.resp[self.trials_of(CATEGORIES[x]) & self.valid], axis=0) - e
        my = np.nanmean(self.resp[self.trials_of(CATEGORIES[y]) & self.valid], axis=0) - e
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        lim = np.nanmax(np.abs(np.r_[mx, my])) * 1.15
        ax.plot([-lim, lim], [-lim, lim], color="0.7", lw=0.5, ls="--")
        ax.axhline(0, color="0.85", lw=0.4); ax.axvline(0, color="0.85", lw=0.4)
        for n in range(self.n_neurons):
            c = CLASS_COLORS[classes.loc[classes.neuron == n, "class"].iloc[0]]
            ax.scatter(mx[n], my[n], s=18, facecolor=c, edgecolor="k" if c in ("none", "#D9D9D9") else c,
                       lw=0.5, zorder=3)
            ax.annotate(str(n), (mx[n], my[n]), fontsize="xx-small", xytext=(2, 2),
                        textcoords="offset points")
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel=f"{x} − empty ({self.ylab})",
               ylabel=f"{y} − empty ({self.ylab})", aspect="equal")
        return ax.figure

    def plot_class_counts(self, classes=None, ax=None, figsize=(2, 1.8), dpi=200):
        classes = self.classes if classes is None else classes
        counts = classes["class"].value_counts().reindex(list(CLASS_COLORS), fill_value=0)
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.bar(range(len(counts)), counts.values,
               color=[c if c != "none" else "white" for c in CLASS_COLORS.values()],
               edgecolor="k", lw=0.5)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=45, ha="right")
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_ylabel("neurons")
        return ax.figure

    def plot_auc_summary(self, contrasts=None, ax=None, figsize=(3.2, 2), dpi=200):
        """Strip plot of per-neuron AUC for each contrast; filled = FDR-significant."""
        auc = self._cache.get("auc")
        if auc is None:
            auc = self.contrast_auc()
        contrasts = contrasts or list(dict.fromkeys(auc["contrast"]))
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.axhline(0.5, color="0.6", lw=0.5, ls="--")
        rng = np.random.default_rng(1)
        for i, c in enumerate(contrasts):
            d = auc[auc.contrast == c]
            xj = i + rng.uniform(-0.15, 0.15, len(d))
            ax.scatter(xj[~d.sig.values], d.auc[~d.sig.values], s=8, facecolor="none", edgecolor="0.4", lw=0.5)
            ax.scatter(xj[d.sig.values], d.auc[d.sig.values], s=10, color="k")
        ax.set_xticks(range(len(contrasts)))
        ax.set_xticklabels(contrasts, rotation=60, ha="right")
        ax.set(ylabel="AUC (A vs B)", ylim=(0, 1))
        return ax.figure

    # ====================================================== 3b. controls
    def identity_consistency(self, pairs=(("Male1", "Male2"), ("Female1", "Female2")),
                             figsize=(4, 2), dpi=200):
        """Per-neuron mean response to two animals of the same sex. If tuning is
        about category (not identity / block position) points sit near unity."""
        rel, stims = self.rel_matrix("Empty")          # neuron offsets removed
        R = dict(zip(stims, rel.T))
        fig, axs = plt.subplots(1, len(pairs), figsize=figsize, dpi=dpi)
        out = {}
        for ax, (a, b) in zip(np.atleast_1d(axs), pairs):
            r, p = stats.pearsonr(R[a], R[b])
            out[f"{a}_vs_{b}"] = {"r": r, "p": p}
            lim = np.nanmax(np.abs(np.r_[R[a], R[b]])) * 1.15
            ax.plot([-lim, lim], [-lim, lim], color="0.7", lw=0.5, ls="--")
            ax.scatter(R[a], R[b], s=14, color=STIM_COLORS[a], edgecolor="k", lw=0.3)
            ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal",
                   xlabel=f"{STIM_LABELS[a]} − empty", ylabel=f"{STIM_LABELS[b]} − empty",
                   title=f"r = {r:.2f}, p = {p:.2g}")
        fig.tight_layout()
        return pd.DataFrame(out).T, fig

    def split_half(self, figsize=(4, 2), dpi=200):
        """Odd vs even blocks: per-neuron correlation of the 8-stimulus tuning
        vector, and social-minus-food preference in each half."""
        odd = (self.block % 2 == 1) & self.valid
        even = (self.block % 2 == 0) & self.valid
        def tune(mask):
            return np.column_stack([np.nanmean(self.resp[self.trials_of(s) & mask], axis=0)
                                    for s in self.stims])
        To, Te = tune(odd), tune(even)
        r = np.array([stats.pearsonr(To[n], Te[n])[0] for n in range(self.n_neurons)])
        def sf(mask):
            return (np.nanmean(self.resp[self.trials_of(CATEGORIES["social"]) & mask], axis=0)
                    - np.nanmean(self.resp[self.trials_of(CATEGORIES["food"]) & mask], axis=0))
        so, se = sf(odd), sf(even)
        df = pd.DataFrame({"neuron": range(self.n_neurons), "tuning_r": r,
                           "social_minus_food_odd": so, "social_minus_food_even": se})
        fig, axs = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
        axs[0].scatter(np.zeros(len(r)) + np.random.default_rng(0).uniform(-0.1, 0.1, len(r)),
                       r, s=12, color="k")
        axs[0].axhline(0, color="0.7", lw=0.5, ls="--")
        axs[0].set(xlim=(-0.5, 0.5), xticks=[], ylabel="odd vs even tuning r", ylim=(-1, 1),
                   title=f"median r = {np.nanmedian(r):.2f}")
        rr, pp = stats.pearsonr(so, se)
        lim = np.nanmax(np.abs(np.r_[so, se])) * 1.15
        axs[1].plot([-lim, lim], [-lim, lim], color="0.7", lw=0.5, ls="--")
        axs[1].scatter(so, se, s=12, color=CATEGORY_COLORS["social"])
        axs[1].set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal", xlabel="social - food (odd)",
                   ylabel="social - food (even)", title=f"r = {rr:.2f}, p = {pp:.2g}")
        fig.tight_layout()
        return df, fig

    def drift(self):
        """Spearman of each neuron's stimulus-centred response vs trial order
        (session time). Large |rho| = slow drift / adaptation across the session."""
        centred = self.resp.copy()
        for s in self.stims:
            m = self.trials_of(s)
            centred[m] -= np.nanmean(centred[m & self.valid], axis=0)
        order = np.arange(self.n_trials)
        rows = []
        for n in range(self.n_neurons):
            rho, p = stats.spearmanr(order[self.valid], centred[self.valid, n])
            rows.append({"neuron": n, "rho": rho, "p": p})
        df = pd.DataFrame(rows)
        df["q"] = fdr_bh(df["p"])
        return df

    def baseline_carryover(self, session_file=None, figsize=(2.6, 1.8), dpi=200):
        """Raw (un-subtracted) baseline-window level by stimulus, from the session
        trace. Because the order is fixed, this is also the level by BLOCK
        POSITION. Tested per neuron with a Friedman test (blocks as repeated
        measures), so block-to-block differences don't mask a position trend.
        Returns (per-neuron table, figure, blocks x stimuli table of the
        population-mean baseline)."""
        session_file = session_file or f"session_{self.signal}.npy"
        S = np.load(os.path.join(self.data_dir, session_file))
        on = self.trials["onset_frame"].to_numpy().astype(int)
        a = int(round(self.baseline_win[0] * self.fs)); b = int(round(self.baseline_win[1] * self.fs))
        base = np.stack([S[:, o + a:o + b].mean(1) for o in on])      # (trials, neurons)
        blocks = np.unique(self.block)
        # (blocks, stims, neurons) block means; blocks missing a stimulus are dropped
        B = np.full((len(blocks), len(self.stims), self.n_neurons), np.nan)
        for i, bl in enumerate(blocks):
            for j, st in enumerate(self.stims):
                m = (self.block == bl) & self.trials_of(st)
                if m.any():
                    B[i, j] = base[m].mean(0)
        full = ~np.isnan(B[:, :, 0]).any(1)
        rows = [{"neuron": n, "friedman_p": stats.friedmanchisquare(*B[full, :, n].T).pvalue,
                 "rho_vs_position": stats.spearmanr(np.tile(np.arange(len(self.stims)), full.sum()),
                                                    B[full, :, n].ravel())[0]}
                for n in range(self.n_neurons)]
        df = pd.DataFrame(rows)
        df["friedman_q"] = fdr_bh(df["friedman_p"])
        per = np.nanmean(B[full], axis=0).T                              # (neurons, stims)
        m, s = _nanmean_sem(per, axis=0)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.bar(range(len(self.stims)), m, yerr=s, color=[STIM_COLORS[x] for x in self.stims],
               error_kw={"lw": 0.6})
        ax.axhline(0, color="0.6", lw=0.5)
        ax.set_xticks(range(len(self.stims)))
        ax.set_xticklabels([f"{i+1}. {STIM_LABELS[x]}" for i, x in enumerate(self.stims)],
                           rotation=60, ha="right")
        ax.set_ylabel(f"raw baseline {self.ylab}\n({self.baseline_win[0]:g} to {self.baseline_win[1]:g} s; mean ± SEM neurons)")
        ax.set_xlabel("block position. stimulus")
        tab = pd.DataFrame(B.mean(2), index=blocks, columns=self.stims)
        tab.index.name = "block"
        return df, fig, tab

    # ============================================================ 4. FOV
    def load_fov(self, suite2p_dir):
        """Load suite2p ops/stat/iscell and map tensor neurons -> ROIs."""
        ops = np.load(os.path.join(suite2p_dir, "ops.npy"), allow_pickle=True).item()
        stat = np.load(os.path.join(suite2p_dir, "stat.npy"), allow_pickle=True)
        iscell = np.load(os.path.join(suite2p_dir, "iscell.npy"))
        # tensor neuron -> suite2p ROI: from the alignment metadata (after cell QC),
        # else every iscell ROI
        meta_idx = getattr(self, "meta", {}).get("cell_idx")
        cell_idx = np.asarray(meta_idx) if meta_idx is not None else np.where(iscell[:, 0] == 1)[0]
        made_from = getattr(self, "meta", {}).get("suite2p_path")
        if made_from and os.path.normpath(made_from) != os.path.normpath(suite2p_dir):
            raise ValueError(f"tensor was built from {made_from} but SUITE2P is {suite2p_dir} — "
                             "re-run processing_main_abf.ipynb for this session")
        if len(cell_idx) != self.n_neurons:
            raise ValueError(f"{len(cell_idx)} iscell ROIs but {self.n_neurons} neurons in tensor — "
                             "the tensor is from a different session or an older iscell curation; "
                             "re-run processing_main_abf.ipynb for this session")
        self.fov = {"ops": ops, "stat": stat[cell_idx], "cell_idx": cell_idx}
        return self

    def _roi_mask(self, n):
        st = self.fov["stat"][n]
        Ly, Lx = self.fov["ops"]["Ly"], self.fov["ops"]["Lx"]
        keep = ~st["overlap"] if "overlap" in st else np.ones(len(st["ypix"]), bool)
        m = np.zeros((Ly, Lx), bool)
        m[st["ypix"][keep], st["xpix"][keep]] = True
        return m

    def plot_fov(self, color_by="class", contrast="social_vs_food", image="meanImg",
                 classes=None, crop=True, pad=40, alpha_fill=0.55, ax=None,
                 figsize=(3, 3), dpi=200, label=True):
        """Suite2p image with ROI footprints.
        color_by="class": fill by class (non-responsive = outline only).
        color_by="auc"  : fill by the AUC of ``contrast`` (diverging, 0.5 = white);
                          non-significant ROIs are drawn with a dashed outline."""
        if self.fov is None:
            raise RuntimeError("call load_fov(suite2p_dir) first")
        ops = self.fov["ops"]
        img = ops[image].astype(float)
        lo, hi = np.percentile(img, [1, 99.5])
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.imshow(np.clip((img - lo) / (hi - lo), 0, 1), cmap="gray", interpolation="none")
        overlay = np.zeros(img.shape + (4,))
        classes = self.classes if (classes is None and color_by == "class") else classes
        if color_by == "auc":
            auc = self._cache.get("auc")
            if auc is None:
                auc = self.contrast_auc()
            d = auc[auc.contrast == contrast].set_index("neuron")
            cmap = plt.get_cmap("RdBu_r")
        for n in range(self.n_neurons):
            m = self._roi_mask(n)
            ls = "-"
            if color_by == "class":
                c = CLASS_COLORS[classes.loc[classes.neuron == n, "class"].iloc[0]]
                edge = "w" if c == "none" else c
                if c != "none":
                    overlay[m] = to_rgba(c, alpha_fill)
            else:
                v = d.loc[n, "auc"]
                c = cmap(v)
                overlay[m] = (*c[:3], alpha_fill)
                edge = c
                ls = "-" if d.loc[n, "sig"] else "--"
            ax.contour(m, levels=[0.5], colors=[edge], linewidths=0.6, linestyles=ls)
            if label:
                y, x = self.fov["stat"][n]["med"]
                ax.text(x, y, str(n), color="w", fontsize="xx-small", ha="center", va="center",
                        path_effects=[pe.withStroke(linewidth=1.2, foreground="k")])
        ax.imshow(overlay, interpolation="none")
        if crop:
            ys = np.concatenate([s["ypix"] for s in self.fov["stat"]])
            xs = np.concatenate([s["xpix"] for s in self.fov["stat"]])
            ax.set_xlim(max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad))
            ax.set_ylim(min(img.shape[0], ys.max() + pad), max(0, ys.min() - pad))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if color_by == "class":
            from matplotlib.patches import Patch
            h = [Patch(facecolor=c if c != "none" else "white", edgecolor="k", lw=0.4, label=k)
                 for k, c in CLASS_COLORS.items()]
            ax.legend(handles=h, frameon=False, loc="upper left", bbox_to_anchor=(1, 1),
                      fontsize="x-small")
            ax.set_title(f"{image}: class")
        else:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
            a_name, b_name = contrast.split("_vs_") if "_vs_" in contrast else ("A", "B")
            ax.figure.colorbar(sm, ax=ax, shrink=0.7,
                               label=f"AUC (>0.5 prefers {a_name}, <0.5 prefers {b_name})")
            ax.set_title(f"{image}: {contrast}")
        return ax.figure


def save_fig(fig, name, out_dir, fmts=("png", "svg")):
    """Save a figure to ``out_dir`` in each format (no-op if out_dir is None)."""
    if out_dir is None:
        return
    os.makedirs(out_dir, exist_ok=True)
    for f in fmts:
        fig.savefig(os.path.join(out_dir, f"{name}.{f}"), bbox_inches="tight", dpi=300)
