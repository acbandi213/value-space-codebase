"""
neural_analysis.py
==================
Stimulus-response analysis for the SocialPresentation calcium tensor produced by
abf_imaging_alignment (presentation-locked, trial x neuron x frame).

Inputs
------
X          : ndarray (n_trials, n_neurons, n_frames)   dff (or z_dff)
labels     : array-like (n_trials,)                    stimulus name per trial
regions    : array-like (n_neurons,) or None           cortical region per neuron
pre_frames : int   baseline frames before onset (onset is at frame index pre_frames)
fs         : float imaging frame rate (Hz)

Everything is region-OPTIONAL: if ``regions`` is None (e.g. a small FOV) the
region arguments are ignored and all neurons are used.

Provides
--------
* modular pre-onset baseline normalization (z-score to the pre-onset window)
* Panel-E figure: population trace + neuron heatmap per condition (all neurons
  or one region)
* single-neuron examples with preferential responses
* binary decoders with shuffle chance, optionally per region:
    social_vs_nonsocial : (Male, Female)  vs (FamFood, NewFood)
    male_vs_female      :  Male           vs  Female
    food1_vs_food2      :  FamFood        vs  NewFood
    fake_vs_real        :  3DMouse        vs (Male, Female)
  Empty is kept as a baseline/control condition.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

# --- experiment-specific stimulus groupings (edit here if names differ) ------
SOCIAL_REAL = ["Male", "Female"]
FOOD = ["FamFood", "NewFood"]
FAKE = ["3DMouse"]
BASELINE = ["Empty"]

# canonical left-to-right order + display labels for all-stimulus figures
STIM_ORDER = ["Empty", "3DMouse", "Male", "Female", "FamFood", "NewFood"]
STIM_LABELS = {"Empty": "empty cage", "3DMouse": "fake mouse", "Male": "male",
               "Female": "female", "FamFood": "familiar food", "NewFood": "unfamiliar food"}

DECODERS = {
    "social_vs_nonsocial": (SOCIAL_REAL, FOOD),
    "male_vs_female":      (["Male"], ["Female"]),
    "food1_vs_food2":      (["FamFood"], ["NewFood"]),
    "fake_vs_real":        (FAKE, SOCIAL_REAL),
}


def set_style(font="Arial", size=7):
    """Publication style for all figures: <font> at <size>pt, no top/right spines."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [font, "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": size, "axes.titlesize": size, "axes.labelsize": size,
        "xtick.labelsize": size, "ytick.labelsize": size,
        "legend.fontsize": size, "figure.titlesize": size, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
    })


set_style()   # applied on import -> every figure from this module is styled


# --------------------------------------------------------------------------
# Kinematics regression: regress out ball motion from neural activity
# --------------------------------------------------------------------------
def build_kinematics_design(ball, lags=(0, 1, 2)):
    """Build a design matrix from ball signals with time-lagged features.

    Parameters
    ----------
    ball : dict with 'roll', 'pitch', 'yaw' arrays of shape (n_frames,)
    lags : tuple of int, frame lags to include (0 = current frame,
           1 = one frame back ~67ms at 15 Hz, etc.)

    Returns (X_design, feature_names) where X_design is (n_frames, n_features).
    Features: roll, pitch, yaw, |roll|, |pitch|, |yaw|, speed — each at every lag.
    """
    roll, pitch, yaw = ball["roll"], ball["pitch"], ball["yaw"]
    speed = np.sqrt(roll**2 + pitch**2 + yaw**2)
    base = {"roll": roll, "pitch": pitch, "yaw": yaw,
            "|roll|": np.abs(roll), "|pitch|": np.abs(pitch), "|yaw|": np.abs(yaw),
            "speed": speed}
    cols, names = [], []
    n = len(roll)
    for lag in lags:
        for name, sig in base.items():
            padded = np.concatenate([np.zeros(lag), sig[:n - lag]]) if lag > 0 else sig
            cols.append(padded)
            names.append(f"{name}_t-{lag}" if lag > 0 else name)
    return np.column_stack(cols), names


def regress_out_kinematics(neural, ball, lags=(0, 1, 2), alpha=1.0):
    """Ridge-regress kinematics out of neural activity (session-level).

    Parameters
    ----------
    neural : ndarray (n_neurons, n_frames) — e.g. dff or z_dff
    ball   : dict with 'roll', 'pitch', 'yaw' (n_frames,)
    lags   : frame lags for design matrix
    alpha  : ridge regularisation strength

    Returns
    -------
    residuals : ndarray (n_neurons, n_frames) — activity with kinematics removed
    r2        : ndarray (n_neurons,) — variance explained per neuron
    coefs     : ndarray (n_neurons, n_features)
    feature_names : list of str
    """
    from sklearn.linear_model import Ridge

    X_design, feature_names = build_kinematics_design(ball, lags=lags)
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X_design, neural.T)  # fit all neurons at once
    predicted = model.predict(X_design).T  # (n_neurons, n_frames)
    residuals = neural - predicted

    ss_res = np.sum((neural - predicted)**2, axis=1)
    ss_tot = np.sum((neural - neural.mean(axis=1, keepdims=True))**2, axis=1)
    r2 = 1 - ss_res / (ss_tot + 1e-12)

    return residuals, r2, model.coef_, feature_names


class NeuralAnalysis:
    def __init__(self, X, labels, regions=None, fs=15.0, pre_frames=15):
        self.X = np.asarray(X, float)
        self.labels = np.asarray(labels)
        self.regions = None if regions is None else np.asarray(regions)
        self.fs = float(fs)
        self.pre = int(pre_frames)
        self.n_trials, self.n_neurons, self.n_frames = self.X.shape
        self.time = (np.arange(self.n_frames) - self.pre) / self.fs  # 0 = onset
        self.Xz = None
        self.normalize()

    @property
    def has_regions(self):
        return self.regions is not None

    def region_list(self):
        return [] if not self.has_regions else sorted(set(self.regions.tolist()))

    # ------------------------------------------------------------------ 1. norm
    def normalize(self, method="baseline_sub"):
        """Normalize the trial tensor for display / analysis.

        baseline_sub  : subtract EACH trial's own pre-onset baseline (per trial,
                        per neuron). Correct when the tensor is already session
                        Z-scored dF/F (Willmore). Recommended default.
        zscore_pre    : per-trial baseline subtraction, then divide by the neuron's
                        pre-onset SD (pooled across trials).
        zscore_session: z-score each neuron across the tensor's frames.
        none          : leave the tensor unchanged.
        """
        X = self.X
        if method in ("baseline_sub", "zscore_pre") and self.pre > 0:
            b = X[:, :, :self.pre].mean(axis=2, keepdims=True)      # per-trial baseline
            if method == "baseline_sub":
                self.Xz = X - b
            else:
                sd = X[:, :, :self.pre].std(axis=(0, 2))[None, :, None]
                self.Xz = (X - b) / (sd + 1e-9)
        elif method == "zscore_session":
            mu = X.mean(axis=(0, 2))[None, :, None]
            sd = X.std(axis=(0, 2))[None, :, None]
            self.Xz = (X - mu) / (sd + 1e-9)
        else:
            self.Xz = X.copy()
        self.norm_method = method
        return self

    # ---------------------------------------------------------------- helpers
    def _trials_for(self, stim):
        stim = [stim] if isinstance(stim, str) else list(stim)
        return np.where(np.isin(self.labels, stim))[0]

    def neuron_mask(self, region=None):
        """Boolean mask over neurons. region=None -> all neurons."""
        if region is None or not self.has_regions:
            return np.ones(self.n_neurons, bool)
        region = [region] if isinstance(region, str) else list(region)
        return np.isin(self.regions, region)

    def condition_mean(self, stim, region=None):
        """Trial-averaged Xz for a stimulus (or group): (n_sel_neurons, n_frames)."""
        tr = self._trials_for(stim)
        nm = self.neuron_mask(region)
        return self.Xz[tr][:, nm, :].mean(axis=0)

    def _resp_window(self, window):
        """(start_frame, stop_frame) from a (start_s, stop_s) window rel. onset."""
        a = self.pre if window[0] is None else self.pre + int(round(window[0] * self.fs))
        b = self.n_frames if window[1] is None else self.pre + int(round(window[1] * self.fs))
        return max(0, a), min(self.n_frames, b)

    # ---------------------------------------------------- 2. Panel-E figure
    def plot_population_heatmap(self, conditions=None, region=None, sort="independent",
                                sort_ref=None, sort_window=(0, None), clip=3.0,
                                event_times=None, cmap="RdBu_r", figsize=None, dpi=None,
                                show_trials=False, smooth_sigma=0):
        """Population trace (top) + neuron heatmap (bottom) per condition.

        conditions : list of stimulus names, or (title, stim_or_list) tuples.
                     Defaults to Male, Female, 3DMouse, Food, Empty.
        region     : None -> all neurons; else restrict to that cortical region.
        sort       : 'independent' (each column sorted by its own response) or
                     'shared' (one order from ``sort_ref``, applied to all columns).
        event_times: list of times (s, rel. onset) to draw as dashed lines
                     (onset at 0 is always drawn).
        show_trials: overlay single-trial traces as faint lines on the top panel.
        smooth_sigma: Gaussian smoothing sigma in frames (0 = no smoothing).
                      Applied to traces and heatmap rows for display only.
        """
        from scipy.ndimage import gaussian_filter1d
        def _smooth(x):
            if smooth_sigma > 0:
                return gaussian_filter1d(x, sigma=smooth_sigma, axis=-1)
            return x

        if conditions is None:
            conditions = [(STIM_LABELS[s], s) for s in STIM_ORDER]   # empty, fake, male, female, fam food, unfam food
        conditions = [(c, c) if isinstance(c, str) else c for c in conditions]
        a, b = self._resp_window(sort_window)

        mats = [self.condition_mean(spec, region) for _, spec in conditions]
        n_sel = mats[0].shape[0]
        if sort == "shared":
            ref = sort_ref if sort_ref is not None else conditions[0][1]
            order = np.argsort(-self.condition_mean(ref, region)[:, a:b].mean(1))
            orders = [order] * len(conditions)
        else:
            orders = [np.argsort(-m[:, a:b].mean(1)) for m in mats]

        ncol = len(conditions)
        figsize = figsize or (2.6 * ncol + 0.8, 6)
        fig, axs = plt.subplots(2, ncol, figsize=figsize, sharex=True, dpi=dpi,
                                gridspec_kw={"height_ratios": [1, 3]})
        if ncol == 1:
            axs = axs.reshape(2, 1)
        for j, (title, spec) in enumerate(conditions):
            m = mats[j]
            # top: optional single-trial traces
            if show_trials:
                spec_list = [spec] if isinstance(spec, str) else list(spec)
                tr_idx = self._trials_for(spec_list)
                for ti in tr_idx:
                    trial_pop = self.Xz[ti][self.neuron_mask(region)].mean(0)
                    axs[0, j].plot(self.time, _smooth(trial_pop), color="tab:red", lw=0.3, alpha=0.25)
            # top: population mean +/- SEM across neurons
            pop = _smooth(m.mean(0))
            sem = _smooth(m.std(0) / np.sqrt(max(m.shape[0], 1)))
            axs[0, j].plot(self.time, pop, color="tab:red", lw=1.2)
            axs[0, j].fill_between(self.time, pop - sem, pop + sem, color="tab:red", alpha=0.3)
            axs[0, j].set_title(title, fontsize=7)
            axs[0, j].axhline(0, color="0.7", lw=0.5)
            axs[0, j].set_xlim(self.time[0], self.time[-1])
            axs[0, j].set_ylim(-0.1, 0.1)
            # bottom: heatmap sorted
            im = axs[1, j].imshow(_smooth(m[orders[j]]), aspect="auto", cmap=cmap,
                                  vmin=-clip, vmax=clip, extent=[self.time[0], self.time[-1], n_sel, 0],
                                  interpolation="nearest")
            # event lines
            for t in [0.0] + list(event_times or []):
                axs[0, j].axvline(t, color="k", ls="--", lw=0.8)
                axs[1, j].axvline(t, color="k", ls="--", lw=0.8)
            axs[1, j].set_xlabel("time from onset (s)", fontsize=7)
            if j == 0:
                axs[0, j].set_ylabel("Z-scored ΔF/F", fontsize=7)
                axs[1, j].set_ylabel("Neurons", fontsize=7)
            else:
                axs[1, j].set_yticklabels([])
        reg_str = "all neurons" if region is None else f"region={region}"
        #fig.suptitle(f"Stimulus responses ({reg_str}, n={n_sel})", fontsize=7)
        fig.tight_layout(rect=[0, 0, 0.93, 0.97])
        cax = fig.add_axes([0.945, 0.12, 0.012, 0.5])
        fig.colorbar(im, cax=cax, label="Z-scored ΔF/F")
        return fig

    # ------------------------------------------------ 3. preferential neurons
    def neuron_responses(self, window=(0, None), region=None):
        """Per-trial per-neuron scalar response = mean Xz over the window.
        Returns (resp[n_trials, n_sel], neuron_ids). neuron_ids index into the
        full neuron axis (so they map back to the tensor / regions)."""
        a, b = self._resp_window(window)
        nm = self.neuron_mask(region)
        ids = np.where(nm)[0]
        resp = self.Xz[:, nm, :][:, :, a:b].mean(2)          # (trials, sel)
        return resp, ids

    def preferential_neurons(self, target, others=None, window=(0, None),
                             region=None, n=10, min_auc=0.5, min_selectivity=1.5):
        """Rank neurons by preference for ``target`` vs ``others`` (default: all
        other stimuli). Preference = AUC of single-trial target-vs-other responses.

        Selectivity = resp_target / max(resp per other stimulus). Neurons must
        respond at least ``min_selectivity`` times more to the target than to
        the best other stimulus (default 1.5x). Set to 0 to disable.

        Returns a DataFrame (neuron_id, region, auc, selectivity, resp_target,
        resp_others, resp_per_stim) sorted by AUC, top ``n``.
        """
        from sklearn.metrics import roc_auc_score
        import pandas as pd
        resp, ids = self.neuron_responses(window, region)
        tgt_labels = [target] if isinstance(target, str) else list(target)
        tgt = self._trials_for(target)
        if others is None:
            others = [s for s in set(self.labels.tolist()) if s not in tgt_labels]
        other_stims = [others] if isinstance(others, str) else list(others)
        oth = self._trials_for(others)
        y = np.r_[np.ones(len(tgt)), np.zeros(len(oth))]
        rows = []
        for k, nid in enumerate(ids):
            scores = np.r_[resp[tgt, k], resp[oth, k]]
            try:
                auc = roc_auc_score(y, scores)
            except ValueError:
                auc = 0.5
            r_target = float(resp[tgt, k].mean())
            r_others = float(resp[oth, k].mean())
            best_other = max(float(resp[self._trials_for(s), k].mean()) for s in other_stims)
            selectivity = r_target / (abs(best_other) + 1e-9) if r_target > 0 else 0.0
            rows.append({
                "neuron_id": int(nid),
                "region": None if not self.has_regions else self.regions[nid],
                "auc": auc,
                "selectivity": selectivity,
                "resp_target": r_target,
                "resp_others": r_others,
            })
        df = pd.DataFrame(rows)
        df = df[(df.auc >= min_auc) & (df.resp_target > 0)]
        if min_selectivity > 0:
            df = df[df.selectivity >= min_selectivity]
        df = df.sort_values("auc", ascending=False).head(n).reset_index(drop=True)
        return df

    def plot_neuron_examples(self, neuron_ids, stims=None, figsize=None,
                             dpi=300, smooth_sigma=0):
        """Trial-averaged trace per stimulus for each example neuron."""
        from scipy.ndimage import gaussian_filter1d
        def _smooth(x):
            if smooth_sigma > 0:
                return gaussian_filter1d(x, sigma=smooth_sigma, axis=-1)
            return x

        stims = stims or STIM_ORDER
        stims = [s for s in stims if len(self._trials_for(s))]
        neuron_ids = list(neuron_ids)
        figsize = figsize or (2.0 * len(neuron_ids), 1.8)
        fig, axs = plt.subplots(1, len(neuron_ids), figsize=figsize, sharex=True,
                                squeeze=False, dpi=dpi)
        cmap = plt.get_cmap("tab10")
        for i, nid in enumerate(neuron_ids):
            for c, stim in enumerate(stims):
                tr = self._trials_for(stim)
                m = self.Xz[tr, nid, :]
                pop = _smooth(m.mean(0))
                sem = _smooth(m.std(0) / np.sqrt(max(len(tr), 1)))
                axs[0, i].plot(self.time, pop, lw=1.1, color=cmap(c), label=STIM_LABELS.get(stim, stim))
                axs[0, i].fill_between(self.time, pop - sem, pop + sem, color=cmap(c), alpha=0.18)
            axs[0, i].axvline(0, color="k", ls="--", lw=0.7)
            reg = "" if not self.has_regions else f" ({self.regions[nid]})"
            axs[0, i].set_title(f"neuron {nid}{reg}", fontsize=7)
            axs[0, i].set_xlabel("time (s)", fontsize=7)
        axs[0, 0].set_ylabel("Z-scored ΔF/F", fontsize=7)
        axs[0, -1].legend(fontsize=6, loc="upper right")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------- 4. decoding
    def decode_binary(self, group_a, group_b, region=None, window=(0, None),
                      n_splits=50, test_size=0.3, n_shuffles=200, C=1.0,
                      n_pcs=50, match_neurons=None, n_subsamples=20, seed=0):
        """Cross-validated binary decoder between two label groups.

        Features = per-neuron mean Xz over ``window``. PCA reduces
        dimensionality before a linear SVM. Repeated stratified shuffle
        splits (Monte Carlo CV, 70/30) give stable estimates despite
        small trial counts.
        """
        from sklearn.svm import SVC
        from sklearn.model_selection import StratifiedShuffleSplit
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        rng = np.random.default_rng(seed)
        a_tr, b_tr = self._trials_for(group_a), self._trials_for(group_b)
        resp, ids = self.neuron_responses(window, region)
        X = np.r_[resp[a_tr], resp[b_tr]]
        y = np.r_[np.zeros(len(a_tr)), np.ones(len(b_tr))].astype(int)

        def _acc(Xf, yy, shuffle=False):
            yy = rng.permutation(yy) if shuffle else yy
            scaler = StandardScaler()
            Xsc = scaler.fit_transform(Xf)
            if n_pcs is not None and Xsc.shape[1] > n_pcs:
                eff = min(n_pcs, Xsc.shape[0] - 1)
                pca = PCA(n_components=eff, random_state=seed)
                Xsc = pca.fit_transform(Xsc)
            sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size,
                                         random_state=int(rng.integers(1e6)))
            accs = []
            for train_idx, test_idx in sss.split(Xsc, yy):
                clf = SVC(kernel="linear", C=C, class_weight="balanced")
                clf.fit(Xsc[train_idx], yy[train_idx])
                accs.append(clf.score(Xsc[test_idx], yy[test_idx]))
            return float(np.mean(accs))

        if match_neurons and X.shape[1] > match_neurons:
            accs = [_acc(X[:, rng.choice(X.shape[1], match_neurons, replace=False)], y)
                    for _ in range(n_subsamples)]
            acc = float(np.mean(accs)); acc_sd = float(np.std(accs))
        else:
            acc = _acc(X, y); acc_sd = np.nan
        null = np.array([_acc(X, y, shuffle=True) for _ in range(n_shuffles)])
        p = float((np.sum(null >= acc) + 1) / (n_shuffles + 1))
        return {"accuracy": acc, "accuracy_sd": acc_sd, "chance": float(null.mean()),
                "chance_ci": (float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))),
                "p_value": p, "n_a": len(a_tr), "n_b": len(b_tr), "n_neurons": X.shape[1]}

    def decode_all(self, decoders=None, by_region=False, match_neurons=None, **kw):
        """Run all named decoders (optionally per region). Returns a DataFrame."""
        import pandas as pd
        decoders = decoders or DECODERS
        regions = [None]
        if by_region and self.has_regions:
            regions = self.region_list()
            if match_neurons is None:
                match_neurons = min(int(self.neuron_mask(r).sum()) for r in regions)
        rows = []
        for name, (ga, gb) in decoders.items():
            for reg in regions:
                r = self.decode_binary(ga, gb, region=reg,
                                       match_neurons=match_neurons if reg is not None else None, **kw)
                rows.append({"decoder": name, "region": reg or "all", **r})
        return pd.DataFrame(rows)

    def plot_decoding(self, results, ax=None):
        """Bar plot of decoder accuracy vs chance (optionally grouped by region)."""
        import pandas as pd
        if ax is None:
            _, ax = plt.subplots(figsize=(1.6 * results["decoder"].nunique() + 2, 4))
        regions = results["region"].unique()
        decs = results["decoder"].unique()
        w = 0.8 / len(regions)
        for i, reg in enumerate(regions):
            sub = results[results.region == reg].set_index("decoder").reindex(decs)
            x = np.arange(len(decs)) + i * w
            ax.bar(x, sub["accuracy"], w, yerr=sub["accuracy_sd"], label=str(reg))
            ax.plot(x, sub["chance"], "k_", ms=12)
            for xi, (_, row) in zip(x, sub.iterrows()):
                if row["p_value"] < 0.05:
                    ax.text(xi, row["accuracy"] + 0.02, "*", ha="center")
        ax.axhline(0.5, color="0.6", ls=":", lw=1)
        ax.set_xticks(np.arange(len(decs)) + w * (len(regions) - 1) / 2)
        ax.set_xticklabels(decs, rotation=25, ha="right", fontsize=7)
        ax.set(ylabel="decoding accuracy", ylim=(0, 1),
               title="binary decoding (chance = _, * p<0.05)")
        if len(regions) > 1:
            ax.legend(fontsize=7, title="region")
        return ax

    # ---------------------------------------- 4b. pairwise decoding matrix
    def decode_pairwise(self, stims=None, region=None, **kw):
        """Decode all unique pairs of stimuli. Returns a DataFrame with
        columns stim_a, stim_b, accuracy, chance, p_value, etc."""
        import pandas as pd
        from itertools import combinations
        stims = stims or sorted(set(self.labels.tolist()))
        rows = []
        for a, b in combinations(stims, 2):
            r = self.decode_binary([a], [b], region=region, **kw)
            rows.append({"stim_a": a, "stim_b": b, **r})
        return pd.DataFrame(rows)

    def plot_pairwise_matrix(self, results, stims=None, cmap="RdYlGn", figsize=None,
                             dpi=300, vmin=0.3, vmax=1.0, annot_fmt=".0%"):
        """6x6 symmetric heatmap of pairwise decoding accuracy."""
        stims = stims or STIM_ORDER
        n = len(stims)
        mat = np.full((n, n), np.nan)
        pmat = np.ones((n, n))
        idx = {s: i for i, s in enumerate(stims)}
        for _, row in results.iterrows():
            i, j = idx.get(row["stim_a"]), idx.get(row["stim_b"])
            if i is None or j is None:
                continue
            mat[i, j] = mat[j, i] = row["accuracy"]
            pmat[i, j] = pmat[j, i] = row["p_value"]
        np.fill_diagonal(mat, np.nan)

        figsize = figsize or (3.5, 3)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        display_labels = [STIM_LABELS.get(s, s) for s in stims]
        ax.set_xticks(range(n)); ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(n)); ax.set_yticklabels(display_labels, fontsize=7)
        for i in range(n):
            for j in range(n):
                if np.isnan(mat[i, j]):
                    continue
                txt = format(mat[i, j], annot_fmt)
                if pmat[i, j] < 0.05:
                    txt += "*"
                ax.text(j, i, txt, ha="center", va="center", fontsize=6,
                        color="white" if mat[i, j] > 0.75 else "black")
        fig.colorbar(im, ax=ax, shrink=0.8, label="decoding accuracy")
        ax.set_title("Pairwise stimulus decoding", fontsize=7)
        fig.tight_layout()
        return fig

    # ------------------------------------------ 5. temporal (frame-by-frame) decoding
    def decode_temporal(self, group_a, group_b, region=None, C=1.0,
                        n_splits=50, test_size=0.3, n_shuffles=200,
                        n_pcs=50, seed=0):
        """Frame-by-frame binary decoder: one classifier per time point.

        Uses repeated stratified shuffle splits (Monte Carlo CV) with a
        70/30 train/test ratio to get stable accuracy estimates despite
        small trial counts.

        PCA is fit once per frame on all trials (unsupervised — no label
        leakage) to reduce dimensionality before classification. Set
        ``n_pcs=None`` to skip PCA and use all neurons (slow).

        Returns dict with 'time', 'accuracy' (n_frames,),
        'accuracy_sem' (n_frames,), 'chance' (n_frames,),
        'chance_ci' (n_frames, 2), and metadata.
        """
        from sklearn.svm import SVC
        from sklearn.model_selection import StratifiedShuffleSplit
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        rng = np.random.default_rng(seed)

        a_tr = self._trials_for(group_a)
        b_tr = self._trials_for(group_b)
        nm = self.neuron_mask(region)
        Xa = self.Xz[np.ix_(a_tr, nm)]
        Xb = self.Xz[np.ix_(b_tr, nm)]
        Xall = np.concatenate([Xa, Xb], axis=0)  # (n_trials, n_sel, n_frames)
        y = np.r_[np.zeros(len(a_tr)), np.ones(len(b_tr))].astype(int)
        n_tr, n_neu, n_fr = Xall.shape

        use_pca = n_pcs is not None and n_neu > n_pcs

        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size,
                                     random_state=int(rng.integers(1e6)))
        splits = list(sss.split(Xall[:, 0, 0], y))

        acc_all = np.zeros((n_fr, n_splits))
        shuffle_mean = np.zeros((n_fr, n_shuffles))

        for f in range(n_fr):
            feats = Xall[:, :, f]  # (n_trials, n_neurons)
            scaler = StandardScaler()
            feats_sc = scaler.fit_transform(feats)
            if use_pca:
                eff = min(n_pcs, feats_sc.shape[0] - 1)
                pca = PCA(n_components=eff, random_state=seed)
                feats_sc = pca.fit_transform(feats_sc)

            for si, (train_idx, test_idx) in enumerate(splits):
                clf = SVC(kernel="linear", C=C, class_weight="balanced")
                clf.fit(feats_sc[train_idx], y[train_idx])
                acc_all[f, si] = clf.score(feats_sc[test_idx], y[test_idx])

            for s in range(n_shuffles):
                y_shuf = rng.permutation(y)
                accs_shuf = []
                for train_idx, test_idx in splits:
                    clf = SVC(kernel="linear", C=C, class_weight="balanced")
                    clf.fit(feats_sc[train_idx], y_shuf[train_idx])
                    accs_shuf.append(clf.score(feats_sc[test_idx], y_shuf[test_idx]))
                shuffle_mean[f, s] = np.mean(accs_shuf)

        return {
            "time": self.time,
            "accuracy": acc_all.mean(axis=1),
            "accuracy_sem": acc_all.std(axis=1) / np.sqrt(n_splits),
            "chance": shuffle_mean.mean(axis=1),
            "chance_ci": np.stack([np.percentile(shuffle_mean, 2.5, axis=1),
                                   np.percentile(shuffle_mean, 97.5, axis=1)], axis=1),
            "n_a": len(a_tr), "n_b": len(b_tr), "n_neurons": n_neu,
            "n_pcs": n_pcs if use_pca else n_neu,
            "n_splits": n_splits, "n_shuffles": n_shuffles,
        }

    def decode_temporal_all(self, decoders=None, region=None, **kw):
        """Run all named decoders frame-by-frame. Returns {name: result_dict}."""
        decoders = decoders or DECODERS
        return {name: self.decode_temporal(ga, gb, region=region, **kw)
                for name, (ga, gb) in decoders.items()}

    def plot_temporal_decoding(self, results, smooth_sigma=0, figsize=None, dpi=300, ax=None):
        """Plot frame-by-frame decoding accuracy for one or more decoders.

        results : dict from decode_temporal (single) or decode_temporal_all (multi).
        """
        from scipy.ndimage import gaussian_filter1d
        def _smooth(x):
            return gaussian_filter1d(x, sigma=smooth_sigma) if smooth_sigma > 0 else x

        if "time" in results:
            results = {"decoder": results}

        if ax is None:
            figsize = figsize or (4, 2.5)
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            fig = ax.figure
        cmap = plt.get_cmap("tab10")
        for i, (name, r) in enumerate(results.items()):
            t = r["time"]
            acc = _smooth(r["accuracy"])
            sem = _smooth(r["accuracy_sem"])
            ax.plot(t, acc, lw=1.2, color=cmap(i), label=name)
            ax.fill_between(t, acc - sem, acc + sem, color=cmap(i), alpha=0.2)
            ch_lo = _smooth(r["chance_ci"][:, 0])
            ch_hi = _smooth(r["chance_ci"][:, 1])
            ax.fill_between(t, ch_lo, ch_hi, color=cmap(i), alpha=0.08)
        ax.axhline(0.5, color="0.6", ls=":", lw=0.8)
        ax.axvline(0, color="k", ls="--", lw=0.8)
        ax.set(xlabel="time from onset (s)", ylabel="decoding accuracy",
               xlim=(t[0], t[-1]))
        ax.legend(fontsize=6, loc="upper right")
        fig.tight_layout()
        return fig
