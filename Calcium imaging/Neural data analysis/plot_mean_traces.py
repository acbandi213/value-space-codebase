#!/usr/bin/env python3
"""
Overlay mean population traces for Male, Female, fake mouse, and familiar food.

Usage
-----
    python plot_mean_traces.py

Edit the CONFIG block below to point at your session, tweak axis labels, and
pick colors for each stimulus.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from neural_analysis import NeuralAnalysis, set_style

# =============================================================================
# CONFIG — edit these
# =============================================================================
DATA = "/Volumes/Akhil Data/Akhil/ProcessedData/IS-2-1L/2026-07-01"
pre_frames = 15
fs = 15.06
region = None  # e.g. "ACx"; None = all neurons

STIMULI = ["Male", "Female", "3DMouse", "FamFood"]

# one color per stimulus (same order as STIMULI, or use a dict keyed by stim name)
COLORS = {
    "Male": "#1f77b4",
    "Female": "#ff7f0e",
    "3DMouse": "#2ca02c",
    "FamFood": "#d62728",
}

# legend text (defaults to neural_analysis.STIM_LABELS if omitted)
LEGEND_LABELS = {
    "Male": "male",
    "Female": "female",
    "3DMouse": "fake mouse",
    "FamFood": "familiar food",
}

XLABEL = "time from onset (s)"
YLABEL = "Z-scored ΔF/F"

show_sem = True
smooth_sigma = 0
figsize = (4, 2.5)
dpi = 300
save_path = None  # e.g. "mean_traces.svg"

# =============================================================================
set_style(font="Arial", size=7)


def load_session(data_dir: str, pre_frames: int, fs: float):
    X = np.load(os.path.join(data_dir, "tensor_zdff.npy"))
    labels = np.load(os.path.join(data_dir, "tensor_labels.npy"), allow_pickle=True)
    try:
        regions = np.load(os.path.join(data_dir, "neuron_regions.npy"), allow_pickle=True)
    except FileNotFoundError:
        regions = None
    na = NeuralAnalysis(X, labels, regions=regions, fs=fs, pre_frames=pre_frames)
    na.normalize(method="baseline_sub")
    return na


def main():
    na = load_session(DATA, pre_frames, fs)
    fig = na.plot_mean_traces_overlay(
        stims=STIMULI,
        region=region,
        colors=COLORS,
        legend_labels=LEGEND_LABELS,
        xlabel=XLABEL,
        ylabel=YLABEL,
        show_sem=show_sem,
        smooth_sigma=smooth_sigma,
        figsize=figsize,
        dpi=dpi,
    )
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"saved {save_path}")
    plt.show()


if __name__ == "__main__":
    main()
