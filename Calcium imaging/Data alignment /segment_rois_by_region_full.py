"""
Segment suite2p ROIs into anatomical regions from a color-filled border image.
White is treated as a real, intentionally-left-white region (kept, not discarded).

Pipeline
--------
1. Load suite2p stat/ops.
2. Load the color-filled border PNG (NEAREST resize -> no invented edge colors).
3. Turn each region's color bounds (incl. white) into a binary mask.
4. Collapse masks into ONE integer label map (each pixel belongs to one region).
5. Assign each ROI to the region holding the MAJORITY of its pixels
   -> every neuron lands in exactly one region; leftovers go to 'unassigned'.
6. Counts, plot, save pickle.

Only dependency beyond your stack: none new (numpy, PIL, matplotlib).
"""

import os
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

# ----------------------------------------------------------------------------
# 0. Config
# ----------------------------------------------------------------------------
mouse_ID = 'IS-2-1L'
date = '2026-07-01'

proc = '/Volumes/Akhil Data/Akhil/ProcessedData/{}/{}'.format(mouse_ID, date)
filled_image_path = ("/Volumes/Akhil Data/Akhil/RawData/{}/cortical borders/{}/"
                     "cortical_borders_filled.png".format(mouse_ID, date))

# ----------------------------------------------------------------------------
# 1. Load suite2p outputs
# ----------------------------------------------------------------------------
stat = np.load(os.path.join(proc, 'suite2p/plane0/stat.npy'), allow_pickle=True)
ops = np.load(os.path.join(proc, 'suite2p/plane0/ops.npy'), allow_pickle=True).item()

Ly, Lx = int(ops['Ly']), int(ops['Lx'])   # full FOV size (do NOT hardcode 1024)

# ----------------------------------------------------------------------------
# 2. Load the filled border image at the FOV size (NEAREST = keep exact colors)
# ----------------------------------------------------------------------------
filled = Image.open(filled_image_path).convert("RGB").resize((Lx, Ly), Image.NEAREST)
img = np.array(filled)   # (Ly, Lx, 3)

# ----------------------------------------------------------------------------
# 3. Region color bounds  (white is a first-class region here)
# ----------------------------------------------------------------------------
regions_bounds = {
    "PPC":   ([0, 120, 0],     [80, 255, 80]),      # green
    "AM":    ([120, 0, 0],     [255, 80, 80]),      # red
    "PM":    ([200, 200, 0],   [255, 255, 100]),    # yellow
    "V1":    ([0, 0, 100],     [100, 100, 255]),    # blue
    "RSC":   ([230, 145, 0],   [255, 149, 30]),     # orange
    "RL":    ([0, 0, 0],       [50, 50, 50]),       # black
    "SMc":   ([100, 100, 100], [200, 200, 200]),    # gray
    "white": ([230, 230, 230], [255, 255, 255]),    # intentionally-white region
}

colors = {
    "PPC": 'g', "AM": 'tomato', "PM": 'y', "V1": 'skyblue',
    "RSC": 'orange', "RL": 'k', "SMc": 'gray', "white": 'magenta',  # plot white as magenta so it's visible
}

region_names = list(regions_bounds.keys())


def color_mask(image, lo, hi):
    lo, hi = np.array(lo), np.array(hi)
    return np.all((image >= lo) & (image <= hi), axis=-1)


# ----------------------------------------------------------------------------
# 4. Build ONE integer label map (0 = background/unassigned, k = region k)
# ----------------------------------------------------------------------------
label_map = np.zeros((Ly, Lx), dtype=np.int32)
overlap = np.zeros((Ly, Lx), dtype=bool)
for k, name in enumerate(region_names, start=1):
    m = color_mask(img, *regions_bounds[name])
    overlap |= (label_map > 0) & m      # track any color-bound collisions
    label_map[m] = k
if overlap.any():
    print(f"WARNING: {int(overlap.sum())} pixels matched >1 region's bounds "
          f"(last-defined region wins there). Tighten bounds if this is large.")


# ----------------------------------------------------------------------------
# 5. Assign each ROI to the region with the MAJORITY of its pixels
# ----------------------------------------------------------------------------
def assign_rois(stat, label_map, region_names, min_pixels=1):
    n = len(region_names)
    out = {name: [] for name in region_names}
    out['unassigned'] = []
    Ly, Lx = label_map.shape
    for idx, roi in enumerate(stat):
        xp = np.asarray(roi['xpix'])
        yp = np.asarray(roi['ypix'])
        keep = (xp >= 0) & (xp < Lx) & (yp >= 0) & (yp < Ly)
        labels = label_map[yp[keep], xp[keep]]
        labels = labels[labels > 0]
        if labels.size < min_pixels:
            out['unassigned'].append(idx)
            continue
        counts = np.bincount(labels, minlength=n + 1)
        out[region_names[int(np.argmax(counts)) - 1]].append(idx)
    return {k: np.array(v, dtype=int) for k, v in out.items()}


roi_indices_by_region = assign_rois(stat, label_map, region_names)

# ----------------------------------------------------------------------------
# 6a. Counts
# ----------------------------------------------------------------------------
for region, idx in roi_indices_by_region.items():
    print(f"{region}: {len(idx)} neurons")

# ----------------------------------------------------------------------------
# 6b. Plot ROIs colored by region
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
ax.imshow(ops['max_proj'], cmap='gist_gray')   # comment out to plot on blank
for name in region_names:
    c = colors[name]
    for idx in roi_indices_by_region[name]:
        roi = stat[idx]
        ax.plot(roi['xpix'], roi['ypix'], 'o', markersize=0.15, color=c, alpha=0.5)
ax.set_xlim([0, Lx]); ax.set_ylim([0, Ly])
ax.set_xticks([]); ax.set_yticks([])
ax.invert_yaxis()
plt.show()

# ----------------------------------------------------------------------------
# 6c. Bar plot of counts (white included)
# ----------------------------------------------------------------------------
names_plot = region_names + ['unassigned']
counts_plot = [len(roi_indices_by_region[n]) for n in names_plot]
bar_colors = [colors.get(n, 'lightgray') for n in names_plot]

fig, ax = plt.subplots(figsize=(3.5, 2.2), dpi=200)
bars = ax.bar(names_plot, counts_plot, color=bar_colors)
ax.set_ylabel('# of neurons')
for tick, n in zip(ax.get_xticklabels(), names_plot):
    tick.set_color(colors.get(n, 'black'))
for b in bars:
    ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
            f'{int(b.get_height())}', ha='center', va='bottom', fontsize=6)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------
# 6d. Save
# ----------------------------------------------------------------------------
with open(os.path.join(proc, 'roi_indices_by_region.pkl'), 'wb') as f:
    pickle.dump(roi_indices_by_region, f)
print("saved roi_indices_by_region.pkl")
