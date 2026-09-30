"""Arena registration for the Neuropixel open-field session (2026-08-26).

Coordinates: 'orig' = pixels in the 2148x2064 camera frame; 'small' = pixels in the cropped/downscaled
DLC video (crop origin (90, 80), scale 0.4). The arena is a glass box seen from above; the rim polygon
was read off a gridded frame (inspect notebook). The four walls are candidate stimulus locations; which
wall held which stimulus is not recorded in the data folder and must be supplied.
"""
import numpy as np

CROP_X0, CROP_Y0, SCALE = 90, 80, 0.4
RIM_ORIG = np.array([[135, 115], [2015, 105], [2010, 2020], [140, 2030]], float)   # TL, TR, BR, BL
WALLS = ["top", "right", "bottom", "left"]           # wall k runs from RIM[k] to RIM[(k+1)%4]
ARENA_SIDE_CM = 45.7                                  # 1.5 ft glass box (confirmed by user 2026-09-09)
STIM_FRACTION = 0.4                                   # central fraction of each wall treated as the stimulus interface


def to_small(xy):
    xy = np.asarray(xy, float)
    return (xy - [CROP_X0, CROP_Y0]) * SCALE


def to_orig(xy):
    return np.asarray(xy, float) / SCALE + [CROP_X0, CROP_Y0]


class Arena:
    def __init__(self, rim_orig=RIM_ORIG, side_cm=ARENA_SIDE_CM, stim_fraction=STIM_FRACTION, space="small"):
        self.space = space
        self.rim = to_small(rim_orig) if space == "small" else np.asarray(rim_orig, float)
        side_px = np.mean([np.linalg.norm(self.rim[(k + 1) % 4] - self.rim[k]) for k in range(4)])
        self.px_per_cm = side_px / side_cm
        self.center = self.rim.mean(0)
        self.walls = {}
        for k, name in enumerate(WALLS):
            a, b = self.rim[k], self.rim[(k + 1) % 4]
            m = (a + b) / 2; half = (b - a) * stim_fraction / 2
            self.walls[name] = dict(a=a, b=b, seg_a=m - half, seg_b=m + half,
                                    inward=self._inward_normal(a, b))

    def _inward_normal(self, a, b):
        d = b - a; n = np.array([-d[1], d[0]]); n /= np.linalg.norm(n)
        return n if np.dot(self.center - (a + b) / 2, n) > 0 else -n

    @staticmethod
    def _closest_on_segment(p, a, b):
        """Closest point on segment ab to each row of p (N,2)."""
        ab = b - a; t = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1)
        return a + t[:, None] * ab

    def fit_boundaries(self, nose_xy, n_bins=24, pct=0.5, central=0.6):
        """Replace each wall's straight stimulus segment by the empirical reach of the nose.

        For each wall, positions are binned along the wall over its central fraction and the extreme
        percentile of the perpendicular coordinate (towards the wall) gives a polyline. This absorbs
        camera perspective and any obstruction along a wall. Stores polylines in self.walls[w]['poly'].
        """
        p = np.asarray(nose_xy, float); p = p[~np.isnan(p).any(1)]
        for name, w in self.walls.items():
            a, b = w["a"], w["b"]; d = b - a; L = np.linalg.norm(d); u = d / L; nrm = w["inward"]
            along = (p - a) @ u; perp = (p - a) @ nrm                 # perp > 0 is inside the arena
            lo, hi = L * (0.5 - central / 2), L * (0.5 + central / 2)
            edges = np.linspace(lo, hi, n_bins + 1); pts = []
            for e0, e1 in zip(edges[:-1], edges[1:]):
                sel = (along >= e0) & (along < e1) & (perp > -20)
                if sel.sum() > 50:
                    pts.append(a + u * (e0 + e1) / 2 + nrm * np.percentile(perp[sel], pct))
            if len(pts) >= 2:
                w["poly"] = np.array(pts)
        return self

    def split_wall(self, wall, names, order="a_to_b"):
        """Replace one wall by consecutive equal-length sub-targets along its empirical polyline.
        `names` are listed from the wall's point a to point b (order='a_to_b') or reversed."""
        w = self.walls[wall]; poly = w["poly"]
        if order != "a_to_b":
            names = list(names)[::-1]
        seg = np.linalg.norm(np.diff(poly, axis=0), axis=1); cum = np.r_[0, np.cumsum(seg)]; L = cum[-1]
        dense_t = np.linspace(0, L, 200)
        dense = np.stack([np.interp(dense_t, cum, poly[:, 0]), np.interp(dense_t, cum, poly[:, 1])], 1)
        new = {}
        for k, name in enumerate(names):
            sel = (dense_t >= k * L / len(names)) & (dense_t <= (k + 1) * L / len(names))
            new[name] = dict(a=w["a"], b=w["b"], inward=w["inward"], poly=dense[sel], parent=wall)
        walls = {}
        for key, val in self.walls.items():
            if key == wall:
                walls.update(new)
            else:
                walls[key] = val
        self.walls = walls
        return self

    def dist_to_wall(self, p, wall, use_stim_segment=True):
        """Distance (cm) and closest point for points p (N,2) to a wall's interface.
        Uses the empirical polyline if `fit_boundaries` was called, else the straight stimulus segment."""
        w = self.walls[wall]
        if "poly" in w and use_stim_segment:
            poly = w["poly"]; best_d = np.full(len(p), np.inf); best_c = np.zeros_like(p)
            for a, b in zip(poly[:-1], poly[1:]):
                c = self._closest_on_segment(p, a, b); dd = np.linalg.norm(p - c, axis=1)
                m = dd < best_d; best_d[m] = dd[m]; best_c[m] = c[m]
            return best_d / self.px_per_cm, best_c
        a, b = (w["seg_a"], w["seg_b"]) if use_stim_segment else (w["a"], w["b"])
        c = self._closest_on_segment(p, a, b)
        return np.linalg.norm(p - c, axis=1) / self.px_per_cm, c

    def inside(self, p, margin_cm=0.0):
        """Boolean mask: points inside the rim polygon shrunk by margin_cm."""
        from matplotlib.path import Path
        rim = self.center + (self.rim - self.center) * (1 - margin_cm * self.px_per_cm / (np.linalg.norm(self.rim[0] - self.center)))
        return Path(rim).contains_points(p)

    def cm(self, p):
        """Pixels -> cm with origin at the arena centre, y up."""
        q = (np.asarray(p, float) - self.center) / self.px_per_cm
        return q * [1, -1]
