"""

blur_test.py — how much smear does the solver tolerate on real, corrected frames?

Takes real frames (dark-corrected, as batch_night.py prepares them), convolves each
with a line kernel of L pixels — the trail a star leaves under uniform rotation of
omega deg/s over exposure t — and runs the same detector + Tetra3 + stilter chain.
Solve rate against smear in arcseconds is the measured epsilon_tol

    python3 blur_test.py --night 3 --block static
    python3 blur_test.py --night 3 --block ladder_1100     # the multirotor rung itself

Smear L (px) = omega (deg/s) * t (s) * 3600 / 48.34 ("/px).

Pedro Clemente, 2026-09-27

"""
import argparse
import csv
import glob
import math
import os
import numpy as np
from datetime import datetime, timezone
from scipy.ndimage import convolve

import tifffile as tiff
import tetra3
from stilter_filter import stilter
from final_estimator import estimator
from matched_centroids import get_centroids_matched
import batch_night as bn

from scipy.ndimage import gaussian_filter

np.math = math
PLATE = 48.34                                   # arcsec per pixel
SMEAR_PX = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24]    # trail lengths to test
ANGLES = [0, 45, 90]                            # trail directions, degrees; averaged
NOISE_ADU = {32: 0.34, 245: 2.58}     # measured temporal noise per gain, 8-bit ADU

def line_kernel(L, angle_deg):
    """Normalised line kernel of length L pixels at the given angle."""
    if L < 1:
        return np.ones((1, 1))
    n = int(np.ceil(L)) + 2
    k = np.zeros((n, n)); c = (n - 1) / 2
    for s in np.linspace(-L / 2, L / 2, max(2, int(L * 4))):
        y = c + s * np.sin(np.radians(angle_deg)); x = c + s * np.cos(np.radians(angle_deg))
        yi, xi = int(round(y)), int(round(x))
        if 0 <= yi < n and 0 <= xi < n:
            k[yi, xi] += 1
    return k / k.sum()


def solve(t3, img, row, gain):
    h, w = img.shape
    cents = get_centroids_matched(img, fwhm_px=bn.FWHM_PX, **bn.FILTER.get(gain, bn.FILTER[32]))
    if len(cents) < 3:
        return False, 0, 0
    sol = t3.solve_from_centroids(cents, size=img.shape, fov_estimate=bn.FOV_ESTIMATE,
                                  fov_max_error=bn.FOV_MAX_ERROR, return_matches=True)
    if sol.get("RA") is None:
        return False, len(cents), 0
    utc = datetime.fromtimestamp(float(row["board_timestamp"]) + bn.CLOCK_OFFSET_S, tz=timezone.utc)
    q = stilter(tolerance_arcsec=bn.TOLERANCE_ARCSEC).process(
        sol=sol, utc_dt=utc, img_width=w, img_height=h, estimator_cls=estimator)
    return bool(q.get("success")), len(cents), int(q.get("inliers_kept", 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--night", type=int, required=True)
    ap.add_argument("--block", default="static")
    ap.add_argument("--max-frames", type=int, default=6)
    ap.add_argument("--kernel", default="line", choices=("line", "gauss"))
    a = ap.parse_args()
    N = bn.NIGHTS[a.night]; root, mdir = N["root"], f"{N['root']}/{N['masters']}"
    blocks = [b for b in bn._blocks(root, N["layout"]) if a.block in b[0]]
    if not blocks:
        raise SystemExit(f"no block matching '{a.block}'")
    name, csv_path, img_dir = blocks[0]
    with open(csv_path) as f:
        rows = {r["filename"].strip(): r for r in csv.DictReader(f)}
    files = sorted(glob.glob(os.path.join(img_dir, "*.tif*")))[:a.max_frames]
    t3 = tetra3.Tetra3(load_database=bn.DATABASE)

    frames = []
    for fp in files:
        row = rows.get(os.path.basename(fp))
        if row is None:
            continue
        gain, exp = int(float(row["gain_reg"])), int(float(row["exposure_reg"]))
        img, note = bn.prepare(fp, mdir, gain, exp, True)
        if "dark" not in note:
            print(f"  {os.path.basename(fp)}: no matched master, skipped"); continue
        frames.append((img, row, gain, exp))
    if not frames:
        raise SystemExit("no corrected frames")
    exp_ms = frames[0][3] * 22.2 / 1000.0
    print(f"{name}: {len(frames)} corrected frames at {exp_ms:.1f} ms, gain {frames[0][2]}\n")
    print(f"{'trail px':>8} {'smear \"':>8} {'omega deg/s':>12} {'solved':>8} {'centroids':>10} {'inliers':>8}")
    print("-" * 60)
    rng = np.random.default_rng(0)
    for L in SMEAR_PX:
        ok = tot = 0; cents = []; inl = []
        for angle in (ANGLES if (L > 0 and a.kernel == "line") else [0]):
            k = line_kernel(L, angle)
            for img, row, gain, exp in frames:
                sig = NOISE_ADU.get(gain, 0.34)
                if L == 0:
                    b = img
                elif a.kernel == "gauss":
                    b = gaussian_filter(img, sigma=L / 2.3548)            # L is FWHM in px
                    b = b + rng.normal(0, sig * np.sqrt(max(1 - 1 / (L * L), 0)), b.shape)
                else:
                    b = convolve(img, k, mode="reflect")
                    b = b + rng.normal(0, sig * np.sqrt(1 - 1 / L), b.shape)
                s, nc, ni = solve(t3, b, row, gain)
                tot += 1; ok += s; cents.append(nc)
                if s: inl.append(ni)
        smear = L * PLATE
        omega = smear / 3600.0 / (exp_ms / 1000.0)
        print(f"{L:>8d} {smear:>8.0f} {omega:>12.2f} {ok:>4d}/{tot:<3d} {np.mean(cents):>10.1f} "
              f"{(np.mean(inl) if inl else 0):>8.1f}")
    print("\nepsilon_tol is the largest smear with solve rate still ~1. Section 6.3 assumed 100\".")
    print("omega is the rotation rate that produces that smear at this exposure.")


if __name__ == "__main__":
    main()