"""

dark_study.py — do the darks change with temperature, and does a flight need a dark every time?

    python dark_study.py

Uses the masters already built by make_masters.py and the frames batch_night.py reads.
Part 1  masters at identical gain and exposure compared across nights (pedestal, hot pixels,
        how many hot pixels are the same pixels, master-to-master difference), next to the
        ambient temperature of each night (Table A.1) and the IMU temperature if logged.
Part 2  swap masters between nights: night 3's static block (gain x1.0, 4545 rows) corrected with
        its own master and with nights 2 and 4's; night 3's 20 ms rung (gain x7.66, 900 rows) with
        its own and night 1's. Solve rate, matched inliers and Tetra3 RMSE for each, plus no dark.
Part 3  a dark from the sky: the median of the OTHER frames of the static block (stars move
        ~56 px between frames 3 min apart, hot pixels do not) used as the master, no lens cap.

Pedro Clemente, 2026-10-03

"""
import csv, glob, os
from datetime import datetime, timezone
import numpy as np
from scipy.ndimage import median_filter
import tifffile as tiff
import tetra3
import batch_night as bn
import make_masters as mm
from stilter_filter import stilter
from final_estimator import estimator
from matched_centroids import get_centroids_matched

AMBIENT = {1: "20->17 C", 2: "26->24 C", 3: "21->16 C", 4: "n/a"}      # Table A.1, start -> end


def mdir(night):
    N = bn.NIGHTS[night]; return f"{N['root']}/{N['masters']}"


def load_master(night, gain, exp):
    base = f"{mdir(night)}/master_g{gain}_e{exp}"
    if not os.path.exists(base + "_dark.npy"):
        return None, None
    return np.load(base + "_dark.npy"), np.load(base + "_hot.npy")


def imu_temp(night, gain, exp):
    """Mean IMU temperature logged with the dark frames, if their folder holds a log."""
    if night not in mm.NIGHTS:
        return None
    root, _, sets = mm.NIGHTS[night]
    for g, e, rel in sets:
        if (g, e) == (gain, exp):
            for p in glob.glob(f"{root}/{rel}/*.csv"):
                with open(p) as f:
                    t = [float(r["temp_c"]) for r in csv.DictReader(f) if r.get("temp_c") not in (None, "", "nan")]
                if t:
                    return float(np.mean(t))
    return None


def correct(img, master, hot):
    out = img - master + float(np.median(master))
    nb = median_filter(out, size=3, mode="reflect")
    out[hot] = nb[hot]
    return out


def hot_mask(master, gain):
    med = float(np.median(master))
    mad = max(float(np.median(np.abs(master - med)) * 1.4826), mm.MAD_FLOOR)
    return master > med + mm.HOT_SIGMA.get(gain, 5.0) * mad


def frames(night, block):
    N = bn.NIGHTS[night]
    for name, csv_path, img_dir in bn._blocks(N["root"], N["layout"]):
        if name != block:
            continue
        with open(csv_path) as f:
            rows = {r["filename"].strip(): r for r in csv.DictReader(f)}
        out = []
        for fp in sorted(glob.glob(os.path.join(img_dir, "*.tif*"))):
            r = rows.get(os.path.basename(fp))
            if r is not None:
                out.append((tiff.imread(fp).astype(np.float32), r))
        return out
    return []


def solve(t3, img, gain, row):
    h, w = img.shape
    cents = get_centroids_matched(img, fwhm_px=bn.FWHM_PX, **bn.FILTER.get(gain, bn.FILTER[32]))
    if len(cents) < 3:
        return dict(ok=False, inl=0, rmse=np.nan)
    sol = t3.solve_from_centroids(cents, size=img.shape, fov_estimate=bn.FOV_ESTIMATE,
                                  fov_max_error=bn.FOV_MAX_ERROR, return_matches=True)
    if sol.get("RA") is None:
        return dict(ok=False, inl=0, rmse=np.nan)
    utc = datetime.fromtimestamp(float(row["board_timestamp"]) + bn.CLOCK_OFFSET_S, tz=timezone.utc)
    q = stilter(tolerance_arcsec=bn.TOLERANCE_ARCSEC).process(sol=sol, utc_dt=utc, img_width=w, img_height=h,
                                                               estimator_cls=estimator)
    return dict(ok=bool(q.get("success")), inl=int(q.get("inliers_kept", 0)), rmse=float(sol.get("RMSE", np.nan)))


def run(t3, fr, gain, label, master=None, hot=None, sky=False):
    res = []
    for i, (img, row) in enumerate(fr):
        if sky:
            others = np.stack([f[0] for j, f in enumerate(fr) if j != i])
            m = np.median(others, axis=0); h = hot_mask(m, gain)
            im = correct(img, m, h)
        elif master is not None:
            im = correct(img, master, hot)
        else:
            im = img
        res.append(solve(t3, im, gain, row))
    ok = [r for r in res if r["ok"]]
    print(f"    {label:34s} solved {len(ok):2d}/{len(res):<2d}  inliers {np.mean([r['inl'] for r in ok]) if ok else 0:5.1f}  "
          f"RMSE {np.nanmean([r['rmse'] for r in ok]) if ok else np.nan:5.1f}\"")


def main():
    # ---------------- Part 1 ----------------
    print("PART 1 - masters at the same gain and exposure, night by night")
    settings = sorted({(g, e) for n in mm.NIGHTS for g, e, _ in mm.NIGHTS[n][2]})
    for g, e in settings:
        have = [(n, *load_master(n, g, e)) for n in sorted(bn.NIGHTS)]
        have = [(n, m, h) for n, m, h in have if m is not None]
        if len(have) < 2:
            continue
        print(f"\n  gain {g}, {e} rows ({e * 22.2 / 1000:.1f} ms)")
        print(f"    {'night':>5} {'ambient':>10} {'IMU':>7} {'pedestal':>8} {'noise':>6} {'hot px':>7} {'hot excess':>10}")
        for n, m, h in have:
            med = float(np.median(m)); good = m[~h]
            t = imu_temp(n, g, e)
            print(f"    {n:>5} {AMBIENT.get(n, 'n/a'):>10} {('%.1f C' % t) if t else '   n/a':>7} {med:8.2f} "
                  f"{1.4826 * np.median(np.abs(good - np.median(good))):6.2f} {int(h.sum()):7d} {float(np.mean(m[h]) - med) if h.any() else 0:10.2f}")
        for i in range(len(have)):
            for j in range(i + 1, len(have)):
                (ni, mi, hi), (nj, mj, hj) = have[i], have[j]
                inter, union = (hi & hj).sum(), (hi | hj).sum()
                d = mi - mj
                print(f"    nights {ni}-{nj}: same hot pixels {inter}/{union} ({100 * inter / max(union, 1):.0f} %)   "
                      f"difference, ordinary pixels sd {d[~(hi | hj)].std():.2f}   hot pixels sd {d[hi | hj].std() if union else 0:.2f}")

    # ---------------- Part 2 and 3 ----------------
    t3 = tetra3.Tetra3(load_database=bn.DATABASE)
    print("\nPART 2 - night 3 corrected with masters from other nights")
    for block, gain, exp, sources in (("static_B", 32, 4545, (3, 2, 4)), ("ladder_900", 245, 900, (3, 1))):
        fr = frames(3, block)
        if not fr:
            print(f"  {block}: no frames"); continue
        print(f"\n  night 3 {block} (gain {gain}, {exp} rows), {len(fr)} frames")
        run(t3, fr, gain, "no dark (raw, matched filter)")
        for src in sources:
            m, h = load_master(src, gain, exp)
            if m is None:
                print(f"    master from night {src}: none at these settings"); continue
            run(t3, fr, gain, f"master from night {src}" + (" (own)" if src == 3 else ""), m, h)
        if block == "static_B":
            print("\nPART 3 - a dark built from the sky frames themselves (no lens cap)")
            run(t3, fr, gain, "median of the other static frames", sky=True)


if __name__ == "__main__":
    main()