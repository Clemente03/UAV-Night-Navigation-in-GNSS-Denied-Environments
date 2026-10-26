"""

batch_night.py — solve every frame from one night and write results_night<N>.csv

    python batch_night.py --night 3                       #corrected, accelerometer mean vertical
    python batch_night.py --night 3 --vertical euler      #flight controller Euler vertical
    python batch_night.py --night 1 --no-dark             #raw frames, tetra3's own centroider
    python batch_night.py --night 3 --block static        #console only: does NOT rewrite the CSV

Per frame: matched exposure dark + neighbour repaired hot pixels + matched filter
detection (falls back to raw + tetra3's centroider when no matched master exists) ->
tetra3 -> stilter -> estimator. Vertical from --vertical
Geometric dilution G per solve

Nights 1 and 2 logged no accelerometer, so --vertical falls back to euler for them

Pedro Clemente, 2026-09-27

"""

import argparse
import csv
import glob
import math
import os
import sys
import numpy as np
from datetime import datetime, timezone
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation as R

try:
    import tifffile as tiff
except ImportError:
    sys.exit("pip install tifffile")

import tetra3
from stilter_filter import stilter
from final_estimator import estimator
from matched_centroids import get_centroids_matched

np.math = math


#edit
IMAGES = "/Users/pedroclemente/Documents/Tese/Images"
TRUE_LAT, TRUE_LON = 38.981479, -8.300340        #surveyed benchmark
DATABASE = "t3_fov10-23_mag8.npz"
R_OFFSET = "R_offset.npy"
CLOCK_OFFSET_S = -3600.0                          #board on local WEST time
FOV_ESTIMATE, FOV_MAX_ERROR = 17.06, 2.0
TOLERANCE_ARCSEC = 200.0
FWHM_PX = 2.0

#matched filter settings by gain register (noise is 7.66x larger in ADU at 245)
FILTER = {32:  dict(sigma=3.0, noise_floor=0.3, min_peak=2.0, max_returned=40),
          245: dict(sigma=3.0, noise_floor=1.0, min_peak=8.0, max_returned=40)}


def _blocks(root, layout):
    #block list for one night. layout picks the folder convention.
    if layout == 1:      #18_09_2026
        B = [(f"ladder_{e}", f"{root}/Exposure Ladder/{e}/{e}.csv", f"{root}/Exposure Ladder/{e}")
             for e in (45, 91, 182, 255, 455, 727, 900, 1100, 2273, 4545, 9091)]
        B += [("static_B", f"{root}/Static/Static_B/static_b.csv", f"{root}/Static/Static_B"),
              ("gain32_ctrl", f"{root}/Static/Static_gain1_exp4545/g1_4545.csv", f"{root}/Static/Static_gain1_exp4545")]
        B += [(f"boresight_{i}", f"{root}/Boresight/Boresight_{i}/boresight_{i}.csv", f"{root}/Boresight/Boresight_{i}")
              for i in range(1, 6)]
    elif layout == 2:    #19_09_2026
        B = [(f"ladder_{e}", f"{root}/Analog_gain/Exposure Ladder/{e}/{e}.csv", f"{root}/Analog_gain/Exposure Ladder/{e}")
             for e in (45, 91, 182, 255, 455, 727, 900, 1100, 2273, 4545, 9091)]
        B += [("static_B", f"{root}/Static/static.csv", f"{root}/Static")]
        B += [(f"boresight_{i}", f"{root}/Boresight/Boresight_{i}/boresight_{i}.csv", f"{root}/Boresight/Boresight_{i}")
              for i in range(1, 11)]
    elif layout == 3:    #25_09_2026
        B = [("static_B", f"{root}/Static/static.csv", f"{root}/Static")]
        B += [(f"boresight_{i}", f"{root}/Boresight/Boresight_{i}/boresight_{i}.csv", f"{root}/Boresight/Boresight_{i}")
              for i in range(1, 11)]
        B += [(f"ladder_{e}", f"{root}/Exposure Ladder/{e}/{e}.csv", f"{root}/Exposure Ladder/{e}")
              for e in (900, 1100, 1500, 2273, 3000, 4545)]
    else:                #26_09_2026
        B = [("static_B", f"{root}/Static/static.csv", f"{root}/Static")]
        B += [(f"boresightA_{i}", f"{root}/Boresight_A/Boresight_{i}/boresight_{i}.csv", f"{root}/Boresight_A/Boresight_{i}")
              for i in range(1, 11)]
        B += [(f"boresightB_{i}", f"{root}/Boresight_B/Boresight_{i}/boresight_{i}.csv", f"{root}/Boresight_B/Boresight_{i}")
              for i in range(1, 11)]
        B += [(f"ladder_{e}", f"{root}/Exposure Ladder/{e}/{e}.csv", f"{root}/Exposure Ladder/{e}")
              for e in (255, 455)]
    return B


NIGHTS = {
    1: dict(root=f"{IMAGES}/18_09_2026", masters="Darks/masters", layout=1),
    2: dict(root=f"{IMAGES}/19_09_2026", masters="masters",       layout=2),
    3: dict(root=f"{IMAGES}/25_09_2026", masters="masters",       layout=3),
    4: dict(root=f"{IMAGES}/26_09_2026", masters="masters",       layout=4),
}
KM_PER_DEG = 111.32


#geometric dilution
def dilution(V, z):
    if len(V) < 3:
        return float("nan")
    M = sum(np.eye(3) - np.outer(v, v) for v in V)
    Mi = np.linalg.pinv(M)
    return float(np.sqrt(len(V) * max(np.trace(Mi) - z @ Mi @ z, 0.0)))


def body_vectors(centroids, mask, w, h, fov_deg):
    f = (w / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    c = np.asarray(centroids)[np.asarray(mask, bool)]
    v = np.column_stack((c[:, 1] - w / 2.0, c[:, 0] - h / 2.0, np.full(len(c), f)))
    return v / np.linalg.norm(v, axis=1)[:, None]


#masters
_cache = {}

def get_master(mdir, gain, exp):
    key = (gain, exp)
    if key not in _cache:
        base = f"{mdir}/master_g{gain}_e{exp}"
        _cache[key] = ((np.load(base + "_dark.npy"), np.load(base + "_hot.npy"))
                       if os.path.exists(base + "_dark.npy") else (None, None))
    return _cache[key]


def prepare(path, mdir, gain, exp, use_dark):
    img = tiff.imread(path).astype(np.float32)
    if not use_dark:
        return img, "raw"
    master, hot = get_master(mdir, gain, exp)
    if master is None or master.shape != img.shape:
        return img, "raw (no matched master)"
    img = img - master + float(np.median(master))
    nb = median_filter(img, size=3, mode="reflect")
    img[hot] = nb[hot]
    return img, "dark+hot"


#one frame
def solve_one(t3, img, row, r_offset, note, gain, vertical):
    h, w = img.shape
    if "dark" in note:
        centroids = get_centroids_matched(img, fwhm_px=FWHM_PX, **FILTER.get(gain, FILTER[32]))
    else:
        centroids = tetra3.get_centroids_from_image(img, sigma=2.0)
    out = {"n_centroids": len(centroids)}
    if len(centroids) < 3:
        out["status"] = "no_centroids"; return out

    sol = t3.solve_from_centroids(centroids, size=img.shape, fov_estimate=FOV_ESTIMATE,
                                  fov_max_error=FOV_MAX_ERROR, return_matches=True)
    if sol.get("RA") is None:
        out["status"] = "no_match"; return out
    out.update({"ra": sol["RA"], "dec": sol["Dec"], "roll": sol["Roll"], "fov": sol.get("FOV"),
                "t3_rmse": sol.get("RMSE"), "n_matches": len(sol.get("matched_stars", []))})

    utc = datetime.fromtimestamp(float(row["board_timestamp"]) + CLOCK_OFFSET_S, tz=timezone.utc)
    q = stilter(tolerance_arcsec=TOLERANCE_ARCSEC).process(
        sol=sol, utc_dt=utc, img_width=w, img_height=h, estimator_cls=estimator)
    if not q.get("success"):
        out["status"] = "filter_failed: " + str(q.get("reason")); return out
    out["n_inliers"] = q["inliers_kept"]
    out["qx"], out["qy"], out["qz"], out["qw"] = [float(v) for v in q["quaternion"]]
    for k in ("rms_arcsec", "max_residual_arcsec"):
        if k in q: out[k] = q[k]

    #vertical
    yaw, pitch, roll = (float(row["imu_yaw"]), float(row["imu_pitch"]), float(row["imu_roll"]))
    vz, used = None, "euler"
    if vertical != "euler":
        k = "_mean" if vertical == "acc_mean" else ""
        try:
            cand = [float(row[f"acc_x{k}"]), float(row[f"acc_y{k}"]), float(row[f"acc_z{k}"])]
            if not any(np.isnan(cand)):
                vz, used = cand, vertical
        except (KeyError, ValueError):
            pass
    out["vertical"] = used

    loc = estimator.estimate_lat_lon_analytical(
        q_engine_result=q, utc_dt=utc, imu_yaw=yaw, imu_pitch=pitch, imu_roll=roll,
        r_offset_matrix=r_offset, dut1=0, v_zenith_body=vz)
    if not loc.get("success"):
        out["status"] = "estimator_failed"; return out

    #dilution
    z_imu = (np.asarray(vz, float) if vz is not None else
             R.from_euler("ZYX", [yaw, 90.0 - pitch, roll], degrees=True).inv().apply([0, 0, -1.0]))
    z_cam = r_offset @ (z_imu / np.linalg.norm(z_imu))
    V = body_vectors(q["matched_centroids"], q["inlier_mask"], w, h, sol.get("FOV", FOV_ESTIMATE))
    out["dilution_G"] = dilution(V, z_cam / np.linalg.norm(z_cam))

    lat, lon = loc["lat_deg"], loc["lon_deg"]
    dlat = (lat - TRUE_LAT) * KM_PER_DEG
    dlon = (lon - TRUE_LON) * KM_PER_DEG * math.cos(math.radians(TRUE_LAT))
    out.update({"status": "ok", "est_lat": lat, "est_lon": lon, "err_lat_km": dlat,
                "err_lon_km": dlon, "radial_error_km": math.hypot(dlat, dlon)})
    return out


#main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--night", type=int, required=True, choices=sorted(NIGHTS))
    ap.add_argument("--vertical", default="acc_mean", choices=("euler", "acc", "acc_mean"))
    ap.add_argument("--no-dark", action="store_true")
    ap.add_argument("--block", default=None, help="substring; console only, CSV not written")
    a = ap.parse_args()
    N = NIGHTS[a.night]; root, mdir = N["root"], f"{N['root']}/{N['masters']}"
    out_csv = f"{root}/results_night{a.night}.csv"

    try:
        r_offset = np.load(R_OFFSET)
    except Exception as e:
        print(f"No {R_OFFSET} ({e}) - identity; positions meaningless, solve counts valid.")
        r_offset = np.eye(3)
    t3 = tetra3.Tetra3(load_database=DATABASE)
    print(f"night {a.night}   dark {'OFF' if a.no_dark else 'ON'}   vertical {a.vertical}   "
          f"masters {mdir}\n")

    results = []
    for name, csv_path, img_dir in _blocks(root, N["layout"]):
        if a.block and a.block not in name:
            continue
        rows = {}
        if os.path.exists(csv_path):
            with open(csv_path) as f:
                rows = {r["filename"].strip(): r for r in csv.DictReader(f)}
        files = sorted(glob.glob(os.path.join(img_dir, "*.tif*")))
        if not files:
            print(f"{name:14s} no images"); continue
        ok = n = 0
        for fp in files:
            base = os.path.basename(fp); row = rows.get(base)
            if row is None:
                continue
            gain, exp = int(float(row["gain_reg"])), int(float(row["exposure_reg"]))
            img, note = prepare(fp, mdir, gain, exp, not a.no_dark)
            r = solve_one(t3, img, row, r_offset, note, gain, a.vertical)
            r.update({"block": name, "filename": base, "gain_reg": gain, "exposure_reg": exp,
                      "dark": note, "board_timestamp": row["board_timestamp"]})
            results.append(r); n += 1; ok += r.get("status") == "ok"
        print(f"{name:14s} {ok:3d}/{n:3d} solved")

    if not results:
        print("nothing processed"); return
    if not a.block:
        keys = ["block", "filename", "board_timestamp", "gain_reg", "exposure_reg", "dark", "vertical",
                "status", "n_centroids", "n_matches", "n_inliers", "t3_rmse", "rms_arcsec",
                "max_residual_arcsec", "fov", "ra", "dec", "roll", "dilution_G",
                "est_lat", "est_lon", "err_lat_km", "err_lon_km", "radial_error_km",
                "qx", "qy", "qz", "qw"]
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader()
            for r in results: w.writerow(r)
        print(f"\nwritten: {out_csv}")

    lad = [r for r in results if r["block"].startswith("ladder")]
    if lad:
        print(f"\n{'rows':>6} {'ms':>6} {'solved':>7} {'centroids':>10} {'inliers':>8} {'err km':>7}")
        for e in sorted({r["exposure_reg"] for r in lad}):
            s = [r for r in lad if r["exposure_reg"] == e]; g = [r for r in s if r.get("status") == "ok"]
            print(f"{e:>6} {e*0.0222:>6.1f} {len(g):>3}/{len(s):<3} {np.mean([r['n_centroids'] for r in s]):>10.1f} "
                  f"{(np.mean([r['n_inliers'] for r in g]) if g else 0):>8.1f} "
                  f"{(np.mean([r['radial_error_km'] for r in g]) if g else float('nan')):>7.1f}")
    st = sorted([r for r in results if r["block"] == "static_B" and r.get("status") == "ok"],
                key=lambda r: float(r["board_timestamp"]))
    if st:
        e = np.array([r["radial_error_km"] for r in st])
        print(f"\nSTATIC n={len(st)}  mean {e.mean():.2f}  CEP50 {np.median(e):.2f}  sigma {e.std(ddof=1):.2f}  "
              f"max {e.max():.2f} km   inliers {np.mean([r['n_inliers'] for r in st]):.1f}   "
              f"G {np.nanmean([r['dilution_G'] for r in st]):.2f}")
        print("  in time order: " + " ".join(f"{v:.1f}" for v in e))


if __name__ == "__main__":
    main()