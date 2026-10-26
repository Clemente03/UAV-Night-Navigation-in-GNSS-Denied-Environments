"""

calibrate_rig.py — fit the camera-to-IMU rotation R_offset and compare vertical sources

    python calibrate_rig.py            #convention sweep, source comparison, nothing written
    python calibrate_rig.py --save     #writes R_offset.npy from SAVE_SOURCE

for every solved frame the script has two statements of the local vertical:
    v_cam  - from the star solution and the surveyed truth (where the zenith is on the sky
             at that instant, rotated into the camera frame by the frame's quaternion)
    v_imu  - from the IMU, by one of four sources:
               euler      the flight controller's fused attitude (Euler angles)
               acc        the raw accelerometer sample logged with the frame
               acc_block  the median of the raw samples over all frames of that attitude
               acc_mean   the 30-s buffer mean logged with the frame
R_offset is the single rotation that best maps every v_imu onto its v_cam. The residual
per frame is the angle between R_offset·v_imu and v_cam; leave-one-attitude-out gives
the held-out figure to quote.

Nights 1-2 logged Euler only: for those, every accelerometer source falls back to Euler

Pedro Clemente, 2026-09-27

"""

import csv
import glob
import os
import sys
import numpy as np
import erfa
from datetime import datetime, timezone
from scipy.spatial.transform import Rotation as R

#edit
IMAGES = "/Users/pedroclemente/Documents/Tese/Images"
ALL_NIGHTS = {1: f"{IMAGES}/18_09_2026", 2: f"{IMAGES}/19_09_2026",
              3: f"{IMAGES}/25_09_2026", 4: f"{IMAGES}/26_09_2026"}
#python calibrate_rig.py --nights 3  (default) or --nights 3 4
_sel = [int(x) for x in sys.argv[sys.argv.index("--nights") + 1:] if x.isdigit()] if "--nights" in sys.argv else [3]
NIGHTS = [(ALL_NIGHTS[n], f"results_night{n}.csv") for n in _sel]
TRUE_LAT = 38.981479
TRUE_LON = -8.300340
CLOCK_OFFSET_S = -3600.0            #board on local WEST time; set 0.0 if it was on UTC
DUT1 = 0.0
SAVE_SOURCE = "acc_block"           #which vertical source --save writes R_offset from
EXCLUDE = ["25_09_boresight_4"]     #disturbed attitude
EXCLUDE_PREFIX = ("ladder",)        #same attitude copies of the static block: never in the fit
SOURCES = ("euler", "acc", "acc_block", "acc_mean")
ARCSEC = 3600.0

def jd_from_datetime(dt):
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp() / 86400.0 + 2440587.5


def era_of(jd, dut1=DUT1):
    a, b = erfa.utcut1(2451545.0, jd - 2451545.0, dut1)
    return erfa.era00(a, b)


def icrs_to_cirs(jd):
    return erfa.c2i06a(2451545.0, jd + 69.184 / 86400.0 - 2451545.0)


def q2m(q):
    qx, qy, qz, qw = q
    return np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]])


def _vec(row, prefix, suffix=""):
    try:
        v = [float(row[f"{prefix}_x{suffix}"]), float(row[f"{prefix}_y{suffix}"]), float(row[f"{prefix}_z{suffix}"])]
        return None if any(np.isnan(v)) else v
    except (KeyError, ValueError, TypeError):
        return None

def load():
    caps = []
    for root, res_name in NIGHTS:
        tag = os.path.basename(root)[:5]
        res = {}
        with open(f"{root}/{res_name}") as f:
            for r in csv.DictReader(f):
                if r.get("status") == "ok" and r.get("qw"):
                    res[r["filename"].strip()] = r
        for path in sorted(glob.glob(f"{root}/**/*.csv", recursive=True)):
            if os.path.basename(path).startswith("results"):
                continue
            with open(path) as f:
                for row in csv.DictReader(f):
                    if "board_timestamp" not in row:
                        continue
                    name = row["filename"].strip()
                    if name not in res:
                        continue
                    rr = res[name]
                    caps.append({
                        "name": name,
                        "att": f"{tag}_{rr['block']}",
                        "utc": datetime.fromtimestamp(float(row["board_timestamp"]) + CLOCK_OFFSET_S,
                                                      tz=timezone.utc),
                        "imu": (float(row["imu_yaw"]), float(row["imu_pitch"]), float(row["imu_roll"])),
                        "q": np.array([float(rr[k]) for k in ("qx", "qy", "qz", "qw")]),
                        "acc": _vec(row, "acc"),
                        "acc_mean": _vec(row, "acc", "_mean"),
                    })
    #per attitude median of the raw accelerometer samples
    for att in {c["att"] for c in caps}:
        grp = [c for c in caps if c["att"] == att and c["acc"] is not None]
        if grp:
            med = np.median(np.array([c["acc"] for c in grp]), axis=0)
            for c in grp:
                c["acc_block"] = med
    caps = [c for c in caps if c["att"] not in EXCLUDE
            and not any(c["att"].split("_", 2)[-1].startswith(p) for p in EXCLUDE_PREFIX)]
    return caps


def vectors(cap, conv, zsign, precess, source="euler"):
    jd = jd_from_datetime(cap["utc"])
    era = era_of(jd)
    dec_r = np.radians(TRUE_LAT)
    ra_r = np.radians(TRUE_LON) + era
    v_true = np.array([np.cos(dec_r) * np.cos(ra_r), np.cos(dec_r) * np.sin(ra_r), np.sin(dec_r)])
    if precess:
        v_true = icrs_to_cirs(jd).T @ v_true
    #the quaternion maps geometric catalogue directions to the apparent sky; feed it the
    #geometric direction of the star that APPEARS at the zenith (true zenith minus aberration)
    _, pvb = erfa.epv00(2451545.0, jd + 69.2 / 86400.0 - 2451545.0)
    beta = np.asarray(pvb[1]) / 173.1446326846693
    v_true = v_true - beta + np.dot(v_true, beta) * v_true
    v_true = v_true / np.linalg.norm(v_true)
    v_cam = q2m(cap["q"]) @ v_true
    v_cam = v_cam / np.linalg.norm(v_cam)

    src = cap.get(source) if source != "euler" else None
    if src is not None:
        v_imu = np.asarray(src, dtype=float)
    else:
        y, p, r = cap["imu"]
        Ri = R.from_euler(conv, [y, 90.0 - p, r], degrees=True)    #driver logs pitch as 90 - theta
        v_imu = Ri.inv().apply([0.0, 0.0, zsign])
    v_imu = v_imu / np.linalg.norm(v_imu)
    return v_cam, v_imu


def fit(caps, conv, zsign, precess, source="euler"):
    cams, imus = [], []
    for c in caps:
        vc, vi = vectors(c, conv, zsign, precess, source)
        cams.append(vc); imus.append(vi)
    cams, imus = np.array(cams), np.array(imus)
    R_off, _ = R.align_vectors(cams, imus)
    dots = np.clip(np.einsum("ij,ij->i", R_off.apply(imus), cams), -1, 1)
    return R_off, np.degrees(np.arccos(dots)) * ARCSEC, cams, imus


def held_out(caps, conv, zsign, precess, source):
    #leave one attitude out RMS residual, arcsec
    held = []
    atts = sorted({c["att"] for c in caps})
    for a in atts:
        tr = [c for c in caps if c["att"] != a]
        te = [c for c in caps if c["att"] == a]
        if len({c["att"] for c in tr}) < 3:
            continue
        R_tr, _, _, _ = fit(tr, conv, zsign, precess, source)
        for c in te:
            vc, vi = vectors(c, conv, zsign, precess, source)
            held.append(np.degrees(np.arccos(np.clip(R_tr.apply(vi) @ vc, -1, 1))) * ARCSEC)
    return float(np.sqrt(np.mean(np.square(held)))) if held else float("nan")


def main():
    caps = load()
    atts = sorted({c["att"] for c in caps})
    if len(atts) < 3:
        sys.exit(f"only {len(atts)} attitudes with solved frames - need at least 3")
    n_acc = sum(c.get("acc") is not None for c in caps)
    print("=" * 74)
    print(f"{len(caps)} solved frames across {len(atts)} attitudes  ({n_acc} with accelerometer)")
    print("=" * 74)
    for a in atts:
        g = [c for c in caps if c["att"] == a]
        print(f"  {a:26s} {len(g):3d} frames   pitch {np.mean([c['imu'][1] for c in g]):7.2f}  "
              f"roll {np.mean([c['imu'][2] for c in g]):8.2f}"
              + ("" if g[0].get("acc") is None else "   acc ok"))

    #convention sweep, Euler source
    print("\n" + "=" * 74)
    print("Euler convention sweep (lowest residual wins)")
    print("=" * 74)
    print(f"{'euler':>6} {'zenith':>7} {'precess':>8} | {'R_off':>8} {'RMS':>9} {'max':>9}")
    best = None
    for conv in ("zyx", "ZYX"):
        for zs in (+1.0, -1.0):
            for pr in (False, True):
                R_off, res, _, _ = fit(caps, conv, zs, pr, "euler")
                rms = float(np.sqrt(np.mean(res ** 2)))
                print(f"{conv:>6} {'+z' if zs > 0 else '-z':>7} {'yes' if pr else 'no':>8} | "
                      f"{np.degrees(R_off.magnitude()):7.2f}d {rms:8.1f}\" {res.max():8.1f}\"")
                if best is None or rms < best[0]:
                    best = (rms, conv, zs, pr)
    _, conv, zs, pr = "ZYX", "ZYX", -1.0, False   # forced: proven on nights 1-2
    print(f"\nUSING: euler '{conv}', zenith -z, precession off  (forced)")

    #direction set conditioning
    _, _, cams, _ = fit(caps, conv, zs, pr, "euler")
    V = np.array([cams[[i for i, c in enumerate(caps) if c["att"] == a][0]] for a in atts])
    sv = np.linalg.svd(V - V.mean(0), compute_uv=False)
    print("\n" + "=" * 74)
    print("conditioning of the attitude set")
    print("=" * 74)
    print(f"  singular values {sv[0]:.3f} {sv[1]:.3f} {sv[2]:.3f}   (night 2: 0.87 0.84 0.13)")

    #source comparison
    print("\n" + "=" * 74)
    print("STEP 3 - vertical source comparison")
    print("=" * 74)
    print(f"{'source':>10} {'in-sample':>10} {'held-out':>10} {'held-out km':>12} {'R_offset':>9}")
    print("-" * 56)
    fits = {}
    for source in SOURCES:
        R_s, res, _, _ = fit(caps, conv, zs, pr, source)
        rms = float(np.sqrt(np.mean(res ** 2)))
        h = held_out(caps, conv, zs, pr, source)
        fits[source] = (R_s, res)
        print(f"{source:>10} {rms:9.0f}\" {h:9.0f}\" {h/ARCSEC*111.32:11.1f} {np.degrees(R_s.magnitude()):8.3f}d")
    print("\n  euler and acc_block should show the same R_offset to ~0.1 deg;")
    print("  the residual difference between them is the flight controller's filter.")

    #per attitude residuals for the save source
    R_s, res = fits[SAVE_SOURCE]
    print("\n" + "=" * 74)
    print(f"per attitude residual, source '{SAVE_SOURCE}'")
    print("=" * 74)
    for a in atts:
        idx = [i for i, c in enumerate(caps) if c["att"] == a]
        print(f"  {a:26s} n={len(idx):2d}  mean {res[idx].mean():7.1f}\"  max {res[idx].max():7.1f}\"")

    if "--save" in sys.argv:
        np.save("R_offset.npy", R_s.as_matrix())
        np.save("R_offset_epoch.npy", np.array([np.mean([c["utc"].timestamp() for c in caps])]))
        print(f"\nSaved R_offset.npy from source '{SAVE_SOURCE}' "
              f"({np.degrees(R_s.magnitude()):.3f} deg, euler 'ZYX', zenith -z, precession off)")
    else:
        print("\nNothing written. Re-run with --save to write R_offset.npy.")


if __name__ == "__main__":
    main()