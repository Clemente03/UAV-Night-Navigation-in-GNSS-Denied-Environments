"""

collect_matches.py — solve every frame once and cache what the offline studies need

    python collect_matches.py                  # nights 1 2 3, every block
    python collect_matches.py --nights 2 3     # a subset

Runs the same preparation and detection as batch_night.py (matched dark + hot-pixel repair +
matched filter, or raw + Tetra3's centroider where no master exists), solves with Tetra3 and
keeps, per solved frame: night, block, file, time, gain, exposure, image size, Tetra3's RA, Dec,
roll, FOV, distortion and RMSE, the matched centroids (y, x) and the matched catalogue RA/Dec.
Writes matches_cache.pkl in the working directory; ransac_vs_vote.py and lens_selfcal.py read it.

Pedro Clemente, 2026-10-03

"""
import argparse, csv, glob, os, pickle, time
from datetime import datetime, timezone
import numpy as np
import tetra3
import batch_night as bn
from matched_centroids import get_centroids_matched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nights", type=int, nargs="+", default=[1, 2, 3])
    a = ap.parse_args()
    t3 = tetra3.Tetra3(load_database=bn.DATABASE)
    out = []
    t0 = time.time()
    for night in a.nights:
        N = bn.NIGHTS[night]; root, mdir = N["root"], f"{N['root']}/{N['masters']}"
        bn._cache.clear()
        for name, csv_path, img_dir in bn._blocks(root, N["layout"]):
            if not os.path.exists(csv_path):
                continue
            with open(csv_path) as f:
                rows = {r["filename"].strip(): r for r in csv.DictReader(f)}
            ok = n = 0
            for fp in sorted(glob.glob(os.path.join(img_dir, "*.tif*"))):
                row = rows.get(os.path.basename(fp))
                if row is None:
                    continue
                n += 1
                gain, exp = int(float(row["gain_reg"])), int(float(row["exposure_reg"]))
                img, note = bn.prepare(fp, mdir, gain, exp, True)
                if "dark" in note:
                    cents = get_centroids_matched(img, fwhm_px=bn.FWHM_PX, **bn.FILTER.get(gain, bn.FILTER[32]))
                else:
                    cents = tetra3.get_centroids_from_image(img, sigma=2.0)
                if len(cents) < 3:
                    continue
                sol = t3.solve_from_centroids(cents, size=img.shape, fov_estimate=bn.FOV_ESTIMATE,
                                              fov_max_error=bn.FOV_MAX_ERROR, return_matches=True)
                if sol.get("RA") is None:
                    continue
                ms = np.asarray(sol["matched_stars"], float)
                out.append(dict(
                    night=night, block=name, filename=os.path.basename(fp),
                    board_timestamp=float(row["board_timestamp"]),
                    utc=datetime.fromtimestamp(float(row["board_timestamp"]) + bn.CLOCK_OFFSET_S, tz=timezone.utc),
                    gain=gain, exposure=exp, processing=note, height=img.shape[0], width=img.shape[1],
                    ra=sol["RA"], dec=sol["Dec"], roll=sol["Roll"], fov=sol["FOV"],
                    distortion=sol.get("distortion"), rmse=sol.get("RMSE"),
                    centroids_yx=np.asarray(sol["matched_centroids"], float),
                    star_ra=ms[:, 0], star_dec=ms[:, 1],
                    imu=(float(row["imu_yaw"]), float(row["imu_pitch"]), float(row["imu_roll"]))))
                ok += 1
            print(f"night {night}  {name:14s} {ok:3d}/{n:3d} solved")
    with open("matches_cache.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"\n{len(out)} solved frames cached in matches_cache.pkl ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()