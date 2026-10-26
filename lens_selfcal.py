"""

lens_selfcal.py — star-field self-calibration of the lens, and the test of the night-3 explanation

    python lens_selfcal.py                 # reads matches_cache.pkl, fits focal length f and distortion k
    python lens_selfcal.py --pp            # also fits the principal point (cx, cy)

Thesis, Section 6.2: the extra frame-to-frame boresight scatter on nights 1 and 3 follows the field
of view Tetra3 refits to each frame; a fixed lens model should remove it. This script fits ONE lens
model to every solved frame at once, from pairwise star angles only (rotation-free, so no attitude
is needed), using Tetra3's radial model  undistorted = distorted * (1 - k (r / (W/2))^2).

Then, per frame, it solves the attitude with Davenport's q-method under three lens models and
compares, per night:
    per-star residual after the attitude fit (the 68" vs 24" question)
    boresight declination scatter about the linear drift, and frame to frame
for   Tetra3 (its own per-frame FOV and distortion) | pinhole (stilter's model) | fixed self-calibrated.

If the fixed model brings nights 1 and 3 towards night 2's 9", the explanation in Section 6.2 holds.

Pedro Clemente, 2026-10-03

"""
import argparse, pickle
import numpy as np
from scipy.optimize import least_squares
from offline_lib import ARCSEC, radec_to_vec, body_vectors, f_from_fov, qmethod, residuals_arcsec, geometric_pipeline


def frame_data(rec):
    W, H = rec["width"], rec["height"]
    r = radec_to_vec(rec["star_ra"], rec["star_dec"])
    b0 = body_vectors(rec["centroids_yx"], W, H, f_from_fov(rec["fov"], W))
    R, mask = geometric_pipeline(b0, r, 600.0)   # loose: only gross mismatches out; the fit itself is robust
    if R is None or mask.sum() < 5:
        return None
    c = rec["centroids_yx"][mask]; r = r[mask]
    iu = np.triu_indices(len(c), 1)
    ang_r = np.arccos(np.clip(np.einsum('ij,ij->i', r[iu[0]], r[iu[1]]), -1, 1))
    return dict(c=c, r=r, iu=iu, ang_r=ang_r, W=W, H=H, rec=rec)


def pair_residuals(params, frames, fit_pp):
    f, k = params[0], params[1]
    cx, cy = (params[2], params[3]) if fit_pp else (None, None)
    out = []
    for d in frames:
        b = body_vectors(d["c"], d["W"], d["H"], f, k,
                         None if cx is None else d["W"] / 2 + cx, None if cy is None else d["H"] / 2 + cy)
        ang_b = np.arccos(np.clip(np.einsum('ij,ij->i', b[d["iu"][0]], b[d["iu"][1]]), -1, 1))
        out.append((ang_b - d["ang_r"]) / ARCSEC)
    return np.concatenate(out)


def boresight_dec(R):
    return np.degrees(np.arcsin(np.clip(R[2, 2], -1, 1)))


def scatter(t_min, dec_deg):
    y = (dec_deg - dec_deg[0]) * 3600
    lin = y - np.polyval(np.polyfit(t_min, y, 1), t_min)
    return lin.std(ddof=2), np.std(np.diff(y), ddof=1) / np.sqrt(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="matches_cache.pkl")
    ap.add_argument("--pp", action="store_true", help="also fit the principal point")
    a = ap.parse_args()
    recs = pickle.load(open(a.cache, "rb"))
    frames = [d for d in (frame_data(r) for r in recs) if d is not None]
    W = frames[0]["W"]
    k_t3 = np.array([d["rec"]["distortion"] for d in frames if d["rec"]["distortion"] is not None], float)
    fov_t3 = np.array([d["rec"]["fov"] for d in frames])
    print(f"{len(frames)} frames, {sum(len(d['ang_r']) for d in frames)} star pairs")
    print(f"Tetra3 per frame:  FOV {fov_t3.mean():.4f} +- {fov_t3.std():.4f} deg   "
          f"distortion k {k_t3.mean():+.4f} +- {k_t3.std():.4f}" if len(k_t3) else "")

    x0 = [f_from_fov(np.median(fov_t3), W), np.median(k_t3) if len(k_t3) else 0.0] + ([0.0, 0.0] if a.pp else [])
    fit = least_squares(pair_residuals, x0, args=(frames, a.pp), loss="soft_l1", f_scale=30.0, x_scale="jac")
    f_px, k = fit.x[0], fit.x[1]
    res = pair_residuals(fit.x, frames, a.pp)
    res0 = pair_residuals([x0[0], 0.0] + ([0.0, 0.0] if a.pp else []), frames, a.pp)
    fov_eq = 2 * np.degrees(np.arctan((W / 2) * (1 - k) / f_px))
    print(f"\nSELF-CALIBRATED LENS:  f = {f_px:.1f} px   k = {k:+.4f}   (edge FOV {fov_eq:.4f} deg)"
          + (f"   principal point offset ({fit.x[2]:+.1f}, {fit.x[3]:+.1f}) px" if a.pp else ""))
    print(f"  pairwise-angle residual RMS: pinhole {np.sqrt(np.mean(res0**2)):.1f}\"  ->  fixed model {np.sqrt(np.mean(res**2)):.1f}\"")
    import json
    json.dump(dict(f_px=float(f_px), k=float(k), cx_off=float(fit.x[2]) if a.pp else 0.0,
                   cy_off=float(fit.x[3]) if a.pp else 0.0, edge_fov_deg=float(fov_eq)), open("lens_model.json", "w"), indent=1)
    print("  saved lens_model.json (ransac_vs_vote.py --lens lens_model.json uses it)")

    # per frame attitude under three lens models
    models = {
        "Tetra3 per frame": lambda d: body_vectors(d["c"], d["W"], d["H"],
                                                   f_from_fov(d["rec"]["fov"], d["W"]) * (1 - (d["rec"]["distortion"] or 0.0)),
                                                   d["rec"]["distortion"] or 0.0),
        "pinhole (stilter)": lambda d: body_vectors(d["c"], d["W"], d["H"], f_from_fov(d["rec"]["fov"], d["W"])),
        "fixed self-cal": lambda d: body_vectors(d["c"], d["W"], d["H"], f_px, k,
                                                 None if not a.pp else d["W"] / 2 + fit.x[2],
                                                 None if not a.pp else d["H"] / 2 + fit.x[3]),
    }
    print("\nPER-STAR RESIDUAL AFTER THE ATTITUDE FIT (RMS, arcsec), by night")
    print(f"  {'model':20s} " + "  ".join(f"night {n}" for n in (1, 2, 3)))
    sols = {}
    for name, bfun in models.items():
        line = []
        for n in (1, 2, 3):
            rr = []
            for d in frames:
                if d["rec"]["night"] != n:
                    continue
                b = bfun(d); R = qmethod(b, d["r"]); rr.append(residuals_arcsec(R, b, d["r"]))
                sols[(name, id(d))] = R
            line.append(f"{np.sqrt(np.mean(np.concatenate(rr)**2)):7.1f}" if rr else "    n/a")
        print(f"  {name:20s} " + "  ".join(line))

    print("\nBORESIGHT DECLINATION SCATTER, STATIC BLOCK (arcsec): about the drift / frame to frame")
    print(f"  {'model':20s} " + "  ".join(f"{'night ' + str(n):>15s}" for n in (1, 2, 3)))
    rows = {"Tetra3's own Dec": None, **{m: None for m in models}}
    for name in rows:
        line = []
        for n in (1, 2, 3):
            st = sorted([d for d in frames if d["rec"]["night"] == n and d["rec"]["block"] == "static_B"],
                        key=lambda d: d["rec"]["board_timestamp"])
            if len(st) < 5:
                line.append(f"{'n/a':>15s}"); continue
            t = np.array([(d["rec"]["board_timestamp"] - st[0]["rec"]["board_timestamp"]) / 60 for d in st])
            if name == "Tetra3's own Dec":
                dec = np.array([d["rec"]["dec"] for d in st])
            else:
                dec = np.array([boresight_dec(sols[(name, id(d))]) for d in st])
            s1, s2 = scatter(t, dec)
            line.append(f"{s1:7.1f} / {s2:5.1f}")
        print(f"  {name:20s} " + "  ".join(line))
    print("\nNight 2's 9\" is the optical chain's predicted precision. If the fixed model brings nights 1 and 3")
    print("close to it, the per-frame refit was the cause, as Section 6.2 states; if not, revise that paragraph.")


if __name__ == "__main__":
    main()