"""

ransac_vs_vote.py — the thesis's geometric vote against RANSAC, on the real campaign frames

    python ransac_vs_vote.py                     # reads matches_cache.pkl
    python ransac_vs_vote.py --seeds 50 --trials 20

Three methods, same matched stars, same 200" tolerance, same bounded refinement afterwards:
    vote (max-1)     the thesis rule: keep stars within one vote of the best
    vote (half-max)  keep stars with at least half the best vote count (and >= 3)
    RANSAC           two-star TRIAD hypotheses, adaptive iteration count (99.9 %)

Part A, real frames as matched by Tetra3: do the methods agree (inlier sets, attitude), and what
does each cost? RANSAC is run with many seeds per frame to expose its run-to-run variation.
Part B, contamination: a fraction p of each frame's catalogue matches is replaced by a wrong
identification (the catalogue direction displaced 0.3-3 deg). Success = no wrong match kept and the
attitude within 60" of the reference, which is the attitude from all of the frame's clean (Tetra3-
verified) matches with the same refinement. Reports success rate and runtime per method.

Body vectors use the pinhole model with Tetra3's FOV, exactly as stilter_filter.py does; with
--lens lens_model.json they use the fixed lens fitted by lens_selfcal.py instead.

Pedro Clemente, 2026-10-03

"""
import argparse, pickle, time
import numpy as np
from offline_lib import (ARCSEC, radec_to_vec, body_vectors, f_from_fov, refine, geometric_pipeline,
                         ransac, rotation_angle_arcsec)

TOL = 200.0


def vote_halfmax(b, r, tol):
    n = len(b)
    Db = np.arccos(np.clip(b @ b.T, -1, 1)); Dr = np.arccos(np.clip(r @ r.T, -1, 1))
    A = (np.abs(Db - Dr) < tol * ARCSEC).astype(int); np.fill_diagonal(A, 0)
    V = A.sum(1)
    if V.max() < 3:
        return None, np.zeros(n, bool)
    m = V >= max(3, 0.5 * V.max())
    return refine(b, r, m) if m.sum() >= 3 else (None, m)


def timed(fn, reps=1):
    best = None; out = None
    for _ in range(reps):
        t = time.perf_counter(); out = fn(); dt = (time.perf_counter() - t) * 1e3
        best = dt if best is None else min(best, dt)
    return out, best


LENS = None   #set from --lens: the fixed model fitted by lens_selfcal.py


def vectors(rec):
    if LENS is not None:
        W, H = rec["width"], rec["height"]
        b = body_vectors(rec["centroids_yx"], W, H, LENS["f_px"], LENS["k"], W / 2 + LENS["cx_off"], H / 2 + LENS["cy_off"])
    else:
        b = body_vectors(rec["centroids_yx"], rec["width"], rec["height"], f_from_fov(rec["fov"], rec["width"]))
    r = radec_to_vec(rec["star_ra"], rec["star_dec"])
    return b, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="matches_cache.pkl")
    ap.add_argument("--seeds", type=int, default=30, help="RANSAC repeats per real frame")
    ap.add_argument("--trials", type=int, default=10, help="contamination draws per frame and level")
    ap.add_argument("--lens", default=None, help="lens_model.json from lens_selfcal.py (default: pinhole, as stilter)")
    a = ap.parse_args()
    global LENS
    if a.lens:
        import json
        LENS = json.load(open(a.lens))
    print("lens model:", "fixed self-calibrated" if LENS else "pinhole with Tetra3's FOV (as stilter_filter.py)")
    recs = [r for r in pickle.load(open(a.cache, "rb")) if len(r["star_ra"]) >= 6]
    rng = np.random.default_rng(2026)
    print(f"{len(recs)} frames with at least 6 matched stars\n")

    #warm-up so the first timing is not numpy's start-up
    b, r = vectors(recs[0]); geometric_pipeline(b, r, TOL); ransac(b, r, TOL, rng)

    #real frames
    tv, th, tr, it_r = [], [], [], []
    same_mask = same_half = 0; dR = []; ransac_varies = 0; nA = 0
    for rec in recs:
        b, r = vectors(rec)
        (Rv, mv), t1 = timed(lambda: geometric_pipeline(b, r, TOL), reps=5)
        (Rh, mh), t2 = timed(lambda: vote_halfmax(b, r, TOL), reps=5)
        if Rv is None:
            continue
        nA += 1; tv.append(t1); th.append(t2)
        masks = set(); ts = []
        for s in range(a.seeds):
            g = np.random.default_rng(1000 + s)
            t = time.perf_counter(); Rr, mr, it = ransac(b, r, TOL, g); ts.append((time.perf_counter() - t) * 1e3)
            it_r.append(it); masks.add(mr.tobytes())
            if s == 0 and Rr is not None:
                dR.append(rotation_angle_arcsec(Rv, Rr)); same_mask += np.array_equal(mv, mr)
        tr.extend(ts); ransac_varies += len(masks) > 1
        same_half += Rh is not None and np.array_equal(mv, mh)
    tv, th, tr = map(np.asarray, (tv, th, tr))
    print("PART A - real frames, matches as Tetra3 returned them")
    print(f"  frames                                  {nA}")
    print(f"  RANSAC keeps the same stars as the vote {same_mask}/{nA}")
    print(f"  half-max keeps the same stars as max-1  {same_half}/{nA}")
    print(f"  attitude difference vote vs RANSAC      median {np.median(dR):.2f}\"  max {np.max(dR):.2f}\"")
    print(f"  RANSAC inlier set changes between seeds  {ransac_varies}/{nA} frames")
    print(f"  runtime per frame (ms)    mean   sd     max")
    for name, t in (("vote (max-1)", tv), ("vote (half-max)", th), ("RANSAC", tr)):
        print(f"    {name:18s}     {t.mean():5.2f}  {t.std():5.2f}  {t.max():6.2f}")
    print(f"  RANSAC iterations          mean {np.mean(it_r):.1f}  max {np.max(it_r)}\n")

    #contamination
    print("PART B - wrong identifications injected into the real matches")
    print(f"  {'outliers':>8} | {'vote max-1':>16} | {'vote half-max':>16} | {'RANSAC':>24}")
    print(f"  {'':>8} | {'success  ms':>16} | {'success  ms':>16} | {'success  ms (max)':>24}")
    rows = []
    for p in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        ok = dict(v=0, h=0, r=0); tt = dict(v=[], h=[], r=[]); tot = 0
        for rec in recs:
            b, r0 = vectors(rec)
            #common reference: the attitude from all of Tetra3's (verified) matches, same refinement
            Rc, mc = refine(b, r0, np.ones(len(b), bool))
            if mc.sum() < 5:
                continue
            for _ in range(a.trials):
                r = r0.copy(); n = len(b)
                bad = rng.choice(n, int(round(p * n)), replace=False)
                for i in bad:
                    u = rng.normal(size=3); u -= u.dot(r[i]) * r[i]; u /= np.linalg.norm(u)
                    ang = np.radians(rng.uniform(0.3, 3.0))
                    r[i] = np.cos(ang) * r[i] + np.sin(ang) * u
                tot += 1
                for key, fn in (("v", lambda: geometric_pipeline(b, r, TOL)),
                                ("h", lambda: vote_halfmax(b, r, TOL)),
                                ("r", lambda: ransac(b, r, TOL, rng)[:2])):
                    t = time.perf_counter(); R, m = fn(); tt[key].append((time.perf_counter() - t) * 1e3)
                    if R is not None and rotation_angle_arcsec(R, Rc) < 60 and not m[bad].any():
                        ok[key] += 1
        s = {k: ok[k] / tot for k in ok}
        print(f"  {p:8.0%} | {s['v']:7.1%} {np.mean(tt['v']):6.2f} | {s['h']:7.1%} {np.mean(tt['h']):6.2f} | "
              f"{s['r']:7.1%} {np.mean(tt['r']):6.2f} ({np.max(tt['r']):5.1f})")
        rows.append((p, s, tt))

if __name__ == "__main__":
    main()