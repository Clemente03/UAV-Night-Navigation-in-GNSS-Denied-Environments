"""

offline_lib.py — shared numerics for the offline studies (no Tetra3, no images)

    radec_to_vec        catalogue RA/Dec (deg) -> unit vectors
    body_vectors        pixel centroids (y, x) -> camera unit vectors, Tetra3's radial model
    qmethod             Davenport q-method, rotation R with  b = R r
    triad               two-vector attitude, used as RANSAC's minimal solver
    vote_filter         the thesis's geometric vote (pairwise angles, keep >= max-1)
    refine              bounded post-fit rejection, median + 3 MAD, at most 5 passes
    geometric_pipeline  vote -> q-method -> refine, as stilter_filter.py does it
    ransac              two-star RANSAC with an adaptive iteration count

Lens model (Tetra3's): undistorted = distorted * (1 - k * (r_d / (W/2))^2),
f in pixels along the optical axis; k is the distortion at half width.

Pedro Clemente, 2026-10-03

"""
import numpy as np

ARCSEC = np.pi / 180 / 3600


def radec_to_vec(ra_deg, dec_deg):
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.column_stack((np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)))


def body_vectors(centroids_yx, width, height, f_px, k=0.0, cx=None, cy=None):
    c = np.asarray(centroids_yx, float)
    cx = width / 2.0 if cx is None else cx
    cy = height / 2.0 if cy is None else cy
    x = c[:, 1] - cx
    y = c[:, 0] - cy
    rd2 = (x * x + y * y) / (width / 2.0) ** 2
    s = 1.0 - k * rd2
    v = np.column_stack((x * s, y * s, np.full(len(c), f_px)))
    return v / np.linalg.norm(v, axis=1)[:, None]


def f_from_fov(fov_deg, width):
    return (width / 2.0) / np.tan(np.radians(fov_deg) / 2.0)


def qmethod(b, r, w=None):
    """Rotation R (3x3) minimising sum w |b - R r|^2  (Davenport, via the K-matrix eigenvector)."""
    b, r = np.asarray(b, float), np.asarray(r, float)
    w = np.ones(len(b)) / len(b) if w is None else np.asarray(w, float)
    B = (w[:, None, None] * b[:, :, None] * r[:, None, :]).sum(0)
    S = B + B.T
    z = np.array([B[1, 2] - B[2, 1], B[2, 0] - B[0, 2], B[0, 1] - B[1, 0]])
    tr = np.trace(B)
    K = np.zeros((4, 4))
    K[:3, :3] = S - tr * np.eye(3)
    K[:3, 3] = z
    K[3, :3] = z
    K[3, 3] = tr
    vals, vecs = np.linalg.eigh(K)
    q = vecs[:, -1]                      # (x, y, z, w)
    x, y, zq, s = q
    return np.array([
        [s*s + x*x - y*y - zq*zq, 2*(x*y + s*zq),          2*(x*zq - s*y)],
        [2*(x*y - s*zq),          s*s - x*x + y*y - zq*zq, 2*(y*zq + s*x)],
        [2*(x*zq + s*y),          2*(y*zq - s*x),          s*s - x*x - y*y + zq*zq]])


def triad(b1, b2, r1, r2):
    def frame(a, c):
        t1 = a / np.linalg.norm(a)
        t2 = np.cross(a, c); n = np.linalg.norm(t2)
        if n < 1e-12:
            return None
        t2 /= n
        return np.column_stack((t1, t2, np.cross(t1, t2)))
    Mb, Mr = frame(b1, b2), frame(r1, r2)
    if Mb is None or Mr is None:
        return None
    return Mb @ Mr.T


def residuals_arcsec(R, b, r):
    d = np.clip(np.einsum('ij,ij->i', b, (R @ r.T).T), -1, 1)
    return np.arccos(d) / ARCSEC


def vote_filter(b, r, tol_arcsec):
    n = len(b)
    if n < 3:
        return np.ones(n, bool)
    Db = np.arccos(np.clip(b @ b.T, -1, 1))
    Dr = np.arccos(np.clip(r @ r.T, -1, 1))
    A = (np.abs(Db - Dr) < tol_arcsec * ARCSEC).astype(int)
    np.fill_diagonal(A, 0)
    V = A.sum(1)
    if V.max() < 3:
        return np.zeros(n, bool)
    return V >= V.max() - 1


def refine(b, r, mask, sigma=3.0, max_iter=5, min_stars=3):
    mask = mask.copy()
    R = qmethod(b[mask], r[mask])
    for _ in range(max_iter):
        res = residuals_arcsec(R, b[mask], r[mask])
        med = np.median(res); mad = 1.4826 * np.median(np.abs(res - med))
        keep = res <= med + sigma * max(mad, 1e-9)
        if keep.all() or keep.sum() < min_stars:
            break
        idx = np.where(mask)[0]
        mask[idx[~keep]] = False
        R = qmethod(b[mask], r[mask])
    return R, mask


def geometric_pipeline(b, r, tol_arcsec):
    mask = vote_filter(b, r, tol_arcsec)
    if mask.sum() < 3:
        return None, mask
    return refine(b, r, mask)


def ransac(b, r, tol_arcsec, rng, conf=0.999, max_iter=2000, min_inliers=3):
    """Two-star RANSAC: TRIAD on a random pair, inliers by residual, adaptive iteration count,
    q-method on the best consensus set, then the same bounded refinement as the vote."""
    n = len(b)
    best = np.zeros(n, bool); best_n = 0
    need = max_iter; it = 0
    while it < min(need, max_iter):
        it += 1
        i, j = rng.choice(n, 2, replace=False)
        R = triad(b[i], b[j], r[i], r[j])
        if R is None:
            continue
        inl = residuals_arcsec(R, b, r) < tol_arcsec
        m = inl.sum()
        if m > best_n:
            best, best_n = inl, m
            eps = 1.0 - m / n                                  # outlier fraction estimate
            p_good = (1.0 - eps) ** 2
            need = 1 if p_good >= 1 else int(np.ceil(np.log(1 - conf) / np.log(1 - max(p_good, 1e-9))))
    if best_n < min_inliers:
        return None, best, it
    R, mask = refine(b, r, best)
    return R, mask, it


def rotation_angle_arcsec(R1, R2):
    c = (np.trace(R1.T @ R2) - 1) / 2
    return np.arccos(np.clip(c, -1, 1)) / ARCSEC