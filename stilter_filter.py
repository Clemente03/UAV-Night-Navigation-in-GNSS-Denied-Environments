"""

stilter_filter.py — geometric outlier rejection and Davenport q-method attitude

    from stilter_filter import stilter
    q = stilter(tolerance_arcsec=200.0).process(sol=sol, utc_dt=utc,
            img_width=1280, img_height=960, estimator_cls=estimator)

process() takes a Tetra3 solution (solved with return_matches=True) and returns the camera
attitude and the stars it trusts:
    reference  - matched catalogue stars, ICRS -> CIRS at the frame epoch
                 (TT = UTC + 69.184 s, matrix from estimator_cls)
    body       - matched centroids as unit vectors through a pinhole model; focal length
                 from the FOV Tetra3 solved (refuses to run without one); no distortion term
    vote       - pairwise angles in image and catalogue; a pair agrees when they differ by
                 less than tolerance_arcsec; stars within one vote of the best are kept;
                 best star under 3 votes -> frame rejected
    attitude   - Davenport q-method: eigenvector of K with the largest eigenvalue,
                 quaternion (x, y, z, w), CIRS -> camera, w >= 0
    refine     - residuals beyond median + 3 MAD dropped and the attitude re-solved,
                 at most 5 times, never below 3 stars
returns a dict: success, quaternion, inliers_kept, inlier_mask, residuals_arcsec,
rms_arcsec, max_residual_arcsec, focal_length_px, fov_used_deg, distortion (residual
against radius from the image centre); on failure, success False and a reason.

options: iterative_rejection=True, rejection_sigma=3.0, max_iterations=5,
use_magnitude_weights=False, min_stars=3

tolerance_arcsec: 100" for synthetic frames, 200" for real ones. With no distortion term
the per-star residual here (~68" on real frames) is larger than Tetra3's own (~24")

Pedro Clemente, 2026-10-03

"""

import numpy as np
from scipy.spatial.transform import Rotation as R


class stilter:
    def __init__(self, tolerance_arcsec = 100.0,
                 iterative_rejection=True,
                 rejection_sigma=3.0,
                 max_iterations=5,
                 use_magnitude_weights=False,
                 min_stars=3):
        self.tolerance_rad = np.radians(tolerance_arcsec / 3600.0)
        self.iterative_rejection = iterative_rejection
        self.rejection_sigma = rejection_sigma
        self.max_iterations = max_iterations
        self.use_magnitude_weights = use_magnitude_weights
        self.min_stars = min_stars

    def filter_outliers_geometric(self, vectors_b, vector_ref):
        N = vectors_b.shape[0]
        if N < 3:
            return np.ones(N, dtype=bool)

        dot_meas = vectors_b @ vectors_b.T
        dot_ref = vector_ref @ vector_ref.T

        dot_meas = np.clip(dot_meas, -1.0, 1.0)
        dot_ref = np.clip(dot_ref, -1.0, 1.0)

        ang_meas = np.arccos(dot_meas)
        ang_ref = np.arccos(dot_ref)

        diff_matrix = np.abs(ang_meas - ang_ref)
        agreement_matrix = diff_matrix < self.tolerance_rad

        np.fill_diagonal(agreement_matrix, False)
        votes = np.sum(agreement_matrix, axis=1)

        max_votes = np.max(votes)
        inlier_mask = votes >= (max_votes - 1)

        if max_votes < 3:
            return np.zeros(N, dtype=bool)
        return inlier_mask

    def solve_davenport_q(self, vectors_b, vectors_ref, weights=None):
        N = vectors_b.shape[0]
        w = (
            np.ones(N) / N
            if weights is None
            else np.asarray(weights) / np.sum(weights)
        )

        B = (vectors_b * w[:, None]).T @ vectors_ref
        sigma = np.trace(B)
        S = B + B.T
        z = np.sum(
            np.cross(vectors_ref, vectors_b) * w[:, None], axis=0)

        K = np.block([
            [S - sigma * np.eye(3), z[:, None]],
            [z[None, :], np.array([[sigma]])]
        ])

        eigenvalues, eigenvectors = np.linalg.eigh(K)
        opt_q = eigenvectors[:, -1]

        if opt_q[3] < 0:
            opt_q = -opt_q

        return opt_q

    @staticmethod
    def quaternion_to_matrix(q):
        qx, qy, qz, qw = q
        return np.array([
            [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
            [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
            [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]])
 
    def residuals_arcsec(self, q, body_vectors, ref_vectors):
        #angle between each measured star direction and where the solved attitude says it should be
        M = self.quaternion_to_matrix(q)          #CIRS -> camera
        predicted = ref_vectors @ M.T
        dots = np.clip(np.einsum("ij,ij->i", predicted, body_vectors), -1.0, 1.0)
        return np.degrees(np.arccos(dots)) * 3600.0
 
    def _solve_with_rejection(self, body, ref, weights):
        #solve, measure, drop the worst offenders, repeat
        keep = np.ones(len(body), dtype=bool)
        history = []
        q = self.solve_davenport_q(body, ref, weights)
 
        if not self.iterative_rejection:
            return q, keep, self.residuals_arcsec(q, body, ref), history
 
        for _ in range(self.max_iterations):
            res = self.residuals_arcsec(q, body[keep], ref[keep])
            if len(res) <= self.min_stars:
                break
            med = np.median(res)
            mad = np.median(np.abs(res - med)) * 1.4826
            if mad <= 0:
                break
            cut = med + self.rejection_sigma * mad
            bad = res > cut
            if not bad.any():
                break
            idx = np.where(keep)[0][bad]
            if keep.sum() - len(idx) < self.min_stars:
                break
            keep[idx] = False
            history.append({"dropped": int(len(idx)), "cut_arcsec": float(cut)})
            w = None if weights is None else weights[keep]
            q = self.solve_davenport_q(body[keep], ref[keep], w)
 
        return q, keep, self.residuals_arcsec(q, body[keep], ref[keep]), history
 
    @staticmethod
    def distortion_diagnostic(centroids, residuals, img_width, img_height):
        """
 
        A pinhole model with no radial distortion term is assumed everywhere in
        this pipeline. If a real lens has barrel distortion, the residual will
        climb toward the corners, and that is a correctable systematic rather
        than noise. Flat means the model is fine

        """
        if len(residuals) < 4:
            return {"n": len(residuals), "correlation": None}
        r = np.hypot(centroids[:, 1] - img_width / 2.0,
                     centroids[:, 0] - img_height / 2.0)
        if np.std(r) == 0 or np.std(residuals) == 0:
            return {"n": len(residuals), "correlation": None}
        c = float(np.corrcoef(r, residuals)[0, 1])
        slope = float(np.polyfit(r, residuals, 1)[0])
        return {
            "n": int(len(residuals)),
            "correlation": c,
            "slope_arcsec_per_px": slope,
            "max_radius_px": float(r.max()),
            "extrapolated_corner_arcsec": float(slope * np.hypot(
                img_width / 2.0, img_height / 2.0)),
            "verdict": ("likely radial distortion" if c > 0.5
                        else "no radial trend" if c is not None else None),
        }

    def process(self, sol, utc_dt, img_width, img_height, estimator_cls):
        if "matched_stars" not in sol or len(sol["matched_stars"]) < self.min_stars:
            return {"success": False, "reason": "Not enough stars matched by Tetra3"}
 
        stars = np.asarray(sol["matched_stars"])
        ras_icrs, decs_icrs = np.radians(stars[:, 0]), np.radians(stars[:, 1])
 
        v_icrs = np.column_stack((
            np.cos(decs_icrs) * np.cos(ras_icrs),
            np.cos(decs_icrs) * np.sin(ras_icrs),
            np.sin(decs_icrs)
        ))
 
        jd = estimator_cls.jd_from_datetime(utc_dt)
        M_icrs_to_cirs = estimator_cls.get_icrs_to_cirs_matrix(jd + (69.184 / 86400.0))
        ref_vectors = v_icrs @ M_icrs_to_cirs.T
 
        centroids = np.asarray(sol["matched_centroids"])
 
        if "FOV" not in sol or sol["FOV"] is None:
            return {"success": False, "reason": "Tetra3 returned no FOV; refusing to assume one"}
        calibrated_fov = sol["FOV"]
 
        #tetra3 reports horizontal FOV, so the focal length comes from the
        #image width. max(w,h) happens to equal w for 1280x960
        f = (img_width / 2.0) / np.tan(np.radians(calibrated_fov) / 2.0)
 
        v_cam = np.column_stack((
            centroids[:, 1] - (img_width / 2.0),
            centroids[:, 0] - (img_height / 2.0),
            np.full_like(centroids[:, 0], f)
        ))
        body_vectors = v_cam / np.linalg.norm(v_cam, axis=1)[:, None]
 
        #magnitude weighting: tetra3's matched_stars is (RA, Dec, magnitude)
        weights = None
        if self.use_magnitude_weights and stars.shape[1] >= 3:
            mags = stars[:, 2].astype(float)
            if np.all(np.isfinite(mags)):
                weights = 10.0 ** (-0.4 * mags)      # relative flux
                weights = weights / weights.max()
 
        geo_mask = self.filter_outliers_geometric(body_vectors, ref_vectors)
        if np.sum(geo_mask) < self.min_stars:
            return {"success": False, "reason": f"Insufficient inliers ({int(np.sum(geo_mask))})"}
 
        w_geo = None if weights is None else weights[geo_mask]
        q_opt, keep_rel, res_final, history = self._solve_with_rejection(
            body_vectors[geo_mask], ref_vectors[geo_mask], w_geo)
 
        #map the post fit rejections back onto the full match list
        inlier_mask = np.zeros(len(body_vectors), dtype=bool)
        inlier_mask[np.where(geo_mask)[0][keep_rel]] = True
        n_inliers = int(inlier_mask.sum())
 
        if n_inliers < self.min_stars:
            return {"success": False, "reason": f"Insufficient inliers after rejection ({n_inliers})"}
 
        rms = float(np.sqrt(np.mean(res_final ** 2)))
 
        return {
            "success": True,
            "quaternion": q_opt,
            "inliers_kept": n_inliers,
            "total_matches": len(inlier_mask),
            "inlier_mask": inlier_mask,
            "matched_centroids": centroids,
            "residuals_arcsec": res_final,
            "rms_arcsec": rms,
            "max_residual_arcsec": float(res_final.max()),
            "rejection_history": history,
            "n_rejected_postfit": int(np.sum(geo_mask) - n_inliers),
            "focal_length_px": float(f),
            "fov_used_deg": float(calibrated_fov),
            "distortion": self.distortion_diagnostic(
                centroids[inlier_mask], res_final, img_width, img_height),
        }