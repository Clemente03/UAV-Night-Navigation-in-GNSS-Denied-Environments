"""

matched_centroids.py — matched filter star detection

Drop in replacement for tetra3.get_centroids_from_image. Returns centroids in
tetra3's (y, x) convention so solve_from_centroids takes them unchanged

Usage:

    from matched_centroids import get_centroids_matched
    centroids = get_centroids_matched(img, sigma=CENTROID_SIGMA, fwhm_px=FWHM)

FWHM: measure it on a bright star in one of your frames. 1.5-2.5 px is typical
for this plate scale

Pedro Clemente, 2026-09-27

"""

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, label, center_of_mass


def get_centroids_matched(img, sigma=2.0, fwhm_px=2.0, max_returned=60,
                          min_sep_px=4, edge_px=6, sat_level=255,
                          noise_floor=0.15, min_peak=1.5):
    """
    img           2D array, any dtype (uint8 works; converted to float)
    sigma         detection threshold in units of the matched filtered noise
    fwhm_px       star FWHM in pixels; the kernel matches this
    max_returned  keep the brightest N, like tetra3
    min_sep_px    merge detections closer than this
    edge_px       ignore detections this close to the border
    sat_level     flag and drop stars whose raw peak is at saturation
    noise_floor   lower bound on the noise estimate, in ADU of the filtered
                  image. On quantised unity gain frames the MAD collapses to
                  ~0 and sigma stops meaning anything; 0.15 ADU is the read
                  noise after a 2-px Gaussian at gain 32
    min_peak      minimum background subtracted peak height in the RAW image,
                  in ADU. Kills single level quantisation bumps. 1.5 = must be
                  more than one 8 bit step above background

    returns (N, 2) array of (y, x) centroids, brightest first
    """
    a = np.asarray(img, dtype=np.float32)
    h, w = a.shape

    #bg: a wide median(ish) estimate via a large Gaussian; subtract so the matched filter sees zero mean noise plus stars
    bg = gaussian_filter(a, sigma=25.0)
    d = a - bg

    #matched filter: Gaussian with the star's width
    k_sigma = fwhm_px / 2.3548
    m = gaussian_filter(d, sigma=k_sigma)

    #noise of the filtered image, robustly, from the MAD
    med = np.median(m)
    mad = max(np.median(np.abs(m - med)) * 1.4826, noise_floor)
    thr = med + sigma * mad

    #candidates: local maxima above threshold
    peaks = (m == maximum_filter(m, size=2 * min_sep_px + 1)) & (m > thr)
    peaks[:edge_px, :] = peaks[-edge_px:, :] = False
    peaks[:, :edge_px] = peaks[:, -edge_px:] = False
    ys, xs = np.nonzero(peaks)
    if len(ys) == 0:
        return np.zeros((0, 2), dtype=np.float64)

    #refine each with a weighted centroid on the background subtracted image
    r = max(2, int(round(1.5 * fwhm_px)))
    out = []
    for y0, x0 in zip(ys, xs):
        y1, y2 = max(0, y0 - r), min(h, y0 + r + 1)
        x1, x2 = max(0, x0 - r), min(w, x0 + r + 1)
        win = d[y1:y2, x1:x2]
        raw = a[y1:y2, x1:x2]
        if raw.max() >= sat_level:
            continue                       #clipped star: biased centroid, skip
        if d[y0, x0] < min_peak:
            continue                       #quantisation bump, not a star
        win = np.clip(win, 0, None)
        s = win.sum()
        if s <= 0:
            continue
        yy, xx = np.mgrid[y1:y2, x1:x2]
        cy = (win * yy).sum() / s
        cx = (win * xx).sum() / s
        out.append((float(m[y0, x0]), cy, cx))

    if not out:
        return np.zeros((0, 2), dtype=np.float64)
    out.sort(reverse=True)
    out = out[:max_returned]
    return np.array([[cy, cx] for _, cy, cx in out], dtype=np.float64)


def measure_fwhm(img, n_stars=5):
    #rough FWHM from the brightest unsaturated stars
    a = np.asarray(img, dtype=np.float32)
    bg = gaussian_filter(a, 25.0)
    d = a - bg
    peaks = (d == maximum_filter(d, 9)) & (d > np.percentile(d, 99.9))
    ys, xs = np.nonzero(peaks)
    order = np.argsort(-d[ys, xs])
    fw = []
    for i in order[:n_stars * 3]:
        y, x = ys[i], xs[i]
        if y < 6 or x < 6 or y > a.shape[0] - 7 or x > a.shape[1] - 7:
            continue
        if a[y, x] >= 250:
            continue
        win = d[y - 5:y + 6, x - 5:x + 6]
        pk = win.max()
        if pk <= 0:
            continue
        #count pixels above half max along the two axes through the peak
        row = win[5, :]; col = win[:, 5]
        fw.append(0.5 * ((row > pk / 2).sum() + (col > pk / 2).sum()))
        if len(fw) >= n_stars:
            break
    return float(np.median(fw)) if fw else float("nan")


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    h, w, fwhm = 960, 1280, 2.0
    img = rng.normal(12.0, 2.6, (h, w))
    truth = []
    for _ in range(40):
        y, x = rng.uniform(20, h - 20), rng.uniform(20, w - 20)
        amp = rng.uniform(4, 30)            #peak counts above background
        yy, xx = np.mgrid[0:h, 0:w]
        img += amp * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * (fwhm / 2.3548) ** 2))
        truth.append((y, x, amp))
    img8 = np.clip(img, 0, 255).astype(np.uint8)

    c = get_centroids_matched(img8, sigma=2.0, fwhm_px=fwhm)
    d = img8.astype(float) - np.median(img8)
    thr = 2.0 * np.median(np.abs(d)) * 1.4826
    naive = ((d == maximum_filter(d, 5)) & (d > thr)).sum()

    def recovered(cents, tol=1.5):
        hit = 0
        for y, x, _ in truth:
            if len(cents) and np.min(np.hypot(cents[:, 0] - y, cents[:, 1] - x)) < tol:
                hit += 1
        return hit

    print(f"matched filter: {len(c)} detections, {recovered(c)} true stars recovered")
    print(f"per pixel threshold at the same sigma: {naive} candidates (mostly noise)")
    print(f"measured FWHM: {measure_fwhm(img8):.2f} px  (planted {fwhm})")