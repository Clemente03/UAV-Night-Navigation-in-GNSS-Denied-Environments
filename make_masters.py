"""

make_masters.py — build master darks and hot pixel masks for one night

    python make_masters.py --night 3

writes <root>/masters/master_g<gain>_e<exp>_dark.npy and _hot.npy for every
(gain, exposure, folder) listed for that night. The hot threshold adapts to the
gain: the unity gain master is flat at 12, so anything consistently above it is
warm and HOT_SIGMA is small; at gain 245 the noise spans several levels.

Pedro Clemente, 2026-09-27

"""
import argparse
import glob
import os
import sys
import numpy as np

try:
    import tifffile as tiff
except ImportError:
    sys.exit("pip install tifffile")

IMAGES = "/Users/pedroclemente/Documents/Tese/Images"
NIGHTS = {
    1: (f"{IMAGES}/18_09_2026", "Darks/masters",
        [(245, e, f"Darks/gain245_exp{e}") for e in (45, 91, 182, 255, 455, 900)]),
    2: (f"{IMAGES}/19_09_2026", "masters",
        [(32, 4545, "night2_darks_gain32/exp4545"), (32, 900, "night2_darks_gain32/exp900")]),
    3: (f"{IMAGES}/25_09_2026", "masters",
        [(32, 4545, "Darks/darks_gain32_exp4545"), (32, 900, "Darks/darks_gain32_exp900")]
        + [(245, e, f"Darks/darks_gain245_exp{e}") for e in (900, 1100, 1500, 2273, 3000, 4545)]),
    4: (f"{IMAGES}/26_09_2026", "masters",
        [(32, 4545, "Darks/darks_gain32_exp4545"), (32, 900, "Darks/darks_gain32_exp900")]
        + [(245, e, f"Darks/darks_gain245_exp{e}") for e in (255, 455)]),
}
HOT_SIGMA = {32: 2.0, 245: 5.0}
MAD_FLOOR = 1.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--night", type=int, required=True, choices=sorted(NIGHTS))
    a = ap.parse_args()
    root, out_rel, sets = NIGHTS[a.night]
    out_dir = f"{root}/{out_rel}"
    os.makedirs(out_dir, exist_ok=True)

    for gain, exp, rel in sets:
        files = sorted(glob.glob(f"{root}/{rel}/*.tif*"))
        if len(files) < 3:
            print(f"g{gain} e{exp}: {len(files)} frames in {rel} - skipped"); continue
        stack = np.stack([tiff.imread(f).astype(np.float32) for f in files])
        master = np.median(stack, axis=0)
        med = float(np.median(master))
        mad = max(float(np.median(np.abs(master - med)) * 1.4826), MAD_FLOOR)
        hot = master > med + HOT_SIGMA.get(gain, 5.0) * mad
        base = f"{out_dir}/master_g{gain}_e{exp}"
        np.save(base + "_dark.npy", master); np.save(base + "_hot.npy", hot)
        temporal = float(np.std(stack[1] - stack[0]) / np.sqrt(2))
        print(f"g{gain:3d} e{exp:5d}: {len(files):2d} frames  median {med:5.1f}  spatial std {master.std():5.2f}  "
              f"temporal {temporal:4.2f}  hot {int(hot.sum()):6d} ({100*hot.mean():.1f}%)")


if __name__ == "__main__":
    main()