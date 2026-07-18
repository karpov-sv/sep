#!/usr/bin/env python3
"""Standalone test for SEP winpos stability in a crowded synthetic field."""

from __future__ import annotations

import argparse
import sys

import numpy as np
from math import erf

try:
    import sep_x as sep
except ImportError as exc:
    raise SystemExit("sep is required; build the extension first.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SEP winpos stability on a crowded synthetic field."
    )
    parser.add_argument("--size", type=int, default=256, help="Image size (square).")
    parser.add_argument(
        "--n-stars", type=int, default=800, help="Number of stars (crowded)."
    )
    parser.add_argument(
        "--n-eval",
        type=int,
        default=300,
        help="Number of stars to evaluate.",
    )
    parser.add_argument("--fwhm", type=float, default=2.5, help="PSF FWHM (pixels).")
    parser.add_argument("--flux-min", type=float, default=500.0, help="Min flux.")
    parser.add_argument("--flux-max", type=float, default=5000.0, help="Max flux.")
    parser.add_argument(
        "--background", type=float, default=1000.0, help="Sky background level."
    )
    parser.add_argument("--noise", type=float, default=5.0, help="Noise sigma.")
    parser.add_argument(
        "--pos-noise",
        type=float,
        default=0.5,
        help="Sigma of initial position noise (pixels).",
    )
    parser.add_argument(
        "--sig",
        type=float,
        default=None,
        help="Gaussian sigma for winpos weighting (default: fwhm/2.355).",
    )
    parser.add_argument(
        "--crowd-thresh",
        type=float,
        default=2.0,
        help="Crowding threshold in units of FWHM for nearest neighbor.",
    )
    parser.add_argument(
        "--use-segmap",
        action="store_true",
        help="Enable segmentation masking for winpos.",
    )
    parser.add_argument(
        "--seg-thresh-sigma",
        type=float,
        default=3.0,
        help="Threshold (sigma) for segmentation when --use-segmap is set.",
    )
    parser.add_argument(
        "--deblend-fwhm",
        type=float,
        default=0.0,
        help="Fixed PSF FWHM for deblending in segmentation (0 disables).",
    )
    parser.add_argument(
        "--deblend-method",
        type=str,
        default="threshold",
        choices=("threshold", "watershed"),
        help="Deblend method for segmentation.",
    )
    parser.add_argument(
        "--bins",
        type=str,
        default="1,2,3,4",
        help="Comma-separated FWHM bin edges for neighbor distance reporting.",
    )
    parser.add_argument("--seed", type=int, default=12345, help="RNG seed.")
    return parser.parse_args()


def _erf_vec(values: np.ndarray) -> np.ndarray:
    if hasattr(np, "erf"):
        return np.erf(values)
    return np.vectorize(erf)(values)


def add_gaussian_stamp(img: np.ndarray, x: float, y: float, flux: float, sigma: float) -> None:
    ny, nx = img.shape
    radius = int(np.ceil(4.0 * sigma))
    x_min = max(0, int(np.floor(x - radius)))
    x_max = min(nx - 1, int(np.floor(x + radius)))
    y_min = max(0, int(np.floor(y - radius)))
    y_max = min(ny - 1, int(np.floor(y + radius)))

    xs = np.arange(x_min, x_max + 1, dtype=np.float64)
    ys = np.arange(y_min, y_max + 1, dtype=np.float64)
    sqrt2 = np.sqrt(2.0)
    dx1 = (xs - 0.5 - x) / (sqrt2 * sigma)
    dx2 = (xs + 0.5 - x) / (sqrt2 * sigma)
    dy1 = (ys - 0.5 - y) / (sqrt2 * sigma)
    dy2 = (ys + 0.5 - y) / (sqrt2 * sigma)
    ex = _erf_vec(dx2) - _erf_vec(dx1)
    ey = _erf_vec(dy2) - _erf_vec(dy1)
    img[y_min : y_max + 1, x_min : x_max + 1] += 0.25 * flux * ey[:, None] * ex[None, :]


def nearest_neighbor_dist(x: np.ndarray, y: np.ndarray, chunk: int = 512) -> np.ndarray:
    n = x.size
    out = np.full(n, np.inf, dtype=np.float64)
    for i in range(0, n, chunk):
        xi = x[i : i + chunk][:, None]
        yi = y[i : i + chunk][:, None]
        dx = xi - x[None, :]
        dy = yi - y[None, :]
        d2 = dx * dx + dy * dy
        for j in range(i, min(i + chunk, n)):
            d2[j - i, j] = np.inf
        out[i : i + chunk] = np.sqrt(np.min(d2, axis=1))
    return out


def stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "rms": np.nan, "p90": np.nan, "p99": np.nan}
    return {
        "n": float(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "rms": float(np.sqrt(np.mean(values * values))),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
    }


def print_stats(label: str, values: np.ndarray) -> None:
    s = stats(values)
    print(
        f"{label}: n={int(s['n'])} mean={s['mean']:.4g} median={s['median']:.4g} "
        f"rms={s['rms']:.4g} p90={s['p90']:.4g} p99={s['p99']:.4g}"
    )


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    sigma = args.fwhm / 2.355 if args.sig is None else args.sig
    margin = int(np.ceil(4.0 * sigma + 2.0))
    if margin * 2 >= args.size:
        raise SystemExit("Image too small for chosen FWHM/sigma.")

    x = rng.uniform(margin, args.size - margin, args.n_stars)
    y = rng.uniform(margin, args.size - margin, args.n_stars)
    flux = rng.uniform(args.flux_min, args.flux_max, args.n_stars)

    data = np.full((args.size, args.size), args.background, dtype=np.float64)
    for xi, yi, fi in zip(x, y, flux):
        add_gaussian_stamp(data, xi, yi, fi, sigma)

    if args.noise > 0.0:
        data += rng.normal(0.0, args.noise, size=data.shape)

    # winpos expects background-subtracted data
    data -= args.background

    n_eval = min(args.n_eval, args.n_stars)
    idx = rng.choice(args.n_stars, size=n_eval, replace=False)
    x_true = x[idx]
    y_true = y[idx]

    x_init = x_true + rng.normal(0.0, args.pos_noise, size=n_eval)
    y_init = y_true + rng.normal(0.0, args.pos_noise, size=n_eval)
    x_init = np.clip(x_init, 0.0, args.size - 1.0)
    y_init = np.clip(y_init, 0.0, args.size - 1.0)

    segmap = None
    seg_id = None
    if args.use_segmap:
        thresh = args.seg_thresh_sigma * args.noise if args.noise > 0.0 else args.seg_thresh_sigma
        objects, segmap = sep.extract(
            data,
            thresh,
            minarea=5,
            segmentation_map=True,
            deblend_fwhm=args.deblend_fwhm,
            deblend_method=args.deblend_method,
        )
        if objects.size > 0:
            obj_ids = np.arange(1, objects.size + 1, dtype=np.int32)
            dx0 = x_init[:, None] - objects["x"][None, :]
            dy0 = y_init[:, None] - objects["y"][None, :]
            match = np.argmin(dx0 * dx0 + dy0 * dy0, axis=1)
            seg_id = obj_ids[match]
        else:
            segmap = None
            seg_id = None

    x_win, y_win, flag = sep.winpos(
        data, x_init, y_init, sigma, segmap=segmap, seg_id=seg_id
    )

    err_init = np.hypot(x_init - x_true, y_init - y_true)
    err_win = np.hypot(x_win - x_true, y_win - y_true)
    shift = np.hypot(x_win - x_init, y_win - y_init)

    improved = err_win < err_init
    worsened = err_win > err_init
    catastrophic = err_win > args.fwhm

    nn = nearest_neighbor_dist(x, y)
    nn_eval = nn[idx] / args.fwhm
    crowded = nn_eval < args.crowd_thresh

    print("Synthetic crowded field")
    print(
        f"- size={args.size}, nstars={args.n_stars}, n_eval={n_eval}, fwhm={args.fwhm:.2f}, sigma={sigma:.3f}"
    )
    print(f"- background={args.background:.2f}, noise={args.noise:.2f}, pos_noise={args.pos_noise:.2f}")
    print(f"- crowd_thresh={args.crowd_thresh:.2f} FWHM")
    if args.use_segmap:
        nseg = int(seg_id.size) if seg_id is not None else 0
        print(
            f"- segmap=on, seg_thresh_sigma={args.seg_thresh_sigma:.2f}, "
            f"deblend_fwhm={args.deblend_fwhm:.2f}, deblend_method={args.deblend_method}, "
            f"matched_ids={nseg}"
        )
    print("")

    print("Overall errors (pixels)")
    print_stats("init", err_init)
    print_stats("winpos", err_win)
    print_stats("winpos shift", shift)
    print(
        f"improved={improved.mean():.3f} worsened={worsened.mean():.3f} "
        f"catastrophic(>1 FWHM)={catastrophic.mean():.3f} flags={np.mean(flag != 0):.3f}"
    )

    if np.any(crowded):
        print("")
        print("Crowded subset (nn < crowd_thresh)")
        print_stats("init", err_init[crowded])
        print_stats("winpos", err_win[crowded])
        print(
            f"improved={np.mean(improved[crowded]):.3f} "
            f"worsened={np.mean(worsened[crowded]):.3f} "
            f"catastrophic={np.mean(catastrophic[crowded]):.3f}"
        )

    if np.any(~crowded):
        print("")
        print("Isolated subset (nn >= crowd_thresh)")
        print_stats("init", err_init[~crowded])
        print_stats("winpos", err_win[~crowded])
        print(
            f"improved={np.mean(improved[~crowded]):.3f} "
            f"worsened={np.mean(worsened[~crowded]):.3f} "
            f"catastrophic={np.mean(catastrophic[~crowded]):.3f}"
        )

    try:
        edges = [float(x) for x in args.bins.split(",") if x.strip() != ""]
    except ValueError:
        print("")
        print("Invalid --bins format; skipping bin report.")
        return 0

    edges = sorted(edges)
    if edges:
        print("")
        print("Nearest-neighbor bins (FWHM units)")
        prev = 0.0
        for edge in edges:
            sel = (nn_eval >= prev) & (nn_eval < edge)
            print_stats(f"[{prev:.1f}, {edge:.1f})", err_win[sel])
            prev = edge
        sel = nn_eval >= prev
        print_stats(f"[{prev:.1f}, inf)", err_win[sel])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
