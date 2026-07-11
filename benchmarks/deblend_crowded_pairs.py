#!/usr/bin/env python3
"""Standalone deblending test on synthetic crowded star pairs."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from math import erf

import numpy as np

try:
    import sep
except ImportError as exc:
    raise SystemExit("sep is required; build the extension first.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SEP deblending on controlled close pairs."
    )
    parser.add_argument("--size", type=int, default=256, help="Image size (square).")
    parser.add_argument(
        "--fwhm", type=float, default=2.5, help="PSF FWHM (pixels)."
    )
    parser.add_argument(
        "--seps",
        type=str,
        default="0.8,1.0,1.2,1.5,2.0,2.5,3.0",
        help="Comma-separated pair separations in FWHM units.",
    )
    parser.add_argument(
        "--pairs-per-sep",
        type=int,
        default=20,
        help="Number of pairs to simulate per separation value.",
    )
    parser.add_argument("--flux-min", type=float, default=800.0, help="Min flux.")
    parser.add_argument("--flux-max", type=float, default=4000.0, help="Max flux.")
    parser.add_argument(
        "--flux-ratio",
        type=float,
        default=1.0,
        help="Flux ratio for the second star (flux2 = ratio * flux1).",
    )
    parser.add_argument(
        "--background", type=float, default=1000.0, help="Sky background level."
    )
    parser.add_argument("--noise", type=float, default=5.0, help="Noise sigma.")
    parser.add_argument(
        "--thresh-sigma",
        type=float,
        default=3.0,
        help="Detection threshold in sigma (relative).",
    )
    parser.add_argument(
        "--minarea", type=int, default=5, help="Minimum area for detection."
    )
    parser.add_argument(
        "--deblend-nthresh", type=int, default=32, help="Deblend thresholds."
    )
    parser.add_argument(
        "--deblend-cont", type=float, default=0.005, help="Deblend contrast."
    )
    parser.add_argument(
        "--deblend-fwhm",
        type=float,
        default=0.0,
        help="Fixed PSF FWHM for deblending (0 disables).",
    )
    parser.add_argument(
        "--deblend-method",
        type=str,
        default="threshold",
        choices=("threshold", "watershed"),
        help="Deblend method to use.",
    )
    parser.add_argument(
        "--match-radius",
        type=float,
        default=0.75,
        help="Match radius in FWHM units.",
    )
    parser.add_argument(
        "--min-center-sep",
        type=float,
        default=5.0,
        help="Minimum separation between pair centers (FWHM units).",
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


def choose_center(
    rng: np.random.Generator,
    centers: list[tuple[float, float]],
    size: int,
    margin: float,
    min_center_sep: float,
    max_tries: int = 2000,
) -> tuple[float, float]:
    for _ in range(max_tries):
        cx = rng.uniform(margin, size - margin)
        cy = rng.uniform(margin, size - margin)
        if not centers:
            return cx, cy
        dx = np.array([cx - c[0] for c in centers])
        dy = np.array([cy - c[1] for c in centers])
        if np.all(dx * dx + dy * dy >= min_center_sep * min_center_sep):
            return cx, cy
    raise SystemExit("Failed to place pairs without overlap; increase size or reduce pairs.")


def classify_pair(
    p1: tuple[float, float],
    p2: tuple[float, float],
    det_x: np.ndarray,
    det_y: np.ndarray,
    match_radius: float,
) -> str:
    if det_x.size == 0:
        return "missed"
    dx1 = det_x - p1[0]
    dy1 = det_y - p1[1]
    dx2 = det_x - p2[0]
    dy2 = det_y - p2[1]
    d1 = np.hypot(dx1, dy1)
    d2 = np.hypot(dx2, dy2)
    set1 = set(np.where(d1 <= match_radius)[0])
    set2 = set(np.where(d2 <= match_radius)[0])
    union = set1 | set2
    if len(union) >= 2 and set1 and set2:
        return "split"
    if len(union) == 1:
        return "merged" if set1 and set2 else "one_detected"
    if len(union) >= 2 and (not set1 or not set2):
        return "over_split"
    return "missed"


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    sigma = args.fwhm / 2.354820045
    sep_fwhm = [float(x) for x in args.seps.split(",") if x.strip()]
    max_sep = max(sep_fwhm) * args.fwhm
    margin = int(np.ceil(4.0 * sigma + 0.5 * max_sep + 2.0))
    min_center_sep = args.min_center_sep * args.fwhm
    match_radius = args.match_radius * args.fwhm

    if margin * 2 >= args.size:
        raise SystemExit("Image too small for chosen FWHM / separations.")

    data = np.full((args.size, args.size), args.background, dtype=np.float64)
    pairs: list[tuple[float, tuple[float, float], tuple[float, float]]] = []
    centers: list[tuple[float, float]] = []

    for sep_f in sep_fwhm:
        sep_pix = sep_f * args.fwhm
        for _ in range(args.pairs_per_sep):
            cx, cy = choose_center(
                rng, centers, args.size, margin, min_center_sep
            )
            angle = rng.uniform(0.0, 2.0 * math.pi)
            dx = 0.5 * sep_pix * math.cos(angle)
            dy = 0.5 * sep_pix * math.sin(angle)
            x1, y1 = cx + dx, cy + dy
            x2, y2 = cx - dx, cy - dy
            f1 = rng.uniform(args.flux_min, args.flux_max)
            f2 = f1 * args.flux_ratio
            add_gaussian_stamp(data, x1, y1, f1, sigma)
            add_gaussian_stamp(data, x2, y2, f2, sigma)
            pairs.append((sep_f, (x1, y1), (x2, y2)))
            centers.append((cx, cy))

    if args.noise > 0.0:
        data += rng.normal(0.0, args.noise, size=data.shape)

    # Use background-subtracted data for detection.
    data -= args.background

    objects = sep.extract(
        data,
        args.thresh_sigma,
        err=args.noise if args.noise > 0.0 else None,
        minarea=args.minarea,
        deblend_nthresh=args.deblend_nthresh,
        deblend_cont=args.deblend_cont,
        deblend_fwhm=args.deblend_fwhm,
        deblend_method=args.deblend_method,
    )

    det_x = objects["x"]
    det_y = objects["y"]

    counts: dict[float, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    totals: dict[str, int] = defaultdict(int)
    for sep_f, p1, p2 in pairs:
        outcome = classify_pair(p1, p2, det_x, det_y, match_radius)
        counts[sep_f][outcome] += 1
        totals[outcome] += 1

    print("Deblending pair test")
    print(
        f"- size={args.size}, fwhm={args.fwhm:.2f}, sigma={sigma:.3f}, "
        f"pairs_per_sep={args.pairs_per_sep}, noise={args.noise:.2f}"
    )
    print(
        f"- thresh={args.thresh_sigma:.2f} sigma, minarea={args.minarea}, "
        f"deblend_nthresh={args.deblend_nthresh}, deblend_cont={args.deblend_cont}, "
        f"deblend_fwhm={args.deblend_fwhm:.2f}, deblend_method={args.deblend_method}"
    )
    print(
        f"- match_radius={args.match_radius:.2f} FWHM, min_center_sep={args.min_center_sep:.2f} FWHM"
    )
    print(f"- detections={det_x.size}")
    print("")

    print("Per-separation results (fractions)")
    for sep_f in sorted(counts.keys()):
        n = sum(counts[sep_f].values())
        split = counts[sep_f].get("split", 0) / n
        merged = counts[sep_f].get("merged", 0) / n
        one = counts[sep_f].get("one_detected", 0) / n
        missed = counts[sep_f].get("missed", 0) / n
        over = counts[sep_f].get("over_split", 0) / n
        print(
            f"{sep_f:.2f} FWHM: split={split:.3f} merged={merged:.3f} "
            f"one_detected={one:.3f} missed={missed:.3f} over_split={over:.3f}"
        )

    print("")
    total_pairs = len(pairs)
    print("Overall (fractions)")
    print(
        f"split={totals['split'] / total_pairs:.3f} "
        f"merged={totals['merged'] / total_pairs:.3f} "
        f"one_detected={totals['one_detected'] / total_pairs:.3f} "
        f"missed={totals['missed'] / total_pairs:.3f} "
        f"over_split={totals['over_split'] / total_pairs:.3f}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
