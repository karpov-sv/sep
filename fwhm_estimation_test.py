#!/usr/bin/env python3
"""Standalone test for SEP FWHM estimation across PSF sizes and crowding."""

from __future__ import annotations

import argparse
import math
from math import erf

import numpy as np

try:
    import sep
except ImportError as exc:
    raise SystemExit("sep is required; build the extension first.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate SEP FWHM estimation across PSF sizes and crowding."
    )
    parser.add_argument("--size", type=int, default=256, help="Image size (square).")
    parser.add_argument(
        "--fwhm-list",
        type=str,
        default="1.2,1.6,2.0,2.5,3.5,5.0,7.0",
        help="Comma-separated FWHM values (pixels).",
    )
    parser.add_argument(
        "--nstars",
        type=int,
        default=200,
        help="Number of stars per realization.",
    )
    parser.add_argument(
        "--min-sep",
        type=str,
        default="0.8,1.2,2.0,4.0",
        help="Comma-separated minimum separations in units of FWHM.",
    )
    parser.add_argument(
        "--flux",
        type=float,
        default=5000.0,
        help="Per-star flux (background-subtracted).",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=5.0,
        help="Gaussian noise sigma.",
    )
    parser.add_argument(
        "--thresh",
        type=float,
        default=5.0,
        help="Detection threshold in sigma (relative if err is provided).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Trials per FWHM/separation setting.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="RNG seed.",
    )
    return parser.parse_args()


def _erf_vec(values: np.ndarray) -> np.ndarray:
    if hasattr(np, "erf"):
        return np.erf(values)
    return np.vectorize(erf)(values)


def gaussian_pixel_integral(dx: np.ndarray, dy: np.ndarray, sigma: float) -> np.ndarray:
    inv = 1.0 / (math.sqrt(2.0) * sigma)
    ex = _erf_vec((dx + 0.5) * inv) - _erf_vec((dx - 0.5) * inv)
    ey = _erf_vec((dy + 0.5) * inv) - _erf_vec((dy - 0.5) * inv)
    return 0.25 * ex * ey


def add_gaussian_scene(
    img: np.ndarray, x: np.ndarray, y: np.ndarray, fwhm: float, flux: float
) -> None:
    sigma = fwhm / 2.354820045
    ny, nx = img.shape
    radius = int(np.ceil(4.0 * sigma))

    for xi, yi in zip(x, y):
        x_min = max(0, int(np.floor(xi - radius)))
        x_max = min(nx - 1, int(np.floor(xi + radius)))
        y_min = max(0, int(np.floor(yi - radius)))
        y_max = min(ny - 1, int(np.floor(yi + radius)))

        xs = np.arange(x_min, x_max + 1, dtype=np.float64)
        ys = np.arange(y_min, y_max + 1, dtype=np.float64)
        dx = xs[None, :] - xi
        dy = ys[:, None] - yi
        img[y_min : y_max + 1, x_min : x_max + 1] += (
            flux * gaussian_pixel_integral(dx, dy, sigma)
        )


def choose_positions(
    rng: np.random.Generator,
    nstars: int,
    size: int,
    margin: float,
    min_sep: float,
    max_tries: int = 200000,
) -> tuple[np.ndarray, np.ndarray]:
    positions: list[tuple[float, float]] = []
    for _ in range(nstars):
        for _ in range(max_tries):
            x = rng.uniform(margin, size - margin)
            y = rng.uniform(margin, size - margin)
            if not positions:
                positions.append((x, y))
                break
            dx = np.array([x - p[0] for p in positions])
            dy = np.array([y - p[1] for p in positions])
            if np.all(dx * dx + dy * dy >= min_sep * min_sep):
                positions.append((x, y))
                break
        else:
            raise SystemExit("Failed to place stars; reduce density or size.")
    pos = np.array(positions, dtype=np.float64)
    return pos[:, 0], pos[:, 1]


def match_detections(
    x_det: np.ndarray,
    y_det: np.ndarray,
    fwhm_det: np.ndarray,
    x_true: np.ndarray,
    y_true: np.ndarray,
    fwhm_true: float,
    tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    if x_det.size == 0:
        return np.array([], dtype=float), np.array([], dtype=int)

    used = np.zeros(x_true.size, dtype=bool)
    errs = []
    for xd, yd, fd in zip(x_det, y_det, fwhm_det):
        dx = x_true - xd
        dy = y_true - yd
        d2 = dx * dx + dy * dy
        idx = np.argmin(d2)
        if d2[idx] <= tol * tol and not used[idx]:
            used[idx] = True
            errs.append((fd - fwhm_true) / fwhm_true)
    return np.asarray(errs, dtype=float), used


def summarize(errors: np.ndarray) -> str:
    if errors.size == 0:
        return "no matches"
    rel = errors
    med = np.median(rel)
    mad = np.median(np.abs(rel - med))
    p90 = np.percentile(np.abs(rel), 90.0)
    return f"n={rel.size} med={med:+.4f} mad={mad:.4f} p90={p90:.4f}"


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    fwhm_list = [float(v.strip()) for v in args.fwhm_list.split(",") if v.strip()]
    min_sep_list = [float(v.strip()) for v in args.min_sep.split(",") if v.strip()]

    print("SEP FWHM estimation test")
    print(
        f"- size={args.size}, nstars={args.nstars}, flux={args.flux}, "
        f"noise={args.noise}, thresh={args.thresh}, trials={args.trials}"
    )

    for fwhm in fwhm_list:
        sigma = fwhm / 2.354820045
        margin = 5.0 * sigma
        print(f"\nFWHM={fwhm:.2f} px")

        for min_sep_factor in min_sep_list:
            min_sep = min_sep_factor * fwhm
            all_errs = []
            total_matched = 0
            total_true = 0

            for _ in range(args.trials):
                x, y = choose_positions(rng, args.nstars, args.size, margin, min_sep)
                img = np.zeros((args.size, args.size), dtype=np.float64)
                add_gaussian_scene(img, x, y, fwhm, args.flux)
                img += rng.normal(0.0, args.noise, img.shape)

                objects = sep.extract(img, args.thresh, err=args.noise)
                errs, used = match_detections(
                    objects["x"],
                    objects["y"],
                    objects["fwhm"],
                    x,
                    y,
                    fwhm,
                    tol=0.5 * fwhm,
                )
                all_errs.append(errs)
                total_matched += errs.size
                total_true += x.size

            if all_errs:
                errors = np.concatenate(all_errs)
            else:
                errors = np.array([], dtype=float)

            match_frac = total_matched / total_true if total_true > 0 else 0.0
            summary = summarize(errors)
            print(
                f"  min_sep={min_sep_factor:.2f}*FWHM "
                f"match={match_frac:.2f} {summary}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
