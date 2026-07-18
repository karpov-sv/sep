#!/usr/bin/env python3
"""Compare aperture vs optimal-extraction photometry in crowded fields."""

from __future__ import annotations

import argparse
import math
from math import erf

import numpy as np

try:
    import sep_x as sep
except ImportError as exc:
    raise SystemExit("sep is required; build the extension first.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare sum_circle vs optimal extraction across crowding levels."
    )
    parser.add_argument("--size", type=int, default=256, help="Image size (square).")
    parser.add_argument("--fwhm", type=float, default=2.5, help="PSF FWHM (pixels).")
    parser.add_argument(
        "--r-factor",
        type=float,
        default=3.0,
        help="Aperture radius in units of FWHM.",
    )
    parser.add_argument(
        "--nstars",
        type=str,
        default="50,150,300,450",
        help="Comma-separated star counts for crowding levels.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of realizations per crowding level.",
    )
    parser.add_argument(
        "--min-sep",
        type=float,
        default=0.5,
        help="Minimum separation in FWHM units.",
    )
    parser.add_argument(
        "--flux-min",
        type=float,
        default=2000.0,
        help="Minimum source flux.",
    )
    parser.add_argument(
        "--flux-max",
        type=float,
        default=12000.0,
        help="Maximum source flux.",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=5.0,
        help="Gaussian noise sigma (background-subtracted).",
    )
    parser.add_argument("--seed", type=int, default=12345, help="RNG seed.")
    return parser.parse_args()


def _erf_vec(values: np.ndarray) -> np.ndarray:
    if hasattr(np, "erf"):
        return np.erf(values)
    return np.vectorize(erf)(values)


def add_gaussian_stamp(
    img: np.ndarray, x: float, y: float, flux: float, sigma: float
) -> None:
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


def choose_positions(
    rng: np.random.Generator,
    nstars: int,
    size: int,
    margin: float,
    min_sep: float,
    max_tries: int = 100000,
) -> np.ndarray:
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
            raise SystemExit("Failed to place stars; increase size or reduce density.")
    return np.array(positions, dtype=np.float64)


def nearest_neighbor_dist(positions: np.ndarray) -> np.ndarray:
    n = positions.shape[0]
    if n < 2:
        return np.array([np.inf])
    dmin = np.full(n, np.inf, dtype=np.float64)
    for i in range(n):
        dx = positions[i, 0] - positions[:, 0]
        dy = positions[i, 1] - positions[:, 1]
        dist2 = dx * dx + dy * dy
        dist2[i] = np.inf
        dmin[i] = np.sqrt(dist2.min())
    return dmin


def metrics(flux: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    frac = (flux - truth) / truth
    med = float(np.median(frac))
    mad = float(np.median(np.abs(frac - med))) * 1.4826
    rmse = float(np.sqrt(np.mean(frac * frac)))
    p90 = float(np.quantile(np.abs(frac), 0.9))
    return {
        "median": med,
        "mad": mad,
        "rmse": rmse,
        "p90": p90,
    }


def run_case(
    rng: np.random.Generator,
    nstars: int,
    size: int,
    fwhm: float,
    r: float,
    min_sep: float,
    flux_min: float,
    flux_max: float,
    noise: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sigma = fwhm / 2.354820045
    margin = r + 2.0
    positions = choose_positions(rng, nstars, size, margin, min_sep)
    fluxes = rng.uniform(flux_min, flux_max, size=nstars)

    img = np.zeros((size, size), dtype=np.float64)
    for (x, y), flux in zip(positions, fluxes):
        add_gaussian_stamp(img, x, y, flux, sigma)

    if noise > 0.0:
        img += rng.normal(0.0, noise, size=img.shape)

    return positions, fluxes, img


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    nstars_list = [int(x) for x in args.nstars.split(",") if x.strip()]
    r = args.r_factor * args.fwhm
    min_sep = args.min_sep * args.fwhm

    print("Optimal extraction vs aperture photometry")
    print(
        f"- size={args.size}, fwhm={args.fwhm:.2f}, r={r:.2f} ({args.r_factor:.2f}*FWHM), "
        f"noise={args.noise:.2f}, trials={args.trials}"
    )
    print("")

    for nstars in nstars_list:
        all_ap = []
        all_opt = []
        all_grp = []
        all_truth = []
        all_nn = []

        for _ in range(args.trials):
            pos, fluxes, img = run_case(
                rng,
                nstars,
                args.size,
                args.fwhm,
                r,
                min_sep,
                args.flux_min,
                args.flux_max,
                args.noise,
            )
            nn = nearest_neighbor_dist(pos)

            flux_ap, _, flag_ap = sep.sum_circle(
                img, pos[:, 0], pos[:, 1], r, err=args.noise, subpix=5
            )
            flux_opt, _, flag_opt = sep.sum_circle_optimal(
                img, pos[:, 0], pos[:, 1], r, args.fwhm, err=args.noise, subpix=5
            )
            flux_grp, _, flag_grp = sep.sum_circle_optimal(
                img,
                pos[:, 0],
                pos[:, 1],
                r,
                args.fwhm,
                err=args.noise,
                grouped=True,
                subpix=5,
            )

            ok = (flag_ap == 0) & (flag_opt == 0) & (flag_grp == 0)
            all_ap.append(flux_ap[ok])
            all_opt.append(flux_opt[ok])
            all_grp.append(flux_grp[ok])
            all_truth.append(fluxes[ok])
            all_nn.append(nn[ok])

        ap = np.concatenate(all_ap)
        opt = np.concatenate(all_opt)
        grp = np.concatenate(all_grp)
        truth = np.concatenate(all_truth)
        nn = np.concatenate(all_nn)

        crowd = float(np.median(nn) / args.fwhm)
        iso_mask = nn > 2.5 * args.fwhm

        print(f"Crowding level: nstars={nstars}, median NN={crowd:.2f} FWHM")
        for label, flux in ("aperture", ap), ("optimal", opt), ("grouped", grp):
            m = metrics(flux, truth)
            print(
                f"  {label:8s} all:   med={m['median']:+.3f}  mad={m['mad']:.3f}  "
                f"rmse={m['rmse']:.3f}  p90={m['p90']:.3f}"
            )
            if iso_mask.any():
                m_iso = metrics(flux[iso_mask], truth[iso_mask])
                print(
                    f"  {label:8s} iso:   med={m_iso['median']:+.3f}  "
                    f"mad={m_iso['mad']:.3f}  rmse={m_iso['rmse']:.3f}  "
                    f"p90={m_iso['p90']:.3f}"
                )
        print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
