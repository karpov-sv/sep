#!/usr/bin/env python3
"""Compare PSF photometry vs aperture and optimal extraction across crowding levels."""

from __future__ import annotations

import argparse
import math
from math import erf

import numpy as np

try:
    import sep_x as sep
except ImportError as exc:
    raise SystemExit("sep is required; build the extension first.") from exc


def make_erf_psf(fwhm: float, size: int = 0, oversampling: int = 2) -> "sep.PSF":
    """Build a Gaussian PSF using exact erf pixel integrals (matching the
    synthetic data generation), to eliminate PSF-model mismatch bias."""
    if size <= 0:
        size = int(np.ceil(4.0 * fwhm))
        if size % 2 == 0:
            size += 1

    sigma = fwhm / 2.354820045
    ossize = size * oversampling
    sigma_os = sigma * oversampling
    cx = ossize // 2
    cy = ossize // 2

    xs = np.arange(ossize, dtype=np.float64)
    ys = np.arange(ossize, dtype=np.float64)
    sqrt2 = np.sqrt(2.0)
    ex = _erf_vec((xs + 0.5 - cx) / (sqrt2 * sigma_os)) - _erf_vec(
        (xs - 0.5 - cx) / (sqrt2 * sigma_os)
    )
    ey = _erf_vec((ys + 0.5 - cy) / (sqrt2 * sigma_os)) - _erf_vec(
        (ys - 0.5 - cy) / (sqrt2 * sigma_os)
    )
    stamp = 0.25 * ey[:, None] * ex[None, :]
    stamp /= stamp.sum()
    data = stamp[np.newaxis, :, :].astype(np.float32)
    return sep.PSF(data, sampling=1.0 / oversampling, degree=0, fwhm=fwhm)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PSF photometry vs aperture and optimal extraction."
    )
    parser.add_argument("--size", type=int, default=256, help="Image size (square).")
    parser.add_argument("--fwhm", type=float, default=3.5, help="PSF FWHM (pixels).")
    parser.add_argument(
        "--r-factor",
        type=float,
        default=3.0,
        help="Aperture radius in units of FWHM.",
    )
    parser.add_argument(
        "--nstars",
        type=str,
        default="30,80,150",
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
    parser.add_argument(
        "--pos-offset",
        type=float,
        default=0.3,
        help="Max initial position offset in pixels for PSF fit.",
    )
    parser.add_argument(
        "--psf-erf",
        action="store_true",
        default=False,
        help="Use erf-based PSF stamp (matches data generation exactly).",
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
    """Add a Gaussian source to the image using exact erf pixel integrals."""
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
    img[y_min : y_max + 1, x_min : x_max + 1] += (
        0.25 * flux * ey[:, None] * ex[None, :]
    )


def choose_positions(
    rng: np.random.Generator,
    nstars: int,
    size: int,
    margin: float,
    min_sep: float,
    max_tries: int = 100000,
) -> np.ndarray:
    """Place sources with a minimum pairwise separation."""
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
    """Compute fractional error statistics."""
    frac = (flux - truth) / truth
    med = float(np.median(frac))
    mad = float(np.median(np.abs(frac - med))) * 1.4826
    rmse = float(np.sqrt(np.mean(frac * frac)))
    p90 = float(np.quantile(np.abs(frac), 0.9))
    return {"median": med, "mad": mad, "rmse": rmse, "p90": p90}


def pos_metrics(
    xfit: np.ndarray,
    yfit: np.ndarray,
    xtrue: np.ndarray,
    ytrue: np.ndarray,
) -> dict[str, float]:
    """Compute position error statistics in pixels."""
    dr = np.sqrt((xfit - xtrue) ** 2 + (yfit - ytrue) ** 2)
    return {
        "median": float(np.median(dr)),
        "p90": float(np.quantile(dr, 0.9)),
        "max": float(np.max(dr)),
    }


def fmt_flux(m: dict[str, float]) -> str:
    return (
        f"med={m['median']:+.4f}  mad={m['mad']:.4f}  "
        f"rmse={m['rmse']:.4f}  p90={m['p90']:.4f}"
    )


def fmt_pos(m: dict[str, float]) -> str:
    return f"med={m['median']:.4f}  p90={m['p90']:.4f}  max={m['max']:.4f}"


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
    pos_offset: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synthetic field and return positions, fluxes, image,
    and perturbed initial positions."""
    sigma = fwhm / 2.354820045
    margin = max(r + 2.0, fwhm * 3.0)
    positions = choose_positions(rng, nstars, size, margin, min_sep)
    fluxes = rng.uniform(flux_min, flux_max, size=nstars)

    img = np.zeros((size, size), dtype=np.float64)
    for (x, y), flux in zip(positions, fluxes):
        add_gaussian_stamp(img, x, y, flux, sigma)

    if noise > 0.0:
        img += rng.normal(0.0, noise, size=img.shape)

    # Perturbed initial positions (as if from a detection catalog)
    init_pos = positions.copy()
    if pos_offset > 0.0:
        init_pos[:, 0] += rng.uniform(-pos_offset, pos_offset, size=nstars)
        init_pos[:, 1] += rng.uniform(-pos_offset, pos_offset, size=nstars)

    return positions, fluxes, img, init_pos


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    nstars_list = [int(x) for x in args.nstars.split(",") if x.strip()]
    r = args.r_factor * args.fwhm
    min_sep = args.min_sep * args.fwhm

    if args.psf_erf:
        psf = make_erf_psf(fwhm=args.fwhm)
        psf_label = "erf-integrated"
    else:
        psf = sep.PSF.from_gaussian(fwhm=args.fwhm)
        psf_label = "from_gaussian"

    print("PSF photometry vs aperture & optimal extraction")
    print(
        f"  size={args.size}, fwhm={args.fwhm:.2f}, r={r:.2f} ({args.r_factor:.1f}*FWHM), "
        f"noise={args.noise:.1f}, pos_offset={args.pos_offset:.2f}, trials={args.trials}"
    )
    print(
        f"  PSF stamp: {psf.stamp_width}x{psf.stamp_height}, "
        f"sampling={psf.sampling:.2f}, min_sep={args.min_sep:.1f}*FWHM, "
        f"model={psf_label}"
    )
    print("")

    for nstars in nstars_list:
        all_ap = []
        all_opt = []
        all_grp = []
        all_psf = []
        all_psf_grp = []
        all_psf_x = []
        all_psf_y = []
        all_psf_grp_x = []
        all_psf_grp_y = []
        all_truth = []
        all_xtrue = []
        all_ytrue = []
        all_nn = []

        for _ in range(args.trials):
            pos, fluxes, img, init_pos = run_case(
                rng,
                nstars,
                args.size,
                args.fwhm,
                r,
                min_sep,
                args.flux_min,
                args.flux_max,
                args.noise,
                args.pos_offset,
            )
            nn = nearest_neighbor_dist(pos)

            # --- Aperture photometry ---
            flux_ap, _, flag_ap = sep.sum_circle(
                img, pos[:, 0], pos[:, 1], r, err=args.noise, subpix=5
            )

            # --- Optimal extraction ---
            flux_opt, _, flag_opt = sep.sum_circle_optimal(
                img, pos[:, 0], pos[:, 1], r, args.fwhm,
                err=args.noise, subpix=5,
            )

            # --- Grouped optimal extraction ---
            flux_grp, _, flag_grp = sep.sum_circle_optimal(
                img, pos[:, 0], pos[:, 1], r, args.fwhm,
                err=args.noise, grouped=True, subpix=5,
            )

            # --- PSF photometry (from perturbed positions) ---
            flux_psf, _, xf_psf, yf_psf, flag_psf, _, _ = sep.psf_fit(
                img, init_pos[:, 0], init_pos[:, 1], psf,
                var=args.noise**2,
            )

            # --- Grouped PSF photometry ---
            flux_psf_g, _, xf_psf_g, yf_psf_g, flag_psf_g, _, _ = sep.psf_fit(
                img, init_pos[:, 0], init_pos[:, 1], psf,
                var=args.noise**2, grouped=True,
            )

            # Keep sources where no method has TRUNC or ALLMASKED flags
            bad = np.int16(0x0010 | 0x0040)  # SEP_APER_TRUNC | SEP_APER_ALLMASKED
            ok = (
                ((flag_ap & bad) == 0) & ((flag_opt & bad) == 0)
                & ((flag_grp & bad) == 0)
                & ((flag_psf & bad) == 0) & ((flag_psf_g & bad) == 0)
            )
            all_ap.append(flux_ap[ok])
            all_opt.append(flux_opt[ok])
            all_grp.append(flux_grp[ok])
            all_psf.append(flux_psf[ok])
            all_psf_grp.append(flux_psf_g[ok])
            all_psf_x.append(xf_psf[ok])
            all_psf_y.append(yf_psf[ok])
            all_psf_grp_x.append(xf_psf_g[ok])
            all_psf_grp_y.append(yf_psf_g[ok])
            all_truth.append(fluxes[ok])
            all_xtrue.append(pos[ok, 0])
            all_ytrue.append(pos[ok, 1])
            all_nn.append(nn[ok])

        ap = np.concatenate(all_ap)
        opt = np.concatenate(all_opt)
        grp = np.concatenate(all_grp)
        psf_f = np.concatenate(all_psf)
        psf_gf = np.concatenate(all_psf_grp)
        psf_x = np.concatenate(all_psf_x)
        psf_y = np.concatenate(all_psf_y)
        psf_gx = np.concatenate(all_psf_grp_x)
        psf_gy = np.concatenate(all_psf_grp_y)
        truth = np.concatenate(all_truth)
        xtrue = np.concatenate(all_xtrue)
        ytrue = np.concatenate(all_ytrue)
        nn = np.concatenate(all_nn)

        crowd = float(np.median(nn) / args.fwhm)
        iso_mask = nn > 2.5 * args.fwhm

        print(
            f"nstars={nstars:4d}  n={len(truth):4d}  "
            f"median NN={crowd:.2f} FWHM  isolated={iso_mask.sum()}"
        )

        # --- Flux accuracy ---
        print("  Flux fractional error:")
        methods = [
            ("aperture    ", ap),
            ("optimal     ", opt),
            ("opt_grouped ", grp),
            ("psf         ", psf_f),
            ("psf_grouped ", psf_gf),
        ]
        for label, flux in methods:
            m = metrics(flux, truth)
            line = f"    {label} all:  {fmt_flux(m)}"
            if iso_mask.any() and iso_mask.sum() > 5:
                m_iso = metrics(flux[iso_mask], truth[iso_mask])
                line += f"  | iso:  {fmt_flux(m_iso)}"
            print(line)

        # --- Position accuracy (PSF fit only) ---
        print("  Position error (pixels):")
        mp = pos_metrics(psf_x, psf_y, xtrue, ytrue)
        print(f"    psf          all:  {fmt_pos(mp)}")
        mpg = pos_metrics(psf_gx, psf_gy, xtrue, ytrue)
        print(f"    psf_grouped  all:  {fmt_pos(mpg)}")
        if iso_mask.any() and iso_mask.sum() > 5:
            mp_iso = pos_metrics(
                psf_x[iso_mask], psf_y[iso_mask],
                xtrue[iso_mask], ytrue[iso_mask],
            )
            print(f"    psf          iso:  {fmt_pos(mp_iso)}")
            mpg_iso = pos_metrics(
                psf_gx[iso_mask], psf_gy[iso_mask],
                xtrue[iso_mask], ytrue[iso_mask],
            )
            print(f"    psf_grouped  iso:  {fmt_pos(mpg_iso)}")

        print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
