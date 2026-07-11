#!/usr/bin/env python3
"""Compare SEP and photutils aperture photometry in a crowded, simulated field."""

from __future__ import annotations

import argparse
import sys

import numpy as np

try:
    import sep
except ImportError as exc:
    raise SystemExit("sep is required; build the extension first.") from exc

try:
    from astropy.stats import SigmaClip
    from photutils.aperture import (
        ApertureStats,
        CircularAnnulus,
        CircularAperture,
        aperture_photometry,
    )
except ImportError as exc:
    raise SystemExit("photutils + astropy are required for this script.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SEP vs photutils aperture photometry on a crowded field."
    )
    parser.add_argument("--size", type=int, default=256, help="Image size (square).")
    parser.add_argument("--n-stars", type=int, default=400, help="Number of stars.")
    parser.add_argument(
        "--n-eval", type=int, default=100, help="Number of stars to evaluate."
    )
    parser.add_argument("--fwhm", type=float, default=2.5, help="PSF FWHM (pixels).")
    parser.add_argument("--flux-min", type=float, default=500.0, help="Min flux.")
    parser.add_argument("--flux-max", type=float, default=5000.0, help="Max flux.")
    parser.add_argument(
        "--background", type=float, default=1000.0, help="Sky background level."
    )
    parser.add_argument(
        "--grad-x",
        type=float,
        default=0.0,
        help="Background gradient in x (per pixel).",
    )
    parser.add_argument(
        "--grad-y",
        type=float,
        default=0.0,
        help="Background gradient in y (per pixel).",
    )
    parser.add_argument("--noise", type=float, default=5.0, help="Noise sigma.")
    parser.add_argument(
        "--aper-radius", type=float, default=4.0, help="Aperture radius."
    )
    parser.add_argument(
        "--annulus-in", type=float, default=6.0, help="Annulus inner radius."
    )
    parser.add_argument(
        "--annulus-out", type=float, default=10.0, help="Annulus outer radius."
    )
    parser.add_argument(
        "--method",
        choices=("exact", "center", "subpixel"),
        default="exact",
        help="Photutils aperture method.",
    )
    parser.add_argument(
        "--subpixels",
        type=int,
        default=5,
        help="Photutils subpixels (method=subpixel).",
    )
    parser.add_argument(
        "--clip-sigma",
        type=float,
        default=3.0,
        help="Sigma value for sigma clipping.",
    )
    parser.add_argument(
        "--clip-iters",
        type=int,
        default=5,
        help="Max iterations for sigma clipping.",
    )
    parser.add_argument(
        "--clip-center",
        choices=("mean", "median"),
        default="median",
        help="Center function for sigma clipping.",
    )
    parser.add_argument(
        "--clip-std",
        choices=("std", "mad"),
        default="mad",
        help="Scale function for sigma clipping (mad uses MAD*1.4826 as a robust std estimate).",
    )
    parser.add_argument(
        "--compare-clip",
        action="store_true",
        help="Also run photutils with median/MAD sigma clipping for comparison.",
    )
    parser.add_argument(
        "--bench",
        action="store_true",
        help="Run a performance benchmark comparing SEP and photutils.",
    )
    parser.add_argument(
        "--bench-loops",
        type=int,
        default=5,
        help="Loops per benchmark repeat.",
    )
    parser.add_argument(
        "--bench-reps",
        type=int,
        default=5,
        help="Number of benchmark repeats.",
    )
    parser.add_argument(
        "--sep-subpix",
        type=int,
        default=None,
        help="SEP subpix (default chosen to match method).",
    )
    parser.add_argument("--seed", type=int, default=12345, help="RNG seed.")
    return parser.parse_args()


def add_gaussian_stamp(img: np.ndarray, x: float, y: float, flux: float, sigma: float) -> None:
    ny, nx = img.shape
    radius = int(np.ceil(4.0 * sigma))
    x_min = max(0, int(np.floor(x - radius)))
    x_max = min(nx - 1, int(np.floor(x + radius)))
    y_min = max(0, int(np.floor(y - radius)))
    y_max = min(ny - 1, int(np.floor(y + radius)))

    yy, xx = np.mgrid[y_min : y_max + 1, x_min : x_max + 1]
    r2 = (xx - x) ** 2 + (yy - y) ** 2
    norm = flux / (2.0 * np.pi * sigma * sigma)
    img[y_min : y_max + 1, x_min : x_max + 1] += norm * np.exp(-0.5 * r2 / sigma**2)


def simulate_field(args: argparse.Namespace):
    rng = np.random.default_rng(args.seed)
    size = args.size
    sigma = args.fwhm / 2.355

    margin = int(np.ceil(args.annulus_out + 4.0 * sigma + 2.0))
    if margin * 2 >= size:
        raise SystemExit("Image too small for chosen annulus/PSF.")

    x = rng.uniform(margin, size - margin, args.n_stars)
    y = rng.uniform(margin, size - margin, args.n_stars)
    flux = rng.uniform(args.flux_min, args.flux_max, args.n_stars)

    yy, xx = np.indices((size, size))
    bkg = (
        args.background
        + args.grad_x * (xx - 0.5 * (size - 1))
        + args.grad_y * (yy - 0.5 * (size - 1))
    )
    data = bkg.astype(np.float64, copy=True)

    for xi, yi, fi in zip(x, y, flux):
        add_gaussian_stamp(data, xi, yi, fi, sigma)

    if args.noise > 0.0:
        data += rng.normal(0.0, args.noise, size=data.shape)

    return data, x, y, flux, bkg, sigma


def summary_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "median": np.nan, "mad": np.nan}
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return {
        "n": float(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(median),
        "mad": float(mad),
    }


def main() -> int:
    args = parse_args()

    data, x_all, y_all, flux_all, bkg_map, sigma = simulate_field(args)

    rng = np.random.default_rng(args.seed + 1)
    n_eval = min(args.n_eval, x_all.size)
    eval_idx = rng.choice(x_all.size, size=n_eval, replace=False)

    x = x_all[eval_idx]
    y = y_all[eval_idx]
    flux = flux_all[eval_idx]

    positions = np.column_stack([x, y])

    method = args.method
    subpixels = args.subpixels
    if args.sep_subpix is None:
        if method == "exact":
            sep_subpix = 0
        elif method == "center":
            sep_subpix = 1
        else:
            sep_subpix = subpixels
    else:
        sep_subpix = args.sep_subpix

    aper = CircularAperture(positions, r=args.aper_radius)
    annulus = CircularAnnulus(positions, r_in=args.annulus_in, r_out=args.annulus_out)

    stdfunc = "std" if args.clip_std == "std" else "mad_std"
    sigma_clip = SigmaClip(
        sigma=args.clip_sigma,
        maxiters=args.clip_iters,
        cenfunc=args.clip_center,
        stdfunc=stdfunc,
    )
    sigma_clip_alt = None
    if args.compare_clip:
        sigma_clip_alt = SigmaClip(
            sigma=args.clip_sigma,
            maxiters=args.clip_iters,
            cenfunc="median",
            stdfunc="mad_std",
        )
    def compute_photutils_bkg(stats_clip):
        try:
            stats = ApertureStats(
                data,
                annulus,
                sigma_clip=stats_clip,
                sum_method=method,
                subpixels=subpixels,
            )
        except TypeError:
            stats = ApertureStats(
                data,
                annulus,
                sigma_clip=stats_clip,
                method=method,
                subpixels=subpixels,
            )
        return np.array(stats.mean, dtype=float)

    bkg_phot = compute_photutils_bkg(sigma_clip)
    bkg_phot_alt = None
    if sigma_clip_alt is not None:
        bkg_phot_alt = compute_photutils_bkg(sigma_clip_alt)

    phot_table = aperture_photometry(
        data, aper, method=method, subpixels=subpixels
    )
    sum_phot = np.array(phot_table["aperture_sum"], dtype=float)

    try:
        area = np.array(
            aper.area_overlap(data, method=method, subpixels=subpixels),
            dtype=float,
        )
    except Exception:
        try:
            area = np.array(
                aper.area_overlap(data.shape, method=method, subpixels=subpixels),
                dtype=float,
            )
        except Exception:
            area = np.full(n_eval, np.pi * args.aper_radius**2, dtype=float)

    flux_phot = sum_phot - bkg_phot * area

    mean, std, median, mad_std, mean_clip, flag = sep.stats_circann(
        data,
        x,
        y,
        args.annulus_in,
        args.annulus_out,
        subpix=sep_subpix,
        clip_sigma=args.clip_sigma,
        clip_iters=args.clip_iters,
    )
    sum_sep, _, flag_sum = sep.sum_circle(
        data, x, y, args.aper_radius, subpix=sep_subpix
    )
    flux_sep = sum_sep - mean_clip * area

    bkg_true = (
        args.background
        + args.grad_x * (x - 0.5 * (args.size - 1))
        + args.grad_y * (y - 0.5 * (args.size - 1))
    )
    flux_true = flux * (1.0 - np.exp(-0.5 * (args.aper_radius / sigma) ** 2))

    bkg_err_phot = bkg_phot - bkg_true
    bkg_err_phot_alt = None
    if bkg_phot_alt is not None:
        bkg_err_phot_alt = bkg_phot_alt - bkg_true
    bkg_err_sep = mean_clip - bkg_true
    flux_err_phot = flux_phot - flux_true
    flux_err_phot_alt = None
    if bkg_phot_alt is not None:
        flux_phot_alt = sum_phot - bkg_phot_alt * area
        flux_err_phot_alt = flux_phot_alt - flux_true
    flux_err_sep = flux_sep - flux_true

    print("Simulated field")
    print(f"- size={args.size}, nstars={args.n_stars}, n_eval={n_eval}")
    print(
        f"- fwhm={args.fwhm:.2f}, aper={args.aper_radius:.2f}, annulus=({args.annulus_in:.2f},{args.annulus_out:.2f})"
    )
    print(
        f"- background={args.background:.2f}, grad=({args.grad_x:.3g},{args.grad_y:.3g}), noise={args.noise:.2f}"
    )
    print(
        f"- method={method}, photutils_subpixels={subpixels}, sep_subpix={sep_subpix}"
    )
    print(
        f"- sigma_clip: sigma={args.clip_sigma}, iters={args.clip_iters}, "
        f"center={args.clip_center}, scale={args.clip_std}"
    )
    print("")

    def show(label: str, stats: dict[str, float]) -> None:
        print(
            f"{label}: n={int(stats['n'])} mean={stats['mean']:.4g} "
            f"median={stats['median']:.4g} std={stats['std']:.4g} mad={stats['mad']:.4g}"
        )

    print("Background error (estimate - true)")
    show("photutils", summary_stats(bkg_err_phot))
    show("sep", summary_stats(bkg_err_sep))
    show("sep - photutils", summary_stats(bkg_err_sep - bkg_err_phot))
    if bkg_err_phot_alt is not None:
        show("photutils(median/MAD)", summary_stats(bkg_err_phot_alt))
        show(
            "sep - photutils(median/MAD)",
            summary_stats(bkg_err_sep - bkg_err_phot_alt),
        )
    print("")

    print("Photometry error (background-subtracted - true)")
    show("photutils", summary_stats(flux_err_phot))
    show("sep", summary_stats(flux_err_sep))
    show("sep - photutils", summary_stats(flux_err_sep - flux_err_phot))
    if flux_err_phot_alt is not None:
        show("photutils(median/MAD)", summary_stats(flux_err_phot_alt))
        show(
            "sep - photutils(median/MAD)",
            summary_stats(flux_err_sep - flux_err_phot_alt),
        )
    print("")

    flag_frac = np.mean(flag != 0)
    flag_sum_frac = np.mean(flag_sum != 0)
    print(f"SEP annulus flags (nonzero): {flag_frac:.3f}")
    print(f"SEP aperture flags (nonzero): {flag_sum_frac:.3f}")

    if args.bench:
        import gc
        import time

        def bench(fn, loops, reps):
            times = []
            gc_was_enabled = gc.isenabled()
            gc.disable()
            try:
                for _ in range(reps):
                    t0 = time.perf_counter()
                    for __ in range(loops):
                        fn()
                    t1 = time.perf_counter()
                    times.append((t1 - t0) / loops)
            finally:
                if gc_was_enabled:
                    gc.enable()
            return np.array(times, dtype=float)

        def photutils_run():
            bkg = compute_photutils_bkg(sigma_clip)
            phot_table = aperture_photometry(
                data, aper, method=method, subpixels=subpixels
            )
            sum_phot = np.array(phot_table["aperture_sum"], dtype=float)
            _ = sum_phot - bkg * area

        def sep_run():
            mean_bkg = sep.stats_circann(
                data,
                x,
                y,
                args.annulus_in,
                args.annulus_out,
                subpix=sep_subpix,
                clip_sigma=args.clip_sigma,
                clip_iters=args.clip_iters,
            )[4]
            sum_sep = sep.sum_circle(
                data, x, y, args.aper_radius, subpix=sep_subpix
            )[0]
            _ = sum_sep - mean_bkg * area

        # Warm-up
        photutils_run()
        sep_run()

        t_phot = bench(photutils_run, args.bench_loops, args.bench_reps)
        t_sep = bench(sep_run, args.bench_loops, args.bench_reps)

        def report(label, times):
            print(
                f"{label}: mean={times.mean() * 1e3:.2f} ms "
                f"std={times.std() * 1e3:.2f} ms "
                f"median={np.median(times) * 1e3:.2f} ms "
                f"(n={times.size})"
            )

        print("")
        print("Benchmark (per run)")
        report("photutils", t_phot)
        report("sep", t_sep)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
