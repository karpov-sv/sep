#!/usr/bin/env python
"""Benchmark aperture and PSF photometry performance on a large synthetic image."""

from __future__ import annotations

import argparse
import time

import numpy as np
import sep_x as sep


def make_image(size: int, nobj: int, background: float, noise: float, rng: np.random.Generator) -> np.ndarray:
    data = rng.normal(loc=background, scale=noise, size=(size, size)).astype(np.float32)

    # Add simple point sources by dumping flux into nearest pixel (cheap).
    ix = rng.integers(0, size, size=nobj)
    iy = rng.integers(0, size, size=nobj)
    flux = rng.uniform(500.0, 5000.0, size=nobj).astype(np.float32)
    np.add.at(data, (iy, ix), flux)
    return data


def make_positions(size: int, nobj: int, r: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    margin = r + 2.0
    x = rng.uniform(margin, size - margin, size=nobj).astype(np.float64)
    y = rng.uniform(margin, size - margin, size=nobj).astype(np.float64)
    return x, y


def time_call(label: str, func, repeat: int, width: int = 38) -> float:
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    tmin = min(times)
    print(f"  {label:{width}s}  {tmin:8.3f} s")
    return tmin


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, default=4096, help="Image size (square).")
    ap.add_argument("--nobj", type=int, default=5000, help="Number of objects.")
    ap.add_argument("--r", type=float, default=3.0, help="Aperture radius.")
    ap.add_argument("--fwhm", type=float, default=2.5, help="FWHM for optimal extraction.")
    ap.add_argument("--rin", type=float, default=6.0, help="Background annulus inner radius.")
    ap.add_argument("--rout", type=float, default=8.0, help="Background annulus outer radius.")
    ap.add_argument("--noise", type=float, default=5.0, help="Per-pixel noise (err).")
    ap.add_argument("--background", type=float, default=1000.0, help="Mean background level.")
    ap.add_argument("--subpix", type=int, default=1, help="Subpixel sampling (0=exact).")
    ap.add_argument("--maxiter", type=int, default=20, help="Max PSF fitting iterations.")
    ap.add_argument("--repeat", type=int, default=3, help="Repeat each benchmark; report min.")
    ap.add_argument("--seed", type=int, default=12345, help="RNG seed.")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    data = make_image(args.size, args.nobj, args.background, args.noise, rng)
    x, y = make_positions(args.size, args.nobj, args.r, rng)

    r = np.full(args.nobj, args.r, dtype=np.float64)
    fwhm = np.full(args.nobj, args.fwhm, dtype=np.float64)
    bkgann = (args.rin, args.rout)
    psf = sep.PSF.from_gaussian(fwhm=args.fwhm)

    w = 38
    n = args.nobj
    timings = {}

    print("Aperture & PSF photometry performance")
    print(f"  size={args.size}x{args.size}, nobj={n}, r={args.r}, fwhm={args.fwhm}")
    print(f"  subpix={args.subpix}, bkgann=({args.rin}, {args.rout}), "
          f"maxiter={args.maxiter}, repeat={args.repeat}")
    print()
    print(f"  {'Method':{w}s}      time")
    print(f"  {'-' * (w + 10)}")

    timings["sum_circle"] = time_call(
        "sum_circle",
        lambda: sep.sum_circle(data, x, y, r, err=args.noise, subpix=args.subpix),
        args.repeat, w,
    )
    timings["sum_circle (bkgann)"] = time_call(
        "sum_circle (bkgann)",
        lambda: sep.sum_circle(
            data, x, y, r, err=args.noise, bkgann=bkgann, subpix=args.subpix
        ),
        args.repeat, w,
    )
    timings["sum_circle_optimal"] = time_call(
        "sum_circle_optimal",
        lambda: sep.sum_circle_optimal(
            data, x, y, r, fwhm, err=args.noise, subpix=args.subpix
        ),
        args.repeat, w,
    )
    timings["sum_circle_optimal (bkgann)"] = time_call(
        "sum_circle_optimal (bkgann)",
        lambda: sep.sum_circle_optimal(
            data, x, y, r, fwhm, err=args.noise, bkgann=bkgann, subpix=args.subpix
        ),
        args.repeat, w,
    )
    timings["sum_circle_optimal grouped"] = time_call(
        "sum_circle_optimal grouped",
        lambda: sep.sum_circle_optimal(
            data, x, y, r, fwhm, err=args.noise, grouped=True, subpix=args.subpix
        ),
        args.repeat, w,
    )
    timings["sum_circle_optimal grp (bkgann)"] = time_call(
        "sum_circle_optimal grp (bkgann)",
        lambda: sep.sum_circle_optimal(
            data, x, y, r, fwhm, err=args.noise,
            grouped=True, bkgann=bkgann, subpix=args.subpix,
        ),
        args.repeat, w,
    )

    print()
    timings["psf_fit (flux only)"] = time_call(
        "psf_fit (flux only)",
        lambda: sep.psf_fit(
            data, x, y, psf, var=args.noise**2, fit_positions=False,
        ),
        args.repeat, w,
    )
    timings["psf_fit"] = time_call(
        "psf_fit",
        lambda: sep.psf_fit(
            data, x, y, psf, var=args.noise**2, maxiter=args.maxiter,
        ),
        args.repeat, w,
    )
    timings["psf_fit grouped"] = time_call(
        "psf_fit grouped",
        lambda: sep.psf_fit(
            data, x, y, psf, var=args.noise**2,
            grouped=True, maxiter=args.maxiter,
        ),
        args.repeat, w,
    )

    # Throughput summary
    print()
    print(f"  {'Throughput summary':{w}s}    obj/s")
    print(f"  {'-' * (w + 10)}")
    for label in ["sum_circle", "sum_circle_optimal",
                   "sum_circle_optimal grouped",
                   "psf_fit (flux only)", "psf_fit", "psf_fit grouped"]:
        t = timings[label]
        rate = n / t if t > 0 else float("inf")
        print(f"  {label:{w}s}  {rate:10.0f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
