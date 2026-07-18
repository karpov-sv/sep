#!/usr/bin/env python
"""Verbose comparison of grouped vs ungrouped optimal extraction."""

from __future__ import annotations

import argparse
import math

import numpy as np
import sep_x as sep


def gaussian_pixel_integral(dx, dy, sigma):
    inv = 1.0 / (math.sqrt(2.0) * sigma)
    ex = np.vectorize(math.erf)((dx + 0.5) * inv) - np.vectorize(math.erf)(
        (dx - 0.5) * inv
    )
    ey = np.vectorize(math.erf)((dy + 0.5) * inv) - np.vectorize(math.erf)(
        (dy - 0.5) * inv
    )
    return 0.25 * ex * ey


def gaussian_scene(shape, x, y, fwhm, flux):
    yy, xx = np.indices(shape)
    sigma = fwhm / 2.354820045
    image = np.zeros(shape, dtype=float)
    for xi, yi, fi in zip(x, y, flux):
        image += fi * gaussian_pixel_integral(xx - xi, yy - yi, sigma)
    return image


def min_sep(x, y):
    n = len(x)
    if n < 2:
        return float("inf")
    dmin2 = float("inf")
    for i in range(n - 1):
        dx = x[i] - x[i + 1 :]
        dy = y[i] - y[i + 1 :]
        d2 = dx * dx + dy * dy
        if d2.size:
            dmin2 = min(dmin2, float(d2.min()))
    return math.sqrt(dmin2)


def sample_positions(nobj, size, min_sep_pix, margin, rng):
    xs = []
    ys = []
    max_attempts = 200000
    attempts = 0
    while len(xs) < nobj and attempts < max_attempts:
        x = rng.uniform(margin, size - margin)
        y = rng.uniform(margin, size - margin)
        ok = True
        for px, py in zip(xs, ys):
            dx = x - px
            dy = y - py
            if dx * dx + dy * dy < min_sep_pix * min_sep_pix:
                ok = False
                break
        if ok:
            xs.append(x)
            ys.append(y)
        attempts += 1
    if len(xs) < nobj:
        raise RuntimeError("Failed to place all objects with requested separation.")
    return np.array(xs), np.array(ys)


def summarize(label, rel_err):
    return (
        f"{label:12s}  min={rel_err.min():+.3e}  "
        f"med={np.median(rel_err):+.3e}  max={rel_err.max():+.3e}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--nobj", type=int, default=6)
    ap.add_argument("--fwhm", type=float, default=3.0)
    ap.add_argument("--r", type=float, default=6.0)
    ap.add_argument("--min-sep", type=float, default=18.0)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--subpix", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    margin = args.r + 2.0

    print("Grouped vs ungrouped optimal extraction (wide separation)")
    print(
        f"- size={args.size} nobj={args.nobj} fwhm={args.fwhm} r={args.r} "
        f"min-sep>={args.min_sep} noise={args.noise} subpix={args.subpix}"
    )

    for t in range(args.trials):
        x, y = sample_positions(args.nobj, args.size, args.min_sep, margin, rng)
        flux_true = rng.uniform(500.0, 2000.0, size=args.nobj)
        data = gaussian_scene((args.size, args.size), x, y, args.fwhm, flux_true)
        if args.noise > 0.0:
            data += rng.normal(scale=args.noise, size=data.shape)

        flux, _, flag = sep.sum_circle_optimal(
            data, x, y, args.r, args.fwhm, subpix=args.subpix
        )
        flux_g, _, flag_g = sep.sum_circle_optimal(
            data, x, y, args.r, args.fwhm, grouped=True, subpix=args.subpix
        )

        rel = (flux - flux_true) / flux_true
        rel_g = (flux_g - flux_true) / flux_true
        diff = flux_g - flux

        print(f"\nTrial {t + 1}: min_sep={min_sep(x, y):.2f} pix")
        print(summarize("ungrouped", rel))
        print(summarize("grouped", rel_g))
        print(f"{'diff':12s}  max|grouped-ungrouped|={np.max(np.abs(diff)):.3e}")
        if np.any(flag) or np.any(flag_g):
            print(f"Flags (ungrouped): {np.unique(flag)}")
            print(f"Flags (grouped): {np.unique(flag_g)}")

        print("\nidx    x      y     true      ungrouped   grouped     rel_u     rel_g")
        for i in range(args.nobj):
            print(
                f"{i:2d}  {x[i]:6.2f}  {y[i]:6.2f}  {flux_true[i]:8.2f}  "
                f"{flux[i]:10.2f}  {flux_g[i]:10.2f}  "
                f"{rel[i]:+8.3e}  {rel_g[i]:+8.3e}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
