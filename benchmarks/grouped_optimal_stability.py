#!/usr/bin/env python3
"""Synthetic stability checks for grouped optimal-extraction photometry."""

from __future__ import annotations

import argparse
import math
from math import erf

import numpy as np

import sep_x as sep


def erf_vec(values: np.ndarray) -> np.ndarray:
    if hasattr(np, "erf"):
        return np.erf(values)
    return np.vectorize(erf)(values)


def add_gaussian_stamp(
    image: np.ndarray, x: float, y: float, flux: float, sigma: float
) -> None:
    ny, nx = image.shape
    radius = int(np.ceil(8.0 * sigma))
    xmin = max(0, int(np.floor(x - radius)))
    xmax = min(nx - 1, int(np.floor(x + radius)))
    ymin = max(0, int(np.floor(y - radius)))
    ymax = min(ny - 1, int(np.floor(y + radius)))

    xs = np.arange(xmin, xmax + 1, dtype=np.float64)
    ys = np.arange(ymin, ymax + 1, dtype=np.float64)
    root2sig = math.sqrt(2.0) * sigma
    ex = erf_vec((xs + 0.5 - x) / root2sig) - erf_vec((xs - 0.5 - x) / root2sig)
    ey = erf_vec((ys + 0.5 - y) / root2sig) - erf_vec((ys - 0.5 - y) / root2sig)
    image[ymin : ymax + 1, xmin : xmax + 1] += 0.25 * flux * ey[:, None] * ex[None, :]


def make_scene(
    shape: tuple[int, int], x: np.ndarray, y: np.ndarray, flux: np.ndarray, fwhm: float
) -> np.ndarray:
    sigma = fwhm / 2.354820045
    image = np.zeros(shape, dtype=np.float64)
    for xi, yi, fi in zip(x, y, flux):
        add_gaussian_stamp(image, float(xi), float(yi), float(fi), sigma)
    return image


def rel_stats(measured: np.ndarray, truth: np.ndarray) -> tuple[float, float, float]:
    rel = (measured - truth) / truth
    return float(np.median(rel)), float(np.max(np.abs(rel))), float(np.sqrt(np.mean(rel * rel)))


def pair_sweep(args: argparse.Namespace) -> None:
    fwhm = args.fwhm
    r = args.r_factor * fwhm
    threshold = 2.0 * r / fwhm
    seps = np.array([float(x) for x in args.seps.split(",") if x.strip()])
    ratios = np.array([float(x) for x in args.ratios.split(",") if x.strip()])
    angles = np.linspace(0.0, np.pi, args.angles, endpoint=False)
    phases = np.linspace(0.0, 0.75, args.phases)
    size = args.size
    center = 0.5 * (size - 1)

    print("Pair sweep")
    print(
        f"- fwhm={fwhm:.3f}, r={r:.3f} ({args.r_factor:.2f}*FWHM), "
        f"group transition={threshold:.3f} FWHM"
    )
    print(
        "sep/FWHM ratio n grouped max|err| rms|err| max|grp-ungrp| "
        "max|left-right jump| flags"
    )

    rows: list[tuple[float, float, float, float, float, int]] = []
    for ratio in ratios:
        prev_max = None
        for sep_f in seps:
            errs = []
            rmses = []
            diffs = []
            flag_or = 0
            for angle in angles:
                for phase in phases:
                    sep_pix = sep_f * fwhm
                    dx = 0.5 * sep_pix * math.cos(angle)
                    dy = 0.5 * sep_pix * math.sin(angle)
                    x = np.array([center + phase - dx, center + phase + dx])
                    y = np.array([center - phase - dy, center - phase + dy])
                    truth = np.array([1000.0, 1000.0 * ratio])
                    data = make_scene((size, size), x, y, truth, fwhm)

                    flux_u, _, flag_u = sep.sum_circle_optimal(
                        data, x, y, r, fwhm, err=1.0, subpix=args.subpix
                    )
                    flux_g, _, flag_g = sep.sum_circle_optimal(
                        data,
                        x,
                        y,
                        r,
                        fwhm,
                        err=1.0,
                        grouped=True,
                        group_radius_factor=args.group_radius_factor,
                        group_halo_factor=args.group_halo_factor,
                        subpix=args.subpix,
                    )
                    _, maxerr, rmse = rel_stats(flux_g, truth)
                    errs.append(maxerr)
                    rmses.append(rmse)
                    diffs.append(float(np.max(np.abs(flux_g - flux_u) / truth)))
                    flag_or |= int(np.bitwise_or.reduce(flag_u | flag_g))

            maxerr = float(np.max(errs))
            rmse = float(np.max(rmses))
            maxdiff = float(np.max(diffs))
            jump = 0.0 if prev_max is None else abs(maxerr - prev_max)
            rows.append((sep_f, ratio, maxerr, rmse, maxdiff, flag_or))
            print(
                f"{sep_f:7.3f} {ratio:5.2f} {len(errs):3d} "
                f"{maxerr:11.3e} {rmse:9.3e} {maxdiff:14.3e} "
                f"{jump:16.3e} {flag_or:5d}"
            )
            prev_max = maxerr

    near = [row for row in rows if abs(row[0] - threshold) <= 0.15]
    if near:
        worst_near = max(row[2] for row in near)
        print(f"- worst max|err| within 0.15 FWHM of transition: {worst_near:.3e}")


def chain_sweep(args: argparse.Namespace) -> None:
    fwhm = args.fwhm
    r = args.r_factor * fwhm
    threshold = 2.0 * r / fwhm
    spacings = np.array([float(x) for x in args.chain_spacings.split(",") if x.strip()])
    nsrc = args.chain_nsrc
    y0 = 0.5 * (args.chain_height - 1)
    rng = np.random.default_rng(args.seed)
    base_truth = rng.uniform(600.0, 1400.0, nsrc)
    offsets = np.array([float(x) for x in args.chain_offsets.split(",") if x.strip()])

    print("")
    print("Long-chain localized-solver sweep")
    print(
        f"- nsrc={nsrc}, transition={threshold:.3f} FWHM, "
        f"offsets={','.join(f'{v:.2f}' for v in offsets)} pix"
    )
    print(
        "spacing/FWHM regime max|err| rms|err| max|reverse diff| "
        "max|offset diff| flags"
    )

    for spacing_f in spacings:
        spacing = spacing_f * fwhm
        width = int(np.ceil(2.0 * args.margin + spacing * (nsrc - 1)))
        x_base = args.margin + np.arange(nsrc, dtype=np.float64) * spacing
        y = np.full(nsrc, y0)
        best_flux = None
        offset_diffs = []
        max_errs = []
        rmses = []
        rev_diffs = []
        flag_or = 0

        for offset in offsets:
            x = x_base + offset
            data = make_scene((args.chain_height, width + int(np.ceil(offsets.max())) + 4), x, y, base_truth, fwhm)
            flux_g, _, flag_g = sep.sum_circle_optimal(
                data,
                x,
                y,
                r,
                fwhm,
                err=1.0,
                grouped=True,
                group_radius_factor=args.group_radius_factor,
                group_halo_factor=args.group_halo_factor,
                subpix=args.subpix,
            )
            order = np.arange(nsrc)[::-1]
            flux_rev, _, flag_rev = sep.sum_circle_optimal(
                data,
                x[order],
                y[order],
                r,
                fwhm,
                err=1.0,
                grouped=True,
                group_radius_factor=args.group_radius_factor,
                group_halo_factor=args.group_halo_factor,
                subpix=args.subpix,
            )
            flux_rev_back = np.empty_like(flux_rev)
            flux_rev_back[order] = flux_rev
            _, maxerr, rmse = rel_stats(flux_g, base_truth)
            max_errs.append(maxerr)
            rmses.append(rmse)
            rev_diffs.append(float(np.max(np.abs(flux_g - flux_rev_back) / base_truth)))
            flag_or |= int(np.bitwise_or.reduce(flag_g | flag_rev))
            if best_flux is None:
                best_flux = flux_g
            else:
                offset_diffs.append(float(np.max(np.abs(flux_g - best_flux) / base_truth)))

        regime = "localized" if spacing_f <= threshold else "isolated"
        max_offset = max(offset_diffs) if offset_diffs else 0.0
        print(
            f"{spacing_f:11.3f} {regime:9s} {max(max_errs):9.3e} "
            f"{max(rmses):9.3e} {max(rev_diffs):17.3e} "
            f"{max_offset:16.3e} {flag_or:5d}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=160)
    parser.add_argument("--fwhm", type=float, default=2.5)
    parser.add_argument("--r-factor", type=float, default=2.0)
    parser.add_argument(
        "--seps",
        default="0.8,1.0,1.25,1.5,2.0,2.5,3.0,3.5,3.8,3.95,4.0,4.05,4.2,4.5",
    )
    parser.add_argument("--ratios", default="1.0,0.3,0.1")
    parser.add_argument("--angles", type=int, default=8)
    parser.add_argument("--phases", type=int, default=4)
    parser.add_argument("--subpix", type=int, default=0)
    parser.add_argument("--group-radius-factor", type=float, default=1.0)
    parser.add_argument("--group-halo-factor", type=float, default=1.2)
    parser.add_argument("--chain-nsrc", type=int, default=48)
    parser.add_argument("--chain-height", type=int, default=96)
    parser.add_argument("--chain-spacings", default="3.6,3.8,3.95,4.0,4.05,4.2")
    parser.add_argument("--chain-offsets", default="0.0,1.3,2.7,4.1,5.5")
    parser.add_argument("--margin", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    pair_sweep(args)
    chain_sweep(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
