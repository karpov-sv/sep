#!/usr/bin/env python3
"""Compare classic, matched-filter, and PSF-matched source detection."""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import sep
except ImportError as exc:
    raise SystemExit("sep is required; build the extension first.") from exc


@dataclass
class DetectionResult:
    name: str
    catalog: np.ndarray
    elapsed: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare SEP classic extraction, fixed-kernel matched filtering, "
            "PSF-SNR extraction, and optionally crowdsource on one FITS image."
        )
    )
    parser.add_argument("image", help="Input FITS image.")
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument("--fwhm", type=float, default=3.7)
    parser.add_argument("--oversampling", type=int, default=4)
    parser.add_argument("--classic-minarea", type=int, default=5)
    parser.add_argument("--psf-minarea", type=int, default=1)
    parser.add_argument("--psf-norm-minarea", type=int, default=3)
    parser.add_argument("--match-radius", type=float, default=1.0)
    parser.add_argument("--snr-bw", type=int, default=64)
    parser.add_argument("--snr-bh", type=int, default=64)
    parser.add_argument("--no-normalized", action="store_true")
    parser.add_argument("--no-raw-psf", action="store_true")
    parser.add_argument("--no-peaks", action="store_true")
    parser.add_argument("--no-fixed-kernel", action="store_true")
    parser.add_argument("--crop", type=str, default="")
    parser.add_argument("--save-prefix", type=str, default="")
    parser.add_argument(
        "--truth-catalog",
        type=str,
        default="",
        help="Parquet catalog with ra/dec columns for truth matching.",
    )
    parser.add_argument("--truth-radius", type=float, default=2.0)
    parser.add_argument(
        "--truth-good-only",
        action="store_true",
        help="Use only rows with good > 0 from the truth catalog.",
    )
    parser.add_argument(
        "--peaks-fit",
        action="store_true",
        help="Fit PSF fluxes at psf_peaks positions and keep fitted-S/N detections.",
    )
    parser.add_argument("--peaks-fit-snr", type=float, default=3.0)
    parser.add_argument(
        "--peaks-fit-snr-grid",
        type=str,
        default="",
        help="Comma-separated fitted-S/N cuts to evaluate from one PSF-fit pass.",
    )
    parser.add_argument(
        "--peaks-fit-flagged",
        action="store_true",
        help="Keep flagged PSF fits in the peaks-fit catalog.",
    )
    parser.add_argument("--crowdsource", action="store_true")
    parser.add_argument(
        "--crowdsource-path",
        default=str(Path.home() / "tmp" / "crowdsource"),
        help="Local crowdsource checkout to import when --crowdsource is set.",
    )
    parser.add_argument("--crowdsource-maxiter", type=int, default=4)
    parser.add_argument("--crowdsource-miniter", type=int, default=2)
    parser.add_argument("--crowdsource-maxstars", type=int, default=60000)
    parser.add_argument("--crowdsource-tiles", type=int, default=4)
    return parser.parse_args()


def read_fits(path: str) -> np.ndarray:
    try:
        from fitsio import read

        data = read(path)
    except ImportError:
        try:
            from astropy.io import fits
        except ImportError as exc:
            raise SystemExit("Install fitsio or astropy to read FITS files.") from exc
        data = fits.getdata(path)

    data = np.asarray(data, dtype=np.float64)
    if not np.all(np.isfinite(data)):
        data = data.copy()
        data[~np.isfinite(data)] = np.nanmedian(data)
    return data


def parse_crop(crop: str, shape: tuple[int, int]) -> tuple[slice, slice]:
    if not crop:
        return slice(None), slice(None)

    parts = [int(value) for value in crop.replace(",", ":").split(":")]
    if len(parts) != 4:
        raise SystemExit("--crop must be y0:y1:x0:x1")

    y0, y1, x0, x1 = parts
    ny, nx = shape
    if not (0 <= y0 < y1 <= ny and 0 <= x0 < x1 <= nx):
        raise SystemExit(f"--crop is outside image shape {shape}")
    return slice(y0, y1), slice(x0, x1)


def read_truth_catalog(
    path: str,
    image_path: str,
    image_shape: tuple[int, int],
    yslice: slice,
    xslice: slice,
    good_only: bool,
) -> np.ndarray:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("Install pyarrow to read parquet truth catalogs.") from exc
    try:
        from astropy.io import fits
        from astropy.wcs import WCS
    except ImportError as exc:
        raise SystemExit("Install astropy to project sky truth catalogs.") from exc

    columns = ["ra", "dec"]
    table_meta = pq.read_schema(path)
    if good_only and "good" in table_meta.names:
        columns.append("good")
    table = pq.read_table(path, columns=columns).to_pydict()

    header = fits.getheader(image_path)
    wcs = WCS(header)
    ra = np.asarray(table["ra"], dtype=np.float64)
    dec = np.asarray(table["dec"], dtype=np.float64)
    x, y = wcs.all_world2pix(ra, dec, 0)

    y0 = 0 if yslice.start is None else yslice.start
    y1 = image_shape[0] if yslice.stop is None else yslice.stop
    x0 = 0 if xslice.start is None else xslice.start
    x1 = image_shape[1] if xslice.stop is None else xslice.stop

    inside = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= x0)
        & (x < x1)
        & (y >= y0)
        & (y < y1)
    )
    if good_only and "good" in table:
        inside &= np.asarray(table["good"]) > 0

    truth = np.column_stack([x[inside] - x0, y[inside] - y0]).astype(np.float64)
    return truth


def psf_native_kernel(psf: "sep.PSF") -> np.ndarray:
    kernel = np.zeros((psf.stamp_height, psf.stamp_width), dtype=np.float64)
    sep.model_psf(
        kernel,
        [psf.stamp_width // 2],
        [psf.stamp_height // 2],
        [1.0],
        psf,
    )
    return kernel


def run_timed(name: str, func) -> DetectionResult:
    start = time.perf_counter()
    catalog = func()
    elapsed = time.perf_counter() - start
    return DetectionResult(name, catalog, elapsed)


def parse_float_grid(value: str) -> list[float]:
    if not value:
        return []
    return [float(item) for item in value.replace(":", ",").split(",") if item]


def xy_from_catalog(catalog: np.ndarray) -> np.ndarray:
    if len(catalog) == 0:
        return np.empty((0, 2), dtype=np.float64)
    return np.column_stack([catalog["x"], catalog["y"]]).astype(np.float64)


def count_matches(
    query: np.ndarray, reference: np.ndarray, radius: float
) -> tuple[int, float, float]:
    if len(query) == 0 or len(reference) == 0:
        return 0, np.nan, np.nan

    try:
        from scipy.spatial import cKDTree

        dist, _ = cKDTree(reference).query(query, distance_upper_bound=radius)
        finite = np.isfinite(dist)
        nearest = cKDTree(reference).query(query, k=1)[0]
        return int(finite.sum()), float(np.median(nearest)), float(np.percentile(nearest, 95))
    except ImportError:
        return count_matches_grid(query, reference, radius)


def count_matches_grid(
    query: np.ndarray, reference: np.ndarray, radius: float
) -> tuple[int, float, float]:
    cell = max(radius, 1.0)
    grid: dict[tuple[int, int], list[int]] = {}
    for idx, (x, y) in enumerate(reference):
        key = int(np.floor(x / cell)), int(np.floor(y / cell))
        grid.setdefault(key, []).append(idx)

    nearest = np.full(len(query), np.inf, dtype=np.float64)
    r2 = radius * radius
    for i, (x, y) in enumerate(query):
        cx = int(np.floor(x / cell))
        cy = int(np.floor(y / cell))
        best = np.inf
        for gx in range(cx - 1, cx + 2):
            for gy in range(cy - 1, cy + 2):
                for idx in grid.get((gx, gy), []):
                    dx = x - reference[idx, 0]
                    dy = y - reference[idx, 1]
                    dist2 = dx * dx + dy * dy
                    if dist2 < best:
                        best = dist2
        if best < np.inf:
            nearest[i] = np.sqrt(best)

    matched = int(np.sum(nearest <= radius))
    return matched, float(np.median(nearest)), float(np.percentile(nearest, 95))


def print_result(
    result: DetectionResult,
    references: dict[str, np.ndarray],
    radius: float,
) -> None:
    cat = result.catalog
    line = f"{result.name:22s} {len(cat):8d} {result.elapsed:8.3f}s"
    for ref_name, ref_xy in references.items():
        matched, med, p95 = count_matches(xy_from_catalog(cat), ref_xy, radius)
        line += f"  {ref_name}: {matched:8d} med={med:5.2f} p95={p95:5.2f}"
    if len(cat) and "flux" in cat.dtype.names:
        pct = np.percentile(cat["flux"], [5, 50, 95])
        line += f"  flux5/50/95={pct[0]:.1f}/{pct[1]:.1f}/{pct[2]:.1f}"
    print(line)


def shifted_match_baseline(
    points: np.ndarray,
    truth: np.ndarray,
    radius: float,
    shape: tuple[int, int],
) -> float:
    if len(points) == 0 or len(truth) == 0:
        return np.nan

    shifts = [(137, 0), (0, 137), (137, 137), (-137, 83), (211, -151)]
    values = []
    ny, nx = shape
    for dx, dy in shifts:
        shifted = points.copy()
        shifted[:, 0] = (shifted[:, 0] + dx) % nx
        shifted[:, 1] = (shifted[:, 1] + dy) % ny
        matched, _, _ = count_matches(shifted, truth, radius)
        values.append(matched / len(points))
    return float(np.mean(values))


def print_truth_report(
    results: list[DetectionResult],
    truth: np.ndarray,
    radius: float,
    shape: tuple[int, int],
) -> None:
    if len(truth) == 0:
        print("\ntruth catalog: no sources inside image/crop")
        return

    print("")
    print(f"truth catalog sources inside image/crop: {len(truth)}")
    print(f"truth match radius: {radius:g} px")
    print("method                    count  truth_match  frac  shifted")
    for result in results:
        points = xy_from_catalog(result.catalog)
        matched, med, p95 = count_matches(points, truth, radius)
        frac = matched / len(points) if len(points) else np.nan
        shifted = shifted_match_baseline(points, truth, radius, shape)
        print(
            f"{result.name:22s} {len(points):8d} {matched:11d} "
            f"{frac:5.3f} {shifted:7.3f}  med={med:5.2f} p95={p95:5.2f}"
        )


def peak_fit_dtype() -> np.dtype:
    return np.dtype(
        [
            ("x", "f8"),
            ("y", "f8"),
            ("flux", "f8"),
            ("fluxerr", "f8"),
            ("fit_snr", "f8"),
            ("peak_snr", "f8"),
            ("xpeak", "i8"),
            ("ypeak", "i8"),
            ("flag", "i2"),
        ]
    )


def fit_peak_candidates_all(
    data: np.ndarray,
    var: np.ndarray,
    psf: "sep.PSF",
    args: argparse.Namespace,
) -> np.ndarray:
    peaks = sep.psf_peaks(
        data,
        args.threshold,
        psf,
        var=var,
        min_distance=1.5,
    )
    if len(peaks) == 0:
        return np.empty(0, dtype=peak_fit_dtype())

    flux, fluxerr, xfit, yfit, flag, _, _ = sep.psf_fit(
        data,
        peaks["x"],
        peaks["y"],
        psf,
        var=var,
        fit_positions=True,
    )
    fit_snr = flux / np.maximum(fluxerr, 1.0e-30)
    catalog = np.empty(len(peaks), dtype=peak_fit_dtype())
    catalog["x"] = xfit
    catalog["y"] = yfit
    catalog["flux"] = flux
    catalog["fluxerr"] = fluxerr
    catalog["fit_snr"] = fit_snr
    catalog["peak_snr"] = peaks["snr"]
    catalog["xpeak"] = peaks["xpeak"]
    catalog["ypeak"] = peaks["ypeak"]
    catalog["flag"] = flag
    return catalog


def filter_peak_fit_catalog(
    catalog: np.ndarray,
    snr_cut: float,
    keep_flagged: bool,
) -> np.ndarray:
    keep = np.isfinite(catalog["fit_snr"]) & (catalog["fit_snr"] > snr_cut)
    if not keep_flagged:
        keep &= catalog["flag"] == 0
    return catalog[keep].copy()


def run_crowdsource(
    data: np.ndarray,
    rms: np.ndarray,
    args: argparse.Namespace,
) -> DetectionResult:
    path = Path(args.crowdsource_path).expanduser()
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

    try:
        crowdsource_base = importlib.import_module("crowdsource_base")
    except ImportError:
        crowdsource_base = importlib.import_module("crowdsource.crowdsource_base")
    crowdsource_psf = importlib.import_module("crowdsource.psf")

    weights = np.zeros_like(rms, dtype=np.float32)
    good = rms > 0.0
    weights[good] = 1.0 / rms[good].astype(np.float32)
    dq = np.zeros(data.shape, dtype=np.int64)
    psf_data = crowdsource_psf.gaussian_psf(
        args.fwhm, stampsz=19, deriv=False
    ).astype(np.float32)
    psf = crowdsource_psf.SimplePSF(psf_data)
    tiles = args.crowdsource_tiles

    def _run():
        result = crowdsource_base.fit_im(
            data.astype(np.float32, copy=False),
            psf,
            weights=weights,
            dq=dq,
            psfderiv=True,
            miniter=args.crowdsource_miniter,
            maxiter=args.crowdsource_maxiter,
            maxstars=args.crowdsource_maxstars,
            threshold=args.threshold,
            ntilex=tiles,
            ntiley=tiles,
            verbose=False,
        )
        if isinstance(result, dict):
            stars = result["stars"]
        elif isinstance(result, tuple):
            stars = result[0]
        else:
            stars = result
        catalog = np.empty(
            len(stars),
            dtype=[("x", "f8"), ("y", "f8"), ("flux", "f8")],
        )
        catalog["x"] = stars["y"]
        catalog["y"] = stars["x"]
        catalog["flux"] = stars["flux"]
        return catalog

    return run_timed("crowdsource", _run)


def save_catalogs(prefix: str, results: list[DetectionResult]) -> None:
    arrays = {result.name.replace(" ", "_"): result.catalog for result in results}
    np.savez(prefix, **arrays)


def main() -> None:
    args = parse_args()

    data = read_fits(args.image)
    full_shape = data.shape
    yslice, xslice = parse_crop(args.crop, full_shape)
    data = data[yslice, xslice].copy()

    print(f"image shape: {data.shape}")
    print(f"threshold: {args.threshold:g}")
    print(f"fwhm: {args.fwhm:g}")

    start = time.perf_counter()
    bkg = sep.Background(data)
    data_sub = data - bkg.back(dtype=np.float64)
    rms = bkg.rms(dtype=np.float64)
    var = rms * rms
    print(
        f"background: {time.perf_counter() - start:.3f}s "
        f"globalrms={bkg.globalrms:.4g}"
    )

    psf = sep.PSF.from_gaussian(args.fwhm, oversampling=args.oversampling)
    kernel = psf_native_kernel(psf)

    results: list[DetectionResult] = []
    results.append(
        run_timed(
            "classic",
            lambda: sep.extract(
                data_sub,
                args.threshold,
                err=rms,
                minarea=args.classic_minarea,
            ),
        )
    )

    if not args.no_fixed_kernel:
        results.append(
            run_timed(
                "fixed matched",
                lambda: sep.extract(
                    data_sub,
                    args.threshold,
                    err=rms,
                    minarea=args.psf_minarea,
                    filter_kernel=kernel,
                    filter_type="matched",
                ),
            )
        )

    if not args.no_raw_psf:
        results.append(
            run_timed(
                "psf raw",
                lambda: sep.psf_extract(
                    data_sub,
                    args.threshold,
                    psf,
                    var=var,
                    minarea=args.psf_minarea,
                ),
            )
        )

    if not args.no_peaks:
        results.append(
            run_timed(
                "psf peaks",
                lambda: sep.psf_peaks(
                    data_sub,
                    args.threshold,
                    psf,
                    var=var,
                    min_distance=1.5,
                ),
            )
        )

    if args.peaks_fit:
        fit_result = run_timed(
            "psf peaks fit all",
            lambda: fit_peak_candidates_all(data_sub, var, psf, args),
        )
        snr_grid = parse_float_grid(args.peaks_fit_snr_grid)
        if not snr_grid:
            snr_grid = [args.peaks_fit_snr]
        for snr_cut in snr_grid:
            catalog = filter_peak_fit_catalog(
                fit_result.catalog, snr_cut, args.peaks_fit_flagged
            )
            results.append(
                DetectionResult(
                    f"psf peaks fit>{snr_cut:g}",
                    catalog,
                    fit_result.elapsed,
                )
            )

    if not args.no_normalized:
        results.append(
            run_timed(
                "psf normalized",
                lambda: sep.psf_extract(
                    data_sub,
                    args.threshold,
                    psf,
                    var=var,
                    minarea=args.psf_norm_minarea,
                    normalize_snr=True,
                    snr_bw=args.snr_bw,
                    snr_bh=args.snr_bh,
                ),
            )
        )

    if args.crowdsource:
        results.append(run_crowdsource(data, rms, args))

    references = {"classic": xy_from_catalog(results[0].catalog)}
    if args.crowdsource:
        references["crowdsource"] = xy_from_catalog(results[-1].catalog)

    print("")
    print("method                    count     time   matches")
    for result in results:
        print_result(result, references, args.match_radius)

    if args.truth_catalog:
        truth = read_truth_catalog(
            args.truth_catalog,
            args.image,
            full_shape,
            yslice,
            xslice,
            args.truth_good_only,
        )
        print_truth_report(results, truth, args.truth_radius, data.shape)

    if args.save_prefix:
        save_catalogs(args.save_prefix, results)
        print(f"\nwrote {args.save_prefix}.npz")


if __name__ == "__main__":
    main()
