#!/usr/bin/env python3
"""Stress-test grouped PSF fitting with noisy positions from a deeper catalog.

The synthetic setup models a catalog measured on an image deeper than the test
image. Every supplied position corresponds to a real source, but a configurable
fraction of those sources fall below the shallower image's nominal detection
threshold. The script reports how grouped PSF fitting behaves for:

- detectable sources in the shallower image
- sub-threshold catalog positions that should not grow large spurious fits
- noisy initial positions whose uncertainty is set by the deeper catalog S/N
"""

from __future__ import annotations

import argparse
import inspect
import math
import time

import numpy as np

try:
    import sep_x as sep
except ImportError as exc:
    raise SystemExit("sep is required; build the extension first.") from exc


METHOD_LABELS = {
    "ungrouped": "ungrouped",
    "grouped": "grouped",
    "grouped_damped": "grouped+damped",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check grouped PSF fitting with noisy input positions from a "
            "catalog deeper than the fitted image."
        )
    )
    parser.add_argument("--size", type=int, default=256, help="Image size.")
    parser.add_argument(
        "--nsources",
        type=int,
        default=180,
        help="Number of catalog sources per trial.",
    )
    parser.add_argument(
        "--trials", type=int, default=5, help="Number of Monte Carlo trials."
    )
    parser.add_argument("--fwhm", type=float, default=3.0, help="PSF FWHM.")
    parser.add_argument(
        "--noise", type=float, default=10.0, help="Gaussian noise sigma."
    )
    parser.add_argument(
        "--catalog-depth",
        type=float,
        default=4.0,
        help="Relative depth of the input catalog image (variance ratio).",
    )
    parser.add_argument(
        "--catalog-sn",
        type=float,
        default=3.5,
        help="Minimum matched-filter S/N required for a source to appear "
        "in the deeper catalog.",
    )
    parser.add_argument(
        "--detect-sn",
        type=float,
        default=5.0,
        help="Matched-filter S/N considered detectable in the shallower image.",
    )
    parser.add_argument(
        "--flux-min",
        type=float,
        default=30.0,
        help="Minimum source flux for trial generation.",
    )
    parser.add_argument(
        "--flux-max",
        type=float,
        default=5000.0,
        help="Maximum source flux for trial generation.",
    )
    parser.add_argument(
        "--cluster-prob",
        type=float,
        default=0.65,
        help="Probability of placing a source near an existing source.",
    )
    parser.add_argument(
        "--cluster-radius",
        type=float,
        default=1.6,
        help="Cluster placement radius in FWHM units.",
    )
    parser.add_argument(
        "--min-sep",
        type=float,
        default=0.55,
        help="Minimum true separation in FWHM units.",
    )
    parser.add_argument(
        "--init-floor",
        type=float,
        default=0.02,
        help="Minimum initial position noise sigma in pixels.",
    )
    parser.add_argument(
        "--init-cap",
        type=float,
        default=0.85,
        help="Maximum initial position noise sigma in pixels.",
    )
    parser.add_argument(
        "--init-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to sigma_psf / deep_SNR for position noise.",
    )
    parser.add_argument(
        "--group-factor",
        type=float,
        default=2.0,
        help="group_factor passed to sep.psf_fit(grouped=True).",
    )
    parser.add_argument(
        "--fit-radius",
        type=float,
        default=0.0,
        help="fit_radius passed to sep.psf_fit.",
    )
    parser.add_argument(
        "--damp-snthresh",
        type=float,
        default=0.0,
        help="damp_snthresh passed to grouped+damped runs when supported.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="ungrouped,grouped",
        help="Comma-separated methods: ungrouped, grouped, grouped_damped.",
    )
    parser.add_argument(
        "--report-trials",
        action="store_true",
        help="Print one-line per-trial summaries.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the selected target method fails thresholds.",
    )
    parser.add_argument(
        "--check-method",
        type=str,
        default="",
        help="Method to check. Default: grouped_damped if present, else grouped.",
    )
    parser.add_argument(
        "--max-detectable-pos-p90",
        type=float,
        default=0.5,
        help="Threshold for --check: detectable-source p90 position error.",
    )
    parser.add_argument(
        "--max-detectable-flux-rmse",
        type=float,
        default=0.25,
        help="Threshold for --check: detectable-source fractional flux RMSE.",
    )
    parser.add_argument(
        "--max-subthreshold-promoted",
        type=float,
        default=0.10,
        help="Threshold for --check: sub-threshold fraction with fitted S/N > 3.",
    )
    parser.add_argument(
        "--max-subthreshold-drift-p90",
        type=float,
        default=0.75,
        help="Threshold for --check: sub-threshold p90 drift from input position.",
    )
    parser.add_argument("--seed", type=int, default=12345, help="RNG seed.")
    return parser.parse_args()


def normalize_methods(args: argparse.Namespace) -> list[str]:
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    valid = set(METHOD_LABELS)
    bad = [name for name in methods if name not in valid]
    if bad:
        raise SystemExit(f"Unknown method(s): {', '.join(bad)}")

    if args.damp_snthresh > 0.0 and "grouped_damped" not in methods:
        methods.append("grouped_damped")

    if "grouped_damped" in methods:
        params = inspect.signature(sep.psf_fit).parameters
        if "damp_snthresh" not in params:
            raise SystemExit(
                "grouped_damped requested, but this sep build does not expose "
                "damp_snthresh in sep.psf_fit."
            )
        if args.damp_snthresh <= 0.0:
            raise SystemExit(
                "grouped_damped requested but --damp-snthresh is not positive."
            )

    return methods


def render_unit_psf_norm(psf: "sep.PSF", fit_radius: float) -> float:
    size = max(psf.stamp_width, psf.stamp_height) + 8
    arr = np.zeros((size, size), dtype=np.float32)
    center = 0.5 * (size - 1)
    sep.model_psf(arr, center, center, 1.0, psf)
    if fit_radius > 0.0:
        yy, xx = np.indices(arr.shape, dtype=np.float64)
        mask = (xx - center) ** 2 + (yy - center) ** 2 <= fit_radius * fit_radius
        arr = np.where(mask, arr, 0.0)
    return float(np.sqrt(np.sum(arr.astype(np.float64) ** 2)))


def sample_catalog_fluxes(
    rng: np.random.Generator,
    nsources: int,
    flux_min: float,
    flux_max: float,
    deep_noise: float,
    psf_l2: float,
    catalog_sn: float,
) -> np.ndarray:
    if flux_min <= 0.0 or flux_max <= flux_min:
        raise SystemExit("Require 0 < flux_min < flux_max.")

    out: list[np.ndarray] = []
    nacc = 0
    while nacc < nsources:
        batch = max(128, 2 * (nsources - nacc))
        flux = 10.0 ** rng.uniform(
            np.log10(flux_min), np.log10(flux_max), size=batch
        )
        deep_snr = flux * psf_l2 / deep_noise
        keep = flux[deep_snr >= catalog_sn]
        if keep.size:
            out.append(keep)
            nacc += keep.size
    return np.concatenate(out)[:nsources]


def place_sources(
    rng: np.random.Generator,
    nsources: int,
    size: int,
    margin: float,
    min_sep: float,
    cluster_prob: float,
    cluster_radius: float,
    max_tries: int = 20000,
) -> np.ndarray:
    positions: list[tuple[float, float]] = []
    for _ in range(nsources):
        for _ in range(max_tries):
            if positions and rng.random() < cluster_prob:
                ax, ay = positions[rng.integers(len(positions))]
                radius = rng.uniform(min_sep, cluster_radius)
                angle = rng.uniform(0.0, 2.0 * math.pi)
                x = ax + radius * math.cos(angle)
                y = ay + radius * math.sin(angle)
            else:
                x = rng.uniform(margin, size - margin)
                y = rng.uniform(margin, size - margin)

            if x < margin or x > size - margin or y < margin or y > size - margin:
                continue

            if not positions:
                positions.append((x, y))
                break

            dx = np.array([x - px for px, _ in positions], dtype=np.float64)
            dy = np.array([y - py for _, py in positions], dtype=np.float64)
            if np.all(dx * dx + dy * dy >= min_sep * min_sep):
                positions.append((x, y))
                break
        else:
            raise SystemExit(
                "Failed to place sources. Increase --size or reduce source density."
            )

    return np.array(positions, dtype=np.float64)


def nearest_neighbor_dist(positions: np.ndarray) -> np.ndarray:
    npos = positions.shape[0]
    out = np.full(npos, np.inf, dtype=np.float64)
    for i in range(npos):
        dx = positions[i, 0] - positions[:, 0]
        dy = positions[i, 1] - positions[:, 1]
        dist2 = dx * dx + dy * dy
        dist2[i] = np.inf
        out[i] = math.sqrt(float(np.min(dist2)))
    return out


def simulate_trial(
    rng: np.random.Generator,
    args: argparse.Namespace,
    psf: "sep.PSF",
    psf_l2: float,
) -> dict[str, np.ndarray]:
    sigma_psf = args.fwhm / 2.354820045
    deep_noise = args.noise / math.sqrt(args.catalog_depth)
    min_sep = args.min_sep * args.fwhm
    cluster_radius = args.cluster_radius * args.fwhm
    margin = max(4.0 * args.fwhm, cluster_radius + 2.0 * sigma_psf + 2.0)

    flux_true = sample_catalog_fluxes(
        rng,
        args.nsources,
        args.flux_min,
        args.flux_max,
        deep_noise,
        psf_l2,
        args.catalog_sn,
    )
    pos_true = place_sources(
        rng,
        args.nsources,
        args.size,
        margin,
        min_sep,
        args.cluster_prob,
        cluster_radius,
    )

    shallow_snr = flux_true * psf_l2 / args.noise
    deep_snr = flux_true * psf_l2 / deep_noise
    detectable = shallow_snr >= args.detect_sn

    pos_sigma = args.init_scale * sigma_psf / np.maximum(deep_snr, 1.0e-6)
    pos_sigma = np.clip(pos_sigma, args.init_floor, args.init_cap)

    x_true = pos_true[:, 0]
    y_true = pos_true[:, 1]
    x_init = np.clip(rng.normal(x_true, pos_sigma), 0.0, args.size - 1.0)
    y_init = np.clip(rng.normal(y_true, pos_sigma), 0.0, args.size - 1.0)

    data = np.zeros((args.size, args.size), dtype=np.float32)
    sep.model_psf(data, x_true, y_true, flux_true, psf)
    if args.noise > 0.0:
        data += rng.normal(0.0, args.noise, size=data.shape).astype(np.float32)

    return {
        "data": data,
        "x_true": x_true,
        "y_true": y_true,
        "x_init": x_init,
        "y_init": y_init,
        "flux_true": flux_true,
        "deep_snr": deep_snr,
        "shallow_snr": shallow_snr,
        "detectable": detectable,
        "nn_dist": nearest_neighbor_dist(pos_true),
        "init_err": np.hypot(x_init - x_true, y_init - y_true),
    }


def run_method(
    method: str,
    trial: dict[str, np.ndarray],
    psf: "sep.PSF",
    args: argparse.Namespace,
) -> dict[str, np.ndarray | float]:
    kwargs = {
        "var": args.noise * args.noise,
        "fit_radius": args.fit_radius,
        "fit_positions": True,
    }
    if method == "ungrouped":
        kwargs["grouped"] = False
    elif method == "grouped":
        kwargs["grouped"] = True
        kwargs["group_factor"] = args.group_factor
    elif method == "grouped_damped":
        kwargs["grouped"] = True
        kwargs["group_factor"] = args.group_factor
        kwargs["damp_snthresh"] = args.damp_snthresh
    else:
        raise ValueError(f"Unexpected method: {method}")

    start = time.perf_counter()
    flux, fluxerr, xfit, yfit, flag, chi2, niter = sep.psf_fit(
        trial["data"], trial["x_init"], trial["y_init"], psf, **kwargs
    )
    elapsed = time.perf_counter() - start

    return {
        "elapsed": elapsed,
        "flux": np.asarray(flux, dtype=np.float64),
        "fluxerr": np.asarray(fluxerr, dtype=np.float64),
        "xfit": np.asarray(xfit, dtype=np.float64),
        "yfit": np.asarray(yfit, dtype=np.float64),
        "flag": np.asarray(flag),
        "chi2": np.asarray(chi2, dtype=np.float64),
        "niter": np.asarray(niter),
    }


def summarize_fractional_error(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"median": float("nan"), "mad": float("nan"), "rmse": float("nan"), "p90_abs": float("nan")}
    return {
        "median": float(np.median(values)),
        "mad": float(np.median(np.abs(values - np.median(values))) * 1.4826),
        "rmse": float(np.sqrt(np.mean(values * values))),
        "p90_abs": float(np.quantile(np.abs(values), 0.90)),
    }


def summarize_positive(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"median": float("nan"), "p90": float("nan"), "p99": float("nan")}
    return {
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
    }


def safe_median(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.median(values))


def safe_quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, q))


def safe_rate(mask: np.ndarray) -> float:
    if mask.size == 0:
        return float("nan")
    return float(np.mean(mask))


def build_metrics(
    trial: dict[str, np.ndarray],
    result: dict[str, np.ndarray | float],
) -> dict[str, float]:
    flux = result["flux"]
    fluxerr = result["fluxerr"]
    xfit = result["xfit"]
    yfit = result["yfit"]
    flag = result["flag"]
    chi2 = result["chi2"]
    niter = result["niter"]

    detectable = trial["detectable"]
    sub = ~detectable

    frac_err = (flux[detectable] - trial["flux_true"][detectable]) / trial["flux_true"][
        detectable
    ]
    pos_err = np.hypot(
        xfit[detectable] - trial["x_true"][detectable],
        yfit[detectable] - trial["y_true"][detectable],
    )
    init_drift = np.hypot(xfit[sub] - trial["x_init"][sub], yfit[sub] - trial["y_init"][sub])
    sub_true_err = np.hypot(xfit[sub] - trial["x_true"][sub], yfit[sub] - trial["y_true"][sub])
    fit_snr = np.divide(
        flux,
        fluxerr,
        out=np.zeros_like(flux, dtype=np.float64),
        where=fluxerr > 0.0,
    )
    abs_sub_snr = np.abs(fit_snr[sub])

    frac_stats = summarize_fractional_error(frac_err)
    pos_stats = summarize_positive(pos_err)
    sub_snr_stats = summarize_positive(abs_sub_snr)
    sub_drift_stats = summarize_positive(init_drift)
    sub_true_stats = summarize_positive(sub_true_err)

    finite_chi2 = chi2[np.isfinite(chi2)]
    metrics = {
        "elapsed_ms": 1000.0 * float(result["elapsed"]),
        "detectable_count": int(np.count_nonzero(detectable)),
        "subthreshold_count": int(np.count_nonzero(sub)),
        "detect_flux_bias": frac_stats["median"],
        "detect_flux_mad": frac_stats["mad"],
        "detect_flux_rmse": frac_stats["rmse"],
        "detect_flux_p90_abs": frac_stats["p90_abs"],
        "detect_pos_median": pos_stats["median"],
        "detect_pos_p90": pos_stats["p90"],
        "detect_pos_p99": pos_stats["p99"],
        "sub_abs_snr_median": sub_snr_stats["median"],
        "sub_abs_snr_p90": sub_snr_stats["p90"],
        "sub_abs_snr_p99": sub_snr_stats["p99"],
        "sub_promoted_pos3": safe_rate(fit_snr[sub] > 3.0),
        "sub_promoted_neg3": safe_rate(fit_snr[sub] < -3.0),
        "sub_drift_median": sub_drift_stats["median"],
        "sub_drift_p90": sub_drift_stats["p90"],
        "sub_drift_p99": sub_drift_stats["p99"],
        "sub_trueerr_median": sub_true_stats["median"],
        "sub_trueerr_p90": sub_true_stats["p90"],
        "neg_flux_rate": float(np.mean(flux < 0.0)),
        "flag_rate": float(np.mean(flag != 0)),
        "chi2_median": safe_median(finite_chi2),
        "niter_median": float(np.median(niter)),
    }
    return metrics


def accumulate_metrics(
    store: dict[str, list[float]],
    metrics: dict[str, float],
) -> None:
    for key, value in metrics.items():
        store.setdefault(key, []).append(float(value))


def summarize_accumulator(store: dict[str, list[float]]) -> dict[str, float]:
    return {key: float(np.mean(values)) for key, values in store.items()}


def target_method(args: argparse.Namespace, methods: list[str]) -> str:
    if args.check_method:
        if args.check_method not in methods:
            raise SystemExit(f"--check-method {args.check_method!r} is not in --methods")
        return args.check_method
    if "grouped_damped" in methods:
        return "grouped_damped"
    if "grouped" in methods:
        return "grouped"
    return methods[-1]


def print_catalog_summary(
    args: argparse.Namespace,
    accum: dict[str, list[float]],
) -> None:
    print("Noisy deeper-catalog grouped PSF test")
    print(
        f"  size={args.size}, nsources={args.nsources}, trials={args.trials}, "
        f"fwhm={args.fwhm:.2f}, noise={args.noise:.2f}"
    )
    print(
        f"  catalog_depth={args.catalog_depth:.2f}, catalog_sn={args.catalog_sn:.2f}, "
        f"shallow_detect_sn={args.detect_sn:.2f}, group_factor={args.group_factor:.2f}"
    )
    if args.fit_radius > 0.0:
        print(f"  fit_radius={args.fit_radius:.2f}")
    if args.damp_snthresh > 0.0:
        print(f"  damp_snthresh={args.damp_snthresh:.2f}")
    print(
        f"  clustering: cluster_prob={args.cluster_prob:.2f}, "
        f"cluster_radius={args.cluster_radius:.2f} FWHM, "
        f"min_sep={args.min_sep:.2f} FWHM"
    )
    print(
        f"  initial offsets: floor={args.init_floor:.3f} px, "
        f"cap={args.init_cap:.3f} px, scale={args.init_scale:.2f}"
    )
    print("")

    detectable_frac = np.mean(accum["detectable_frac"])
    subthreshold_frac = np.mean(accum["subthreshold_frac"])
    print("Catalog statistics")
    print(
        f"  detectable fraction in shallow image: {detectable_frac:.3f}  "
        f"sub-threshold fraction: {subthreshold_frac:.3f}"
    )
    print(
        f"  nearest-neighbor separation: median={np.mean(accum['nn_median_fwhm']):.2f} "
        f"FWHM  p10={np.mean(accum['nn_p10_fwhm']):.2f} FWHM"
    )
    print(
        f"  input position error: detectable med={np.mean(accum['init_detect_med']):.3f} px  "
        f"sub-threshold med={np.mean(accum['init_sub_med']):.3f} px"
    )
    print("")


def print_method_report(name: str, summary: dict[str, float]) -> None:
    print(METHOD_LABELS[name])
    print(
        f"  runtime: {summary['elapsed_ms']:.2f} ms / trial  "
        f"niter_med={summary['niter_median']:.2f}  chi2_med={summary['chi2_median']:.3f}"
    )
    print(
        f"  detectable: flux bias={summary['detect_flux_bias']:+.4f}  "
        f"mad={summary['detect_flux_mad']:.4f}  rmse={summary['detect_flux_rmse']:.4f}  "
        f"|frac|_p90={summary['detect_flux_p90_abs']:.4f}"
    )
    print(
        f"              pos med={summary['detect_pos_median']:.4f} px  "
        f"p90={summary['detect_pos_p90']:.4f} px  p99={summary['detect_pos_p99']:.4f} px"
    )
    print(
        f"  sub-threshold: |fit S/N| med={summary['sub_abs_snr_median']:.3f}  "
        f"p90={summary['sub_abs_snr_p90']:.3f}  p99={summary['sub_abs_snr_p99']:.3f}"
    )
    print(
        f"                 promoted +3sigma={summary['sub_promoted_pos3']:.3f}  "
        f"-3sigma={summary['sub_promoted_neg3']:.3f}"
    )
    print(
        f"                 drift-from-input med={summary['sub_drift_median']:.4f} px  "
        f"p90={summary['sub_drift_p90']:.4f} px  p99={summary['sub_drift_p99']:.4f} px"
    )
    print(
        f"                 true-pos err med={summary['sub_trueerr_median']:.4f} px  "
        f"p90={summary['sub_trueerr_p90']:.4f} px"
    )
    print(
        f"  overall: negative flux rate={summary['neg_flux_rate']:.3f}  "
        f"flag rate={summary['flag_rate']:.3f}"
    )
    print("")


def check_summary(
    args: argparse.Namespace,
    method: str,
    summary: dict[str, float],
) -> list[str]:
    failures: list[str] = []
    if summary["detect_pos_p90"] > args.max_detectable_pos_p90:
        failures.append(
            f"{METHOD_LABELS[method]} detectable p90 position error "
            f"{summary['detect_pos_p90']:.3f} > {args.max_detectable_pos_p90:.3f}"
        )
    if summary["detect_flux_rmse"] > args.max_detectable_flux_rmse:
        failures.append(
            f"{METHOD_LABELS[method]} detectable flux RMSE "
            f"{summary['detect_flux_rmse']:.3f} > {args.max_detectable_flux_rmse:.3f}"
        )
    if summary["sub_promoted_pos3"] > args.max_subthreshold_promoted:
        failures.append(
            f"{METHOD_LABELS[method]} sub-threshold promoted fraction "
            f"{summary['sub_promoted_pos3']:.3f} > {args.max_subthreshold_promoted:.3f}"
        )
    if summary["sub_drift_p90"] > args.max_subthreshold_drift_p90:
        failures.append(
            f"{METHOD_LABELS[method]} sub-threshold p90 input drift "
            f"{summary['sub_drift_p90']:.3f} > {args.max_subthreshold_drift_p90:.3f}"
        )
    return failures


def main() -> int:
    args = parse_args()
    methods = normalize_methods(args)
    rng = np.random.default_rng(args.seed)

    psf = sep.PSF.from_gaussian(args.fwhm)
    psf_l2 = render_unit_psf_norm(psf, args.fit_radius)

    catalog_accum: dict[str, list[float]] = {}
    method_accum: dict[str, dict[str, list[float]]] = {name: {} for name in methods}

    for trial_idx in range(args.trials):
        trial = simulate_trial(rng, args, psf, psf_l2)
        detectable = trial["detectable"]
        accumulate_metrics(
            catalog_accum,
            {
                "detectable_frac": float(np.mean(detectable)),
                "subthreshold_frac": float(np.mean(~detectable)),
                "nn_median_fwhm": safe_median(trial["nn_dist"]) / args.fwhm,
                "nn_p10_fwhm": safe_quantile(trial["nn_dist"], 0.10) / args.fwhm,
                "init_detect_med": safe_median(trial["init_err"][detectable]),
                "init_sub_med": safe_median(trial["init_err"][~detectable]),
            },
        )

        for method in methods:
            result = run_method(method, trial, psf, args)
            metrics = build_metrics(trial, result)
            accumulate_metrics(method_accum[method], metrics)

        if args.report_trials:
            sub_frac = float(np.mean(~detectable))
            print(
                f"trial {trial_idx + 1:02d}: sub-threshold fraction={sub_frac:.3f}, "
                f"init_err_med={np.median(trial['init_err']):.3f} px"
            )

    print_catalog_summary(args, catalog_accum)

    summaries = {
        method: summarize_accumulator(store) for method, store in method_accum.items()
    }
    for method in methods:
        print_method_report(method, summaries[method])

    if args.check:
        method = target_method(args, methods)
        failures = check_summary(args, method, summaries[method])
        if failures:
            print("CHECK FAILED")
            for failure in failures:
                print(f"  - {failure}")
            return 1
        print(f"CHECK PASSED for {METHOD_LABELS[method]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
