# SEP benchmarks and scientific diagnostics

These scripts exercise performance, recovery accuracy, and crowded-field
behavior over larger parameter grids than the package test suite. They print
reports rather than enforce CI pass/fail thresholds, because timing and
ensemble metrics depend on the machine and chosen simulation parameters.

Run them from an environment where the local SEP extension is installed, for
example after `python -m pip install --editable .`.

## Synthetic benchmarks

- `bench.py`: established SEP-versus-photutils timing suite for background,
  extraction, and aperture operations.
- `bench_aperture_performance.py`: throughput of aperture, optimal, and PSF
  photometry on a large synthetic image.
- `check_grouped_psf_noisy_catalog.py`: grouped PSF fitting with noisy
  positions from a catalog deeper than the fitted image.
- `compare_optimal_grouped_ungrouped.py`: detailed isolated-source comparison
  of grouped and ungrouped optimal extraction.
- `compare_optimal_photometry.py`: aperture and optimal-extraction accuracy as
  crowding increases.
- `compare_psf_photometry.py`: aperture, optimal, and PSF-fit accuracy across
  crowding levels.
- `deblend_crowded_pairs.py`: equal-flux pair recovery versus separation.
- `deblend_fluxratio_pairs.py`: high-contrast companion recovery versus
  separation and flux ratio.
- `fwhm_estimation.py`: FWHM recovery versus source width and crowding.
- `grouped_optimal_stability.py`: pair and long-chain stability around the
  grouped-solver transition.
- `winpos_crowded_stability.py`: windowed-position accuracy versus nearest
  neighbor distance.

All synthetic scripts use fixed random seeds by default. Command-line options
control image size, source density, noise, and the parameter grids.

## Optional-dependency and real-image tools

- `compare_aperture_photometry.py` compares SEP with `photutils` and requires
  `astropy` plus `photutils`.
- `bench_psf_detection.py` accepts a FITS image and can optionally use an
  external truth catalog or `crowdsource`. FITS input requires `fitsio` or
  `astropy`; truth-catalog support additionally uses `pyarrow` and WCS support.

Examples:

```sh
python benchmarks/deblend_crowded_pairs.py --pairs-per-sep 30
python benchmarks/fwhm_estimation.py --fwhm-list 1.2,2.5,5,7
python benchmarks/grouped_optimal_stability.py
python benchmarks/bench_aperture_performance.py --size 2048 --nobj 2000
```
