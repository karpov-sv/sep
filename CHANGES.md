Unreleased
==========

* Add `deblend_saddle` to `extract` for rejecting low-prominence watershed
  basins by their peak-to-saddle depth in detection-statistic units (S/N
  units with matched filtering and an error input). The default, zero,
  disables prominence merging.
* Merge rejected watershed basins along higher-peak saddle-parent chains
  where possible, rather than assigning their pixels solely by proximity.
* Preserve higher watershed peaks when rejecting shallow saddles to weaker
  neighbors; previously both peaks could be rejected, merging distinct
  sources elsewhere in the same footprint.
* Add `sampled` option to `PSF` for supersampled models that are point samples
  of the pixel-integrated PSF (PSFEx convention); they are resampled by
  interpolation instead of area overlap, which broadened them and biased PSF
  fluxes by several percent. `PSF.from_psfex` sets it.
* Add C entry points `sep_extract_ex` and `sep_extract_with_pixels_ex` for
  `deblend_saddle` control. Preserve the original extraction signatures and
  behavior, with prominence merging disabled for existing C callers.
* Add `sampled` to the public `sep_psf` structure; C clients using PSF models
  should verify structure layout compatibility on their target platform.

v1.5.1 (18 July 2026)
====================

* Rename the Python extension to `sep_x` so SEP-X can be installed alongside
  upstream SEP. Use `import sep_x as sep` when migrating existing code.
* Fix local annulus background subtraction in optimal-extraction photometry.

v1.5.0 (11 July 2026)
====================

* Add optimal-extraction circular photometry (`sum_circle_optimal`) with
  auto-grouping for overlapping apertures and configurable grouping radius.
* Add sigma-clipped annulus statistics (`stats_circann`, `stats_ellipann`)
  for robust local background estimation.
* Add sigma-clipped annulus background controls (`clip_sigma`, `clip_iters`)
  to `sum_circle`, `sum_ellipse`, and `sum_circle_optimal`, with a fast
  legacy fallback when `clip_iters=0`.
* Add PSF models, PSFEx loading, source rendering, PSF photometry, grouped
  fitting, PSF-matched peak detection, and optional OpenMP acceleration.
* Add optional watershed and fixed-PSF deblending, source FWHM estimates,
  and improvements to faint-source detection and extraction performance.
* Add PSF-weighted windowed centroiding and controls to constrain centroid
  motion and use segmentation maps.
* Improve masking, grouped photometry, and PSF fitting stability and speed.
* Document the SEP-X fork, organize scientific regression benchmarks, and
  fix release builds and wheel configuration.

v1.3.7 (8 November 2024)
========================

* Test against Python 3.13.
* Update the Makefile to support Semantic Versioning from git tags.
* Update the C libraries to allow for passing the version as a compiler
  flag.
* Update `setup.py` to pass the SCM version to the Cython compiler.
* Include C tests in the Tox suite (for linux and macos only).
* Document any and all changes to the C API since forking.
* Restructure changelog documentation.

v1.3.6 (7 October 2024)
=======================

* Fix wrong int type in Windows
  ([#2](https://github.com/PJ-Watson/sep-pjw/issues/2), thanks to
  @acenko for pointing this out).
* Update tests to run on multiple operating systems.

v1.3.5 (12 June 2024)
=====================

* Small fixes and updates to ensure compatibility with NumPy 2.0.

v1.3.4 (21 February 2024)
========================

* Include .clang-format as a pre-commit hook, to ensure consistent code
  style (improved readability, easier maintenance).
* Fixed `make test` to account for the changes in
  [v1.3.0](https://github.com/PJ-Watson/sep-pjw/releases/tag/v1.3.0).
* All header files include the correct definitions.

v1.3.3 (7 February 2024)
========================

* Add changelog to documentation.
* Add tests for re-running with seg map.
* Fix array boundary bugs when re-running with seg map.
* Fix bug with precision loss when calculating threshold.
* Improve error handling when object pixels exceed pix stack.

v1.3.2 (5 February 2024)
========================

* Move documentation to new location, fix package names and imports.
* Add wheels for Python 3.11/3.12.
* Fix C compilation errors on windows (VLAs).
* Publish updated version to PyPI under new name.

v1.3.1 (31 January 2024)
========================

* Formatting changes (follow [black](https://github.com/psf/black)
  formatting style).
* Fix `bench.py` and `test.py`, removing deprecated functions.
* Move metadata into `pyproject.toml`.
* Add pre-commit hooks for code and docstring validation.
* Change to dynamic versioning (git tag/commit based).

v1.3.0 (1 December 2023)
========================

* The `segmentation_map` argument of `sep.extract()` will now accept
  either an array or boolean. If an existing segmentation map is passed,
  the object detection stage is skipped, and sources will be individually
  analysed according to the provided map. This change is
  backwards-compatible with respect to the Python module.

  Please note that as no deblending is performed, the calculated
  thresholds (and any dependent parameters) may not be the same as
  originally derived.

* Use 64-bit integers throughout, to fix memory addressing with large
  arrays
  ([#122](https://github.com/kbarbary/sep/issues/122 "Original issue"),
  inspired by [Gabe Brammer's fork](https://github.com/gbrammer/sep)
  with additional fixes).
