PSF photometry
==============

SEP provides PSF photometry through a PSF model class (`sep.PSF`) and a
fitting function (`sep.psf_fit`). The implementation supports fixed-position
flux extraction, iterative position+flux fitting, and grouped fitting for
blended sources.

Building a PSF model
--------------------

Create a model directly from component images:

.. code-block:: python

    # data shape: (ncomp, h, w) or (h, w)
    psf = sep.PSF(data, sampling=0.5, degree=0, fwhm=3.5)

or from helpers:

.. code-block:: python

    psf = sep.PSF.from_gaussian(fwhm=3.5, oversampling=2)
    psf = sep.PSF.from_psfex("model.psf")

For spatially varying models, SEP uses a polynomial expansion in normalized
coordinates:

.. math::

   u = (x - x_0) / s_x,\;\;\; v = (y - y_0) / s_y

and combines the component images up to ``degree``.

Sampling and PSF resampling
---------------------------

The ``sampling`` parameter is the PSF pixel size in image pixels:

- ``sampling < 1``: supersampled PSF grid
- ``sampling = 1``: native image sampling
- ``sampling > 1``: coarser-than-image sampling

The native fitting stamp size is derived from continuous scaling:

.. math::

   w_\mathrm{native} = \mathrm{round}(w_\mathrm{psf} \cdot sampling),\;\;
   h_\mathrm{native} = \mathrm{round}(h_\mathrm{psf} \cdot sampling)

For supersampled models (``sampling < 1``), SEP uses a conservative
area-overlap remap (separable x/y passes) to downsample to native pixels.
For ``sampling >= 1``, SEP uses Lanczos interpolation. In both cases, the
resampled stamp is normalized before flux estimation/fitting.

Running PSF photometry
----------------------

`sep.psf_fit` returns ``flux``, ``fluxerr``, fitted positions, flags,
reduced chi-squared, and iteration counts:

.. code-block:: python

    flux, fluxerr, xfit, yfit, flag, chi2, niter = sep.psf_fit(data, x, y, psf)

Use ``fit_positions=False`` for flux-only mode at fixed input positions:

.. code-block:: python

    flux, fluxerr, xfit, yfit, flag, chi2, niter = sep.psf_fit(
        data, x, y, psf, fit_positions=False
    )

In flux-only mode, SEP computes the optimal-extraction estimator

.. math::

   F = \frac{\sum_i P_i D_i / V_i}{\sum_i P_i^2 / V_i}

where :math:`P_i` is the resampled PSF, :math:`D_i` is data, and :math:`V_i`
is per-pixel variance.

In flux-only mode, ``chi2`` is returned as ``nan`` and ``niter`` as 0 because
no iterative position fit is performed.

With ``fit_positions=True`` (default), SEP iteratively solves for flux and
subpixel shifts ``(dx, dy)`` using a linearized least-squares model per
iteration.

PSF-matched significance image
------------------------------

Use `sep.psf_snr` to compute a PSF-matched significance image:

.. code-block:: python

    snr = sep.psf_snr(data, psf, var=variance)

At each pixel, SEP evaluates the PSF centered on that pixel and computes

.. math::

   \mathrm{SNR} = \frac{\sum_i P_i D_i / V_i}
                      {\sqrt{\sum_i P_i^2 / V_i}}

Masked pixels and pixels with non-positive variance are excluded from each
local sum. The result can be used as a PSF-weighted detection image.

Set ``local_bkg=True`` to fit and remove a constant background term inside
each PSF footprint:

.. code-block:: python

    snr = sep.psf_snr(data, psf, var=variance, local_bkg=True)

Use `sep.psf_extract` to run source extraction directly on this detection
image. By default, ``mode="segments"`` uses SEP's connected-component
extraction and deblending on the PSF-matched significance image:

.. code-block:: python

    objects = sep.psf_extract(data, 5.0, psf, var=variance)

This is equivalent to calling `sep.psf_snr` followed by `sep.extract` with
``filter_kernel=None``. Catalog ``flux`` and ``peak`` fields are therefore
measured on the S/N image, not on the original data.

For crowded fields, unresolved wings, or correlated noise, the raw
PSF-matched image may have a biased background or a non-unit empirical RMS.
Set ``normalize_snr=True`` to estimate a SEP background model on the
PSF-matched image before thresholding:

.. code-block:: python

    objects = sep.psf_extract(
        data, 5.0, psf, var=variance, normalize_snr=True
    )

The ``snr_bw``, ``snr_bh``, ``snr_fw``, ``snr_fh``, and ``snr_fthresh``
arguments control the background mesh used for this normalization.

For crowded fields where connected-component extraction may merge nearby
sources, use `sep.psf_peaks` to find local maxima in the PSF-matched
significance image:

.. code-block:: python

    peaks = sep.psf_peaks(data, 5.0, psf, var=variance)

The returned table contains peak positions and S/N values, not connected
object footprints. Use these candidates as inputs to PSF fitting or grouped
deblending.

Alternatively, use ``mode="peaks"`` in `sep.psf_extract` to find local
maxima and immediately prune them with PSF fits:

.. code-block:: python

    objects = sep.psf_extract(
        data, 5.0, psf, var=variance, mode="peaks", fit_snr=5.0
    )

In this mode, the returned table contains fitted positions, fluxes, fitted
S/N values, peak S/N values, PSF-weighted quality diagnostics (``qf``,
``rchi2``, and ``fracflux``), peak pixel positions, fit chi-square values,
iteration counts, and fit flags. Set ``fit_snr=None`` to return the raw peak
table without fitting.

The fitted peak catalog may also be pruned with optional quality cuts:
``min_qf`` rejects incomplete PSF footprints, ``max_rchi2`` rejects poor PSF
fits, and ``min_fracflux`` rejects candidates whose fitted source model
accounts for only a small fraction of the local PSF-weighted flux.

Set ``grouped=True`` to fit overlapping peak candidates simultaneously before
applying the fitted-S/N cut:

.. code-block:: python

    objects = sep.psf_extract(
        data, 5.0, psf, var=variance, mode="peaks",
        fit_snr=5.0, grouped=True
    )

Building a PSF image model
--------------------------

Use `sep.model_psf` to render one or more PSF sources into an image array:

.. code-block:: python

    model = np.zeros_like(data, dtype=np.float32)
    sep.model_psf(model, x, y, flux, psf)

The array is updated in-place. Inputs ``x``, ``y``, and ``flux`` follow
NumPy broadcasting rules, similar to `sep.mask_ellipse`.

Grouped fitting for blends
--------------------------

Set ``grouped=True`` to fit overlapping sources simultaneously:

.. code-block:: python

    flux, fluxerr, xfit, yfit, flag, chi2, niter = sep.psf_fit(
        data, x, y, psf, grouped=True, group_factor=2.0
    )

Sources are grouped by direct fit-support overlap and each group is solved
jointly. If ``fit_radius`` is nonzero, that radius defines the support. If
``fit_radius`` is zero, SEP derives an effective influence radius from the
PSF profile instead of using the full stamp extent.
For small groups SEP uses an exact simultaneous fit. Large connected
components are handled with overlapping local fits that reuse neighbor
parameters between passes.

``group_factor`` controls the local fitting halo, not the connectivity graph.
Values larger than 1 therefore expand the local context used for the fit
without merging sources that do not directly overlap. This avoids the
pathological giant-group behavior that can occur in crowded fields when a
single radius is used for both grouping and fitting extent. Values below 1
behave like 1 and therefore do not shrink the support-based neighborhood.

Error model, masks, and flags
-----------------------------

Noise can be provided via ``var`` or ``err`` (scalar or image). If ``gain``
is provided, SEP adds a Poisson term from source counts to the variance model,
and includes the corresponding term in ``fluxerr``.

Pixels are excluded from fitting when they are masked or blocked by ``segmap``
(unless the segmentation ID matches ``seg_id`` for that source). Edge
truncation and masking propagate to output flags:

- ``sep.APER_TRUNC``: stamp touches image boundary
- ``sep.APER_HASMASKED``: one or more pixels were excluded
- ``sep.APER_ALLMASKED``: no valid pixels remained
