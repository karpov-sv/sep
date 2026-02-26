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

`sep.psf_fit` returns ``flux``, ``fluxerr``, fitted positions, and flags:

.. code-block:: python

    flux, fluxerr, xfit, yfit, flag = sep.psf_fit(data, x, y, psf)

Use ``fit_positions=False`` for flux-only mode at fixed input positions:

.. code-block:: python

    flux, fluxerr, xfit, yfit, flag = sep.psf_fit(
        data, x, y, psf, fit_positions=False
    )

In flux-only mode, SEP computes the optimal-extraction estimator

.. math::

   F = \frac{\sum_i P_i D_i / V_i}{\sum_i P_i^2 / V_i}

where :math:`P_i` is the resampled PSF, :math:`D_i` is data, and :math:`V_i`
is per-pixel variance.

With ``fit_positions=True`` (default), SEP iteratively solves for flux and
subpixel shifts ``(dx, dy)`` using a linearized least-squares model per
iteration.

Grouped fitting for blends
--------------------------

Set ``grouped=True`` to fit overlapping sources simultaneously:

.. code-block:: python

    flux, fluxerr, xfit, yfit, flag = sep.psf_fit(
        data, x, y, psf, grouped=True, group_factor=2.0
    )

Sources are grouped by stamp overlap and each group is solved jointly.
This generally improves deblending relative to fitting each source
independently.

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
