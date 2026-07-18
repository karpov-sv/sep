SEP-X
=====

*Extended Python library for source extraction and photometry*


About
-----

`Source Extractor <http://www.astromatic.net/software/sextractor>`_
(Bertin & Arnouts 1996) is a widely used
command-line program for segmentation and analysis of astronomical
images. It reads in FITS format files, performs a configurable series
of tasks, including background estimation, source detection,
deblending and a wide array of source measurements, and finally
outputs a FITS format catalog file.

While Source Extractor is highly useful, the fact that it can only be
used as an executable can limit its applicability or lead to awkward
workflows. There is often a desire to have programmatic access to
perform one or more of the above tasks on in-memory images as part of
a larger custom analysis.

**SEP-X makes the core algorithms of Source Extractor available as a
library of stand-alone functions and classes.** These operate directly
on in-memory arrays (no FITS files or configuration files).  The code
is derived from the Source Extractor code base (written in C) and aims
to produce results compatible with Source Extractor whenever possible.
SEP consists of a C library with no dependencies outside the standard
library, and a Python module that wraps the C library in a Pythonic
API. The Python wrapper operates on NumPy arrays with NumPy as its
only dependency. See below for language-specfic build and usage
instructions.

**Some features:**

- spatially variable background and noise estimation
- source extraction, with on-the-fly convolution and source deblending
- circular and elliptical aperture photometry
- fast: implemented in C with Python bindings via Cython

**Additional features not in Source Extractor:**

- Optimized matched filter for variable noise in source extraction.
- Circular annulus and elliptical annulus aperture photometry functions.
- Local background subtraction in shape consistent with aperture in
  aperture photometry functions.
- Exact pixel overlap mode in all aperture photometry functions.
- Masking of elliptical regions on images.


SEP-X and Upstream SEP
......................

SEP-X is an independently maintained fork of
`upstream SEP <https://github.com/sep-developers/sep>`_. It retains SEP's
Python and C APIs where practical, but has substantial additional extraction,
photometry, PSF-fitting, and centroiding functionality.

The published distribution is named ``sep-x`` and its Python extension is
``sep_x``. SEP-X and upstream ``sep`` can therefore be installed in the same
Python environment. To use SEP-X with the familiar local API name::

    import sep_x as sep

This separation applies to Python only. SEP-X retains the C library name
``libsep`` and header name ``sep.h`` for C source compatibility. Do not
install SEP-X and upstream SEP's C libraries into the same prefix or link both
into one process.

Major changes include robust local-background estimation, grouped optimal
extraction, a PSF modelling and fitting API, enhanced detection and
deblending, and PSF- and segmentation-aware windowed centroids. See
``CHANGES.md`` and :doc:`the C API change notes <changelogs/changes_to_c_api>`
for detailed compatibility information.


Installation
------------

with pip
........

Once a release is published, SEP-X can be installed with
`pip <https://pip.pypa.io>`_. After ensuring that numpy is installed, run ::

    python -m pip install sep-x

If you get an error about permissions, you are probably using your
system Python. In this case, I recommend using `pip's "user install"
<https://pip.pypa.io/en/latest/user_guide/#user-installs>`_ option to
install sep-x into your user directory ::

    python -m pip install --user sep-x

Do **not** install ``sep-x`` or other third-party Python packages using
``sudo`` unless you are fully aware of the risks.


Usage Guide
-----------

.. toctree::
   :maxdepth: 1

   tutorial
   filter
   apertures
   psf
   changelogs/changelog

.. toctree::
   :hidden:

   reference

For complete API documentation, see :doc:`reference`.


Contributing
------------

Report a bug or documentation issue:
http://github.com/karpov-sv/sep/issues

Development of ``sep-x`` takes place on GitHub at
http://github.com/karpov-sv/sep. Contributions of bug fixes,
documentation improvements and minor feature additions are welcome via
GitHub pull requests. For major features, it is best to open an issue
discussing the change first.


License and Citation
--------------------

The license for SEP is the Lesser GNU Public License (LGPL), granted
with the permission from the original author of Source Extractor.

If you use SEP in a publication, please cite `Barbary (2016)
<http://dx.doi.org/10.21105/joss.00058>`_ and the original Source
Extractor paper: `Bertin & Arnouts 1996
<http://adsabs.harvard.edu/abs/1996A%26AS..117..393B>`_.

The DOI for the sep v1.0.0 code release is `10.5281/zenodo.159035
<http://dx.doi.org/10.5281/zenodo.159035>`_.
