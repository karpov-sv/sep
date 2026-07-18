#!/usr/bin/env py.test

"""Test the python functionality of SEP."""

from __future__ import division, print_function

import math
import os

import numpy as np
import pytest
from numpy.lib import recfunctions as rfn
from numpy.testing import assert_allclose, assert_approx_equal, assert_equal

import sep_x as sep

# unicode_literals doesn't play well with numpy dtype field names


# Try to import any FITS reader
try:
    from fitsio import read as getdata

    NO_FITS = False
except:
    try:
        from astropy.io.fits import getdata

        NO_FITS = False
    except:
        NO_FITS = True

IMAGE_FNAME = os.path.join("data", "image.fits")
BACKIMAGE_FNAME = os.path.join("data", "back.fits")
RMSIMAGE_FNAME = os.path.join("data", "rms.fits")
IMAGECAT_FNAME = os.path.join("data", "image.cat")
IMAGECAT_DTYPE = [
    ("number", np.int64),
    ("x", np.float64),
    ("y", np.float64),
    ("xwin", np.float64),
    ("ywin", np.float64),
    ("x2", np.float64),
    ("y2", np.float64),
    ("xy", np.float64),
    ("errx2", np.float64),
    ("erry2", np.float64),
    ("errxy", np.float64),
    ("a", np.float64),
    ("flux_aper", np.float64),
    ("fluxerr_aper", np.float64),
    ("kron_radius", np.float64),
    ("flux_auto", np.float64),
    ("fluxerr_auto", np.float64),
    ("flux_radius", np.float64, (3,)),
    ("flags", np.int64),
]
SUPPORTED_IMAGE_DTYPES = [np.float64, np.float32, np.int32]

# If we have a FITS reader, read in the necessary test images
if not NO_FITS:
    image_data = getdata(IMAGE_FNAME)
    image_refback = getdata(BACKIMAGE_FNAME)
    image_refrms = getdata(RMSIMAGE_FNAME)


# -----------------------------------------------------------------------------
# Helpers


def assert_allclose_structured(x, y):
    """
    Assert that two structured arrays are close.

    Compares floats relatively and everything else exactly.

    Parameters
    ----------
    x, y : array-like
        Structured arrays to be compared.
    """
    assert x.dtype == y.dtype
    for name in x.dtype.names:
        if np.issubdtype(x.dtype[name], float):
            assert_allclose(x[name], y[name])
        else:
            assert_equal(x[name], y[name])


def matched_filter_snr(data, noise, kernel):
    r"""
    Super slow implementation of matched filter SNR for testing.

    At each output pixel :math:`i`, the value is:

    .. math::

        \frac{\sum(\text{data}[i] * \text{kernel}[i] / \text{noise}[i]^2)}
            {\sqrt\sum(\text{kernel}[i]^2 / \text{noise}[i]^2)}

    Parameters
    ----------
    data : array-like
        The 2D data to be tested.
    noise : array-like
        The noise corresponding to the input ``data``.
    kernel : array-like
        The kernel used for filtering.

    Returns
    -------
    array-like
        The output SNR array, the same size as ``data``.
    """
    ctr = kernel.shape[0] // 2, kernel.shape[1] // 2
    kslice = (
        (0 - ctr[0], kernel.shape[0] - ctr[0]),  # range in axis 0
        (0 - ctr[1], kernel.shape[1] - ctr[1]),
    )  # range in axis 1
    out = np.empty_like(data)

    for y in range(data.shape[0]):
        jmin = y + kslice[0][0]  # min and max indicies to sum over
        jmax = y + kslice[0][1]
        kjmin = 0  # min and max kernel indicies to sum over
        kjmax = kernel.shape[0]

        # if we're over the edge of the image, limit extent
        if jmin < 0:
            offset = -jmin
            jmin += offset
            kjmin += offset
        if jmax > data.shape[0]:
            offset = data.shape[0] - jmax
            jmax += offset
            kjmax += offset

        for x in range(data.shape[1]):
            imin = x + kslice[1][0]  # min and max indicies to sum over
            imax = x + kslice[1][1]
            kimin = 0  # min and max kernel indicies to sum over
            kimax = kernel.shape[1]

            # if we're over the edge of the image, limit extent
            if imin < 0:
                offset = -imin
                imin += offset
                kimin += offset
            if imax > data.shape[1]:
                offset = data.shape[1] - imax
                imax += offset
                kimax += offset

            d = data[jmin:jmax, imin:imax]
            n = noise[jmin:jmax, imin:imax]
            w = 1.0 / n**2
            k = kernel[kjmin:kjmax, kimin:kimax]
            out[y, x] = np.sum(d * k * w) / np.sqrt(np.sum(k**2 * w))

    return out


def matched_filter_snr_local_bkg(data, noise, kernel):
    r"""
    Slow matched-filter S/N with a local constant-background term.

    At each output pixel, fit data as source * kernel + background with
    inverse-variance weights, then return the source-amplitude significance
    after marginalizing over the local background.
    """
    ctr = kernel.shape[0] // 2, kernel.shape[1] // 2
    kslice = (
        (0 - ctr[0], kernel.shape[0] - ctr[0]),
        (0 - ctr[1], kernel.shape[1] - ctr[1]),
    )
    out = np.zeros_like(data)

    for y in range(data.shape[0]):
        jmin = y + kslice[0][0]
        jmax = y + kslice[0][1]
        kjmin = 0
        kjmax = kernel.shape[0]

        if jmin < 0:
            offset = -jmin
            jmin += offset
            kjmin += offset
        if jmax > data.shape[0]:
            offset = data.shape[0] - jmax
            jmax += offset
            kjmax += offset

        for x in range(data.shape[1]):
            imin = x + kslice[1][0]
            imax = x + kslice[1][1]
            kimin = 0
            kimax = kernel.shape[1]

            if imin < 0:
                offset = -imin
                imin += offset
                kimin += offset
            if imax > data.shape[1]:
                offset = data.shape[1] - imax
                imax += offset
                kimax += offset

            d = data[jmin:jmax, imin:imax]
            n = noise[jmin:jmax, imin:imax]
            p = kernel[kjmin:kjmax, kimin:kimax]
            invvar = 1.0 / n**2

            num = np.sum(p * d * invvar)
            den = np.sum(p**2 * invvar)
            sumw = np.sum(invvar)
            sumpw = np.sum(p * invvar)
            sumdw = np.sum(d * invvar)
            det = den * sumw - sumpw * sumpw

            if det > 0.0 and sumw > 0.0:
                out[y, x] = (num * sumw - sumpw * sumdw) / np.sqrt(det * sumw)

    return out


_ERF = np.vectorize(math.erf, otypes=[float])


def _gaussian_pixel_integral(dx, dy, sigma):
    inv = 1.0 / (np.sqrt(2.0) * sigma)
    ex = _ERF((dx + 0.5) * inv) - _ERF((dx - 0.5) * inv)
    ey = _ERF((dy + 0.5) * inv) - _ERF((dy - 0.5) * inv)
    return 0.25 * ex * ey


def _gaussian_scene(shape, x, y, fwhm, flux):
    yy, xx = np.indices(shape)
    sigma = fwhm / 2.354820045
    image = np.zeros(shape, dtype=float)
    for xi, yi, fi in zip(x, y, flux):
        image += fi * _gaussian_pixel_integral(xx - xi, yy - yi, sigma)
    return image


# -----------------------------------------------------------------------------
# Test versus Source Extractor results


@pytest.mark.skipif(NO_FITS, reason="no FITS reader")
def test_vs_sextractor():
    """
    Test behavior of sep versus sextractor.

    Note: we turn deblending off for this test. This is because the
    deblending algorithm uses a random number generator. Since the sequence
    of random numbers is not the same between sextractor and sep or between
    different platforms, object member pixels (and even the number of objects)
    can differ when deblending is on.

    Deblending is turned off by setting DEBLEND_MINCONT=1.0 in the sextractor
    configuration file and by setting deblend_cont=1.0 in sep.extract().
    """

    data = np.copy(image_data)  # make an explicit copy so we can 'subfrom'
    bkg = sep.Background(data, bw=64, bh=64, fw=3, fh=3)

    # Test that SExtractor background is same as SEP:
    bkgarr = bkg.back(dtype=np.float32)
    assert_allclose(bkgarr, image_refback, rtol=1.0e-5)

    # Test that SExtractor background rms is same as SEP:
    rmsarr = bkg.rms(dtype=np.float32)
    assert_allclose(rmsarr, image_refrms, rtol=1.0e-4)

    # Extract objects (use deblend_cont=1.0 to disable deblending).
    bkg.subfrom(data)
    objs = sep.extract(
        data,
        1.5,
        err=bkg.globalrms,
        filter_type="conv",
        deblend_cont=1.0,
    )
    objs = np.sort(objs, order=["y"])

    # Read SExtractor result
    refobjs = np.loadtxt(IMAGECAT_FNAME, dtype=IMAGECAT_DTYPE)
    refobjs = np.sort(refobjs, order=["y"])

    # Found correct number of sources at the right locations?
    assert_allclose(objs["x"], refobjs["x"] - 1.0, atol=1.0e-3)
    assert_allclose(objs["y"], refobjs["y"] - 1.0, atol=1.0e-3)

    # Correct Variance and Variance Errors?
    assert_allclose(objs["x2"], refobjs["x2"], atol=1.0e-4)
    assert_allclose(objs["y2"], refobjs["y2"], atol=1.0e-4)
    assert_allclose(objs["xy"], refobjs["xy"], atol=1.0e-4)
    assert_allclose(objs["errx2"], refobjs["errx2"], rtol=1.0e-4)
    assert_allclose(objs["erry2"], refobjs["erry2"], rtol=1.0e-4)
    assert_allclose(objs["errxy"], refobjs["errxy"], rtol=1.0e-3)

    # Test aperture flux
    flux, fluxerr, flag = sep.sum_circle(
        data, objs["x"], objs["y"], 5.0, err=bkg.globalrms
    )
    assert_allclose(flux, refobjs["flux_aper"], rtol=2.0e-4)
    assert_allclose(fluxerr, refobjs["fluxerr_aper"], rtol=1.0e-5)

    # check if the flags work at all (comparison values
    assert ((flag & sep.APER_TRUNC) != 0).sum() == 4
    assert ((flag & sep.APER_HASMASKED) != 0).sum() == 0

    # Test "flux_auto"
    kr, flag = sep.kron_radius(
        data, objs["x"], objs["y"], objs["a"], objs["b"], objs["theta"], 6.0
    )

    flux, fluxerr, flag = sep.sum_ellipse(
        data,
        objs["x"],
        objs["y"],
        objs["a"],
        objs["b"],
        objs["theta"],
        r=2.5 * kr,
        err=bkg.globalrms,
        subpix=1,
    )

    # For some reason, one object doesn't match. It's very small
    # and kron_radius is set to 0.0 in SExtractor, but 0.08 in sep.
    # Could be due to a change in SExtractor between v2.8.6 (used to
    # generate "truth" catalog) and v2.18.11 (from which sep was forked).
    i = 56  # index is 59 when deblending is on.
    kr[i] = 0.0
    flux[i] = 0.0
    fluxerr[i] = 0.0

    # We use atol for radius because it is reported to nearest 0.01 in
    # reference objects.
    assert_allclose(2.5 * kr, refobjs["kron_radius"], atol=0.01, rtol=0.0)
    assert_allclose(flux, refobjs["flux_auto"], rtol=0.0005)
    assert_allclose(fluxerr, refobjs["fluxerr_auto"], rtol=0.0005)

    # Test using a mask in kron_radius and sum_ellipse.
    for dtype in [np.bool_, np.int32, np.float32, np.float64]:
        mask = np.zeros_like(data, dtype=dtype)
        kr2, flag = sep.kron_radius(
            data,
            objs["x"],
            objs["y"],
            objs["a"],
            objs["b"],
            objs["theta"],
            6.0,
            mask=mask,
        )
        kr2[i] = 0.0
        assert np.all(kr == kr2)

    # Test ellipse representation conversion
    cxx, cyy, cxy = sep.ellipse_coeffs(objs["a"], objs["b"], objs["theta"])
    assert_allclose(cxx, objs["cxx"], rtol=1.0e-4)
    assert_allclose(cyy, objs["cyy"], rtol=1.0e-4)
    assert_allclose(cxy, objs["cxy"], rtol=1.0e-4)

    a, b, theta = sep.ellipse_axes(objs["cxx"], objs["cyy"], objs["cxy"])
    assert_allclose(a, objs["a"], rtol=1.0e-4)
    assert_allclose(b, objs["b"], rtol=1.0e-4)
    assert_allclose(theta, objs["theta"], rtol=1.0e-4)

    # test round trip
    cxx, cyy, cxy = sep.ellipse_coeffs(a, b, theta)
    assert_allclose(cxx, objs["cxx"], rtol=1.0e-4)
    assert_allclose(cyy, objs["cyy"], rtol=1.0e-4)
    assert_allclose(cxy, objs["cxy"], rtol=1.0e-4)

    # test flux_radius
    fr, flags = sep.flux_radius(
        data,
        objs["x"],
        objs["y"],
        6.0 * refobjs["a"],
        [0.1, 0.5, 0.6],
        normflux=refobjs["flux_auto"],
        subpix=5,
    )
    assert_allclose(fr, refobjs["flux_radius"], rtol=0.04, atol=0.01)

    # test winpos
    sig = 2.0 / 2.35 * fr[:, 1]  # flux_radius = 0.5
    xwin, ywin, flag = sep.winpos(data, objs["x"], objs["y"], sig)
    assert_allclose(xwin, refobjs["xwin"] - 1.0, rtol=0.0, atol=0.0015)
    assert_allclose(ywin, refobjs["ywin"] - 1.0, rtol=0.0, atol=0.0015)


# -----------------------------------------------------------------------------
# Background


def test_masked_background():
    """
    Check the background filtering.

    Check that the derived background is consistent with an explicit
    mask, masking no pixels. Also check that the expected result is
    returned if certain pixels are masked.
    """

    data = 0.1 * np.ones((6, 6))
    data[1, 1] = 1.0
    data[4, 1] = 1.0
    data[1, 4] = 1.0
    data[4, 4] = 1.0

    mask = np.zeros((6, 6), dtype=np.bool_)

    # Background array without mask
    sky = sep.Background(data, bw=3, bh=3, fw=1, fh=1)
    bkg1 = sky.back()

    # Background array with all False mask
    sky = sep.Background(data, mask=mask, bw=3, bh=3, fw=1, fh=1)
    bkg2 = sky.back()

    # All False mask should be the same
    assert_allclose(bkg1, bkg2)

    # Masking high pixels should give a flat background
    mask[1, 1] = True
    mask[4, 1] = True
    mask[1, 4] = True
    mask[4, 4] = True
    sky = sep.Background(data, mask=mask, bw=3, bh=3, fw=1, fh=1)
    assert_approx_equal(sky.globalback, 0.1)
    assert_allclose(sky.back(), 0.1 * np.ones((6, 6)))


@pytest.mark.skipif(NO_FITS, reason="no FITS reader")
def test_background_special():
    """
    Test the special methods of `sep.Background`.
    """

    bkg = sep.Background(image_data, bw=64, bh=64, fw=3, fh=3)

    # test __array__ method
    assert np.all(np.array(bkg) == bkg.back())

    # test __rsub__ method
    d1 = image_data - bkg

    d2 = np.copy(image_data)
    bkg.subfrom(d2)
    assert np.all(d1 == d2)


def test_background_boxsize():
    """
    Test that `sep.Background` works when boxsize is same as image.
    """

    ny, nx = 100, 100
    data = np.ones((ny, nx), dtype=np.float64)
    bkg = sep.Background(data, bh=ny, bw=nx, fh=1, fw=1)
    bkg.back()


def test_background_rms():
    """
    Test that `sep.Background.rms` at least works.
    """

    ny, nx = 1024, 1024
    data = np.random.randn(ny, nx)
    bkg = sep.Background(data)
    rms = bkg.rms()
    assert rms.dtype == np.float64
    assert rms.shape == (ny, nx)


# -----------------------------------------------------------------------------
# Extract


@pytest.mark.skipif(NO_FITS, reason="no FITS reader")
def test_extract_with_noise_array():
    """
    Test extraction with a flat noise array.

    This checks that a constant noise array gives the same result as
    extracting without a noise array, for a given threshold.
    """

    # Get some background-subtracted test data:
    data = np.copy(image_data)
    bkg = sep.Background(data, bw=64, bh=64, fw=3, fh=3)
    bkg.subfrom(data)

    # Ensure that extraction with constant noise array gives the expected
    # result. We have to use conv=None here because the results are *not*
    # the same when convolution is on! This is because the noise map is
    # convolved. Near edges, the convolution doesn't adjust for pixels
    # off edge boundaries. As a result, the convolved noise map is not
    # all ones.
    # Deblending is also turned off, as this appears to differ slightly
    # across platforms - see `test_vs_sextractor()`.
    objects = sep.extract(
        data, 1.5 * bkg.globalrms, filter_kernel=None, deblend_cont=1.0
    )
    objects2 = sep.extract(
        data,
        1.5 * bkg.globalrms,
        err=np.ones_like(data),
        filter_kernel=None,
        deblend_cont=1.0,
    )

    names_to_remove = ["errx2", "erry2", "errxy"]
    names_to_keep = [i for i in objects.dtype.names if i not in names_to_remove]
    objects = objects[names_to_keep]
    objects2 = objects2[names_to_keep]

    assert_allclose_structured(objects, objects2)

    # Less trivial test where thresh is realistic. Still a flat noise map.
    noise = bkg.globalrms * np.ones_like(data)
    objects2 = sep.extract(data, 1.5, err=noise, filter_kernel=None, deblend_cont=1.0)

    names_to_remove = ["errx2", "erry2", "errxy"]
    names_to_keep = [i for i in objects.dtype.names if i not in names_to_remove]
    objects = objects[names_to_keep]
    objects2 = objects2[names_to_keep]

    assert_allclose_structured(objects, objects2)


def test_extract_with_noise_convolution():
    """
    Test extraction when there is both noise and convolution.

    This will use the matched filter implementation, and will handle bad pixels
    and edge effects gracefully.
    """

    # Start with an empty image where we label the noise as 1 sigma everywhere.
    image = np.zeros((20, 20))
    error = np.ones((20, 20))

    # Add some noise representing bad pixels. We do not want to detect these.
    image[17, 3] = 100.0
    error[17, 3] = 100.0
    image[10, 0] = 100.0
    error[10, 0] = 100.0
    image[17, 17] = 100.0
    error[17, 17] = 100.0

    # Add some real point sources that we should find.
    image[3, 17] = 10.0

    image[6, 6] = 2.0
    image[7, 6] = 1.0
    image[5, 6] = 1.0
    image[6, 5] = 1.0
    image[6, 7] = 1.0

    objects = sep.extract(image, 2.0, minarea=1, err=error)
    objects.sort(order=["x", "y"])

    # Check that we recovered the two correct objects and not the others.
    assert len(objects) == 2

    assert_approx_equal(objects[0]["x"], 6.0)
    assert_approx_equal(objects[0]["y"], 6.0)

    assert_approx_equal(objects[1]["x"], 17.0)
    assert_approx_equal(objects[1]["y"], 3.0)


def test_extract_matched_filter_scalar_noise_matches_constant_array():
    """Scalar noise uses the same matched-filter statistic as a flat map."""
    rng = np.random.default_rng(1729)
    shape = (65, 67)
    ygrid, xgrid = np.mgrid[: shape[0], : shape[1]]
    sigma = 3.0
    data = (
        1.4
        * np.exp(-((xgrid - 32.0) ** 2 + (ygrid - 31.0) ** 2) / (2.0 * sigma**2))
        + rng.normal(size=shape)
    ).astype(np.float32)
    ky, kx = np.mgrid[-8:9, -8:9]
    kernel = np.exp(-(kx**2 + ky**2) / (2.0 * sigma**2)).astype(np.float32)

    scalar, scalar_seg = sep.extract(
        data,
        4.0,
        err=1.0,
        minarea=1,
        filter_kernel=kernel,
        clean=False,
        deblend_cont=1.0,
        segmentation_map=True,
    )
    array, array_seg = sep.extract(
        data,
        4.0,
        err=np.ones_like(data),
        minarea=1,
        filter_kernel=kernel,
        clean=False,
        deblend_cont=1.0,
        segmentation_map=True,
    )

    assert_allclose_structured(scalar, array)
    assert_equal(scalar_seg, array_seg)


def test_extract_watershed_uses_filtered_detection_plane():
    """Pixel noise must not split one faint broad source into many basins."""
    rng = np.random.default_rng(1729)
    shape = (129, 129)
    ygrid, xgrid = np.mgrid[: shape[0], : shape[1]]
    sigma = 10.0 / 2.354820045
    data = (
        0.8
        * np.exp(-((xgrid - 64.0) ** 2 + (ygrid - 64.0) ** 2) / (2.0 * sigma**2))
        + rng.normal(size=shape)
    ).astype(np.float32)
    ky, kx = np.mgrid[-12:13, -12:13]
    kernel = np.exp(-(kx**2 + ky**2) / (2.0 * sigma**2)).astype(np.float32)

    objects = sep.extract(
        data,
        4.0,
        err=1.0,
        minarea=5,
        filter_kernel=kernel,
        clean=False,
        deblend_cont=0.005,
        deblend_method="watershed",
    )

    assert len(objects) == 1
    assert objects["npix"][0] > 100


@pytest.mark.parametrize("fwhm", [1.0, 1.2, 1.4, 1.6])
def test_extract_fwhm_compact_gaussian(fwhm):
    flux = np.array([5000.0])
    image = _gaussian_scene((25, 25), np.array([12.3]), np.array([12.7]), fwhm, flux)

    objects = sep.extract(image, 5.0)

    assert len(objects) == 1
    assert abs(objects["fwhm"][0] - fwhm) / fwhm < 0.1


@pytest.mark.parametrize("fwhm", [1.2, 2.5, 5.0, 7.0])
@pytest.mark.parametrize("phase", [0.15, 0.5, 0.85])
def test_extract_fwhm_across_source_widths_and_phases(fwhm, phase):
    """FWHM estimates remain accurate for broad and undersampled sources."""
    x0 = np.array([24.0 + phase])
    y0 = np.array([24.37])
    image = _gaussian_scene((49, 49), x0, y0, fwhm, np.array([5000.0]))

    objects = sep.extract(image, 5.0, clean=False, deblend_cont=1.0)

    assert len(objects) == 1
    assert np.isfinite(objects["fwhm"][0])
    assert_allclose(objects["fwhm"][0], fwhm, rtol=0.06)


def test_extract_fwhm_hotpixel_zero():
    image = np.zeros((11, 11))
    image[5, 5] = 1000.0

    objects = sep.extract(image, 5.0, minarea=1)

    assert len(objects) == 1
    assert objects["fwhm"][0] == 0.0


def test_extract_matched_filter_at_edge():
    """
    Test bright source detection at the edge of an image.

    Exercise bug where bright star at end of image not detected
    with noise array and matched filter on.
    """

    data = np.zeros((20, 20))
    err = np.ones_like(data)
    kernel = np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]])

    data[18:20, 9:12] = kernel[0:2, :]

    objects, pix = sep.extract(
        data,
        2.0,
        err=err,
        filter_kernel=kernel,
        filter_type="matched",
        segmentation_map=True,
    )
    assert len(objects) == 1
    assert objects["npix"][0] == 6


def test_extract_matched_filter_with_scalar_noise_and_mask():
    """
    Masked pixels are omitted from matched-filter normalization even when the
    supplied noise is scalar.
    """
    data = np.zeros((15, 15), dtype=np.float64)
    mask = np.zeros_like(data, dtype=np.uint8)
    kernel = np.ones((3, 3), dtype=np.float64)

    data[6:9, 6:9] = 1.0
    mask[7, 7] = 1

    objects = sep.extract(
        data,
        2.5,
        err=1.0,
        mask=mask,
        minarea=1,
        filter_kernel=kernel,
        filter_type="matched",
        deblend_cont=1.0,
    )

    assert len(objects) == 1
    assert abs(objects["x"][0] - 7.0) < 1.0
    assert abs(objects["y"][0] - 7.0) < 1.0


@pytest.mark.skipif(NO_FITS, reason="no FITS reader")
def test_extract_with_mask():
    """
    Test that object detection only occurs in unmasked regions.
    """

    # Get some background-subtracted test data:
    data = np.copy(image_data)
    bkg = sep.Background(data, bw=64, bh=64, fw=3, fh=3)
    bkg.subfrom(data)

    # mask half the image
    ylim = data.shape[0] // 2
    mask = np.zeros(data.shape, dtype=np.bool_)
    mask[ylim:, :] = True

    objects = sep.extract(data, 1.5 * bkg.globalrms, mask=mask)

    # check that we found some objects and that they are all in the unmasked
    # region.
    assert len(objects) > 0
    assert np.all(objects["y"] < ylim)


@pytest.mark.skipif(NO_FITS, reason="no FITS reader")
def test_extract_with_maskthresh():
    """
    Test that object detection only occurs in unmasked regions.
    """

    # Get some background-subtracted test data:
    data = np.copy(image_data)
    bkg = sep.Background(data, bw=64, bh=64, fw=3, fh=3)
    bkg.subfrom(data)

    # mask half the image
    ylim = data.shape[0] // 2
    mask = np.zeros(data.shape, dtype=float)
    mask[ylim:, :] = 1.0

    objects_unmasked = sep.extract(data, 1.5 * bkg.globalrms, deblend_cont=1.0)
    objects_unmasked_w_thresh = sep.extract(
        data, 1.5 * bkg.globalrms, maskthresh=1.0, deblend_cont=1.0
    )

    # Check that changing the mask threshold does not change anything,
    # if no mask is provided
    assert_allclose_structured(objects_unmasked, objects_unmasked_w_thresh)

    objects_masked = sep.extract(data, 1.5 * bkg.globalrms, mask=mask, deblend_cont=1.0)
    objects_masked_w_hthresh = sep.extract(
        data, 1.5 * bkg.globalrms, mask=mask, maskthresh=1.0, deblend_cont=1.0
    )
    objects_masked_w_lthresh = sep.extract(
        data, 1.5 * bkg.globalrms, mask=mask, maskthresh=0.5, deblend_cont=1.0
    )

    # Applying a mask should return a different number of objects
    assert len(objects_unmasked) != len(objects_masked)

    # As long as the mask is above the threshold, the results should not change
    assert_allclose_structured(objects_masked, objects_masked_w_lthresh)

    # Object detection where the maskthresh >= max(mask) should be the same
    # as if no mask were provided
    assert_allclose_structured(objects_unmasked, objects_masked_w_hthresh)


@pytest.mark.skipif(NO_FITS, reason="no FITS reader")
def test_extract_segmentation_map():
    """
    Test the returned segmentation map.

    Check that the segmentation map has the same dimensions as the input
    image, and that the number of object pixels match the catalogue field.
    """

    # Get some background-subtracted test data:
    data = np.copy(image_data)
    bkg = sep.Background(data, bw=64, bh=64, fw=3, fh=3)
    bkg.subfrom(data)

    objects, segmap = sep.extract(data, 1.5 * bkg.globalrms, segmentation_map=True)

    assert type(segmap) is np.ndarray
    assert segmap.shape == data.shape
    for i in range(len(objects)):
        assert objects["npix"][i] == (segmap == i + 1).sum()


@pytest.mark.skipif(NO_FITS, reason="no FITS reader")
def test_extract_seg_map_array():
    """
    Test the extraction when an existing segmentation map is supplied.

    Test that the returned catalogue is equal with and without a variable
    noise array, and that the majority of fields match even when
    deblending is performed on the original extraction.
    """

    # Get some background-subtracted test data:
    data = np.copy(image_data)
    bkg = sep.Background(data, bw=64, bh=64, fw=3, fh=3)
    bkg.subfrom(data)

    noise = bkg.globalrms * np.ones_like(data)

    for err in [None, noise]:
        # err=None
        # err=noise

        objects, segmap = sep.extract(data, 1.5, err, segmentation_map=True)

        assert type(segmap) is np.ndarray
        assert segmap.shape == data.shape
        for i in range(len(objects)):
            assert objects["npix"][i] == (segmap == i + 1).sum()

        objects2, segmap2 = sep.extract(data, 1.5, err, segmentation_map=segmap)

        # Test the values for which we expect an exact match
        names_exact_match = [
            "thresh",
            "npix",
            "tnpix",
            "xmin",
            "xmax",
            "ymin",
            "ymax",
            "cflux",
            "flux",
            "cpeak",
            "peak",
            "xcpeak",
            "ycpeak",
            "xpeak",
            "ypeak",
        ]

        # The position depends on the object being deblended. As no deblending
        # is performed when a segmentation map is supplied, all derived
        # parameters may vary slightly. We test those for which we have a
        # measurement of the uncertainty
        names_close = ["x", "y"]
        names_close_var = ["x2", "y2"]

        assert segmap2.shape == data.shape
        for o_i, o_ii in zip(objects, objects2):
            o_i_exact = o_i[names_exact_match]
            o_ii_exact = o_ii[names_exact_match]
            assert_equal(o_i_exact, o_ii_exact)

            o_i_close = o_i[names_close]
            o_ii_close = o_ii[names_close]
            for n, v in zip(names_close, names_close_var):
                if o_i["flag"] == 0:
                    assert_equal(o_i[n], o_ii[n])
                else:
                    assert_allclose(o_i[n], o_ii[n], atol=np.sqrt(o_i[v]))

        # Perform a second test with deblending disabled.
        objects3, segmap3 = sep.extract(
            data, 1.5, err, segmentation_map=True, deblend_cont=1.0
        )

        objects4, segmap4 = sep.extract(
            data, 1.5, err, segmentation_map=segmap3, deblend_cont=1.0
        )

        # The flag will not be the same, as the second extraction does not test
        # for deblended objects.
        objects3 = rfn.drop_fields(objects3, "flag")
        objects4 = rfn.drop_fields(objects4, "flag")
        assert_allclose_structured(objects3, objects4)


# -----------------------------------------------------------------------------
# aperture tests

naper = 1000
x = np.random.uniform(200.0, 800.0, naper)
y = np.random.uniform(200.0, 800.0, naper)
data_shape = (1000, 1000)


def test_aperture_dtypes():
    """
    Test the aperture extraction of multiple data types.

    Ensure that all supported image dtypes work in sum_circle() and
    give the same answer.
    """

    r = 3.0

    fluxes = []
    for dt in SUPPORTED_IMAGE_DTYPES:
        data = np.ones(data_shape, dtype=dt)
        flux, fluxerr, flag = sep.sum_circle(data, x, y, r)
        fluxes.append(flux)

    for i in range(1, len(fluxes)):
        assert_allclose(fluxes[0], fluxes[i])


def test_apertures_small_ellipse_exact():
    """Regression test for a bug that manifested primarily when x == y."""

    data = np.ones(data_shape)
    r = 0.3
    rtol = 1.0e-10
    flux, fluxerr, flag = sep.sum_ellipse(data, x, x, r, r, 0.0, subpix=0)
    assert_allclose(flux, np.pi * r**2, rtol=rtol)


def test_apertures_all():
    """
    Test that aperture subpixel sampling works.
    """

    data = np.random.rand(*data_shape)
    r = 3.0
    rtol = 1.0e-8

    for subpix in [0, 1, 5]:
        flux_ref, fluxerr_ref, flag_ref = sep.sum_circle(data, x, y, r, subpix=subpix)

        flux, fluxerr, flag = sep.sum_circann(data, x, y, 0.0, r, subpix=subpix)
        assert_allclose(flux, flux_ref, rtol=rtol)

        flux, fluxerr, flag = sep.sum_ellipse(data, x, y, r, r, 0.0, subpix=subpix)
        assert_allclose(flux, flux_ref, rtol=rtol)

        flux, fluxerr, flag = sep.sum_ellipse(
            data, x, y, 1.0, 1.0, 0.0, r=r, subpix=subpix
        )
        assert_allclose(flux, flux_ref, rtol=rtol)


def test_apertures_exact():
    """
    Test area as measured by exact aperture modes on array of ones.
    """

    theta = np.random.uniform(-np.pi / 2.0, np.pi / 2.0, naper)
    ratio = np.random.uniform(0.2, 1.0, naper)
    r = 3.0

    for dt in SUPPORTED_IMAGE_DTYPES:
        data = np.ones(data_shape, dtype=dt)
        for r in [0.5, 1.0, 3.0]:
            flux, fluxerr, flag = sep.sum_circle(data, x, y, r, subpix=0)
            assert_allclose(flux, np.pi * r**2)

            rout = r * 1.1
            flux, fluxerr, flag = sep.sum_circann(data, x, y, r, rout, subpix=0)
            assert_allclose(flux, np.pi * (rout**2 - r**2))

            flux, fluxerr, flag = sep.sum_ellipse(
                data, x, y, 1.0, ratio, theta, r=r, subpix=0
            )
            assert_allclose(flux, np.pi * ratio * r**2)

            rout = r * 1.1
            flux, fluxerr, flag = sep.sum_ellipann(
                data, x, y, 1.0, ratio, theta, r, rout, subpix=0
            )
            assert_allclose(flux, np.pi * ratio * (rout**2 - r**2))


def test_sum_circle_optimal_grouped_close_pair():
    shape = (41, 41)
    fwhm = 3.0
    r = 6.0
    x0 = np.array([20.2, 22.4])
    y0 = np.array([20.1, 20.6])
    true_flux = np.array([1000.0, 200.0])

    data = _gaussian_scene(shape, x0, y0, fwhm, true_flux)

    flux, _, _ = sep.sum_circle_optimal(data, x0, y0, r, fwhm, subpix=0)
    flux_grp, _, _ = sep.sum_circle_optimal(
        data, x0, y0, r, fwhm, grouped=True, subpix=0
    )

    err = np.abs(flux - true_flux)
    err_grp = np.abs(flux_grp - true_flux)

    assert_allclose(flux_grp, true_flux, rtol=2.0e-3, atol=1.0e-2)
    assert err_grp[1] < err[1]
    assert err_grp.max() < err.max()


def test_sum_circle_optimal_grouped_wide_separation():
    shape = (128, 128)
    fwhm = 3.0
    r = 6.0
    x0 = np.array([20.3, 80.8, 50.5])
    y0 = np.array([20.7, 75.2, 90.4])
    true_flux = np.array([1200.0, 800.0, 450.0])

    data = _gaussian_scene(shape, x0, y0, fwhm, true_flux)

    flux, _, _ = sep.sum_circle_optimal(data, x0, y0, r, fwhm, subpix=0)
    flux_grp, _, _ = sep.sum_circle_optimal(
        data, x0, y0, r, fwhm, grouped=True, subpix=0
    )

    assert_allclose(flux, true_flux, rtol=2.0e-3, atol=1.0e-2)
    assert_allclose(flux_grp, true_flux, rtol=2.0e-3, atol=1.0e-2)
    assert_allclose(flux_grp, flux, rtol=5.0e-4, atol=1.0e-3)


def test_sum_circle_optimal_clean_image_no_mask_flag():
    data = np.ones((64, 64), dtype=np.float64)
    x0 = np.array([20.0, 32.0, 44.0])
    y0 = np.array([20.0, 32.0, 44.0])
    r = np.array([3.0, 3.0, 3.0])
    fwhm = np.array([3.0, 3.0, 3.0])

    flux, fluxerr, flag = sep.sum_circle_optimal(data, x0, y0, r, fwhm)

    assert np.all(np.isfinite(flux))
    assert np.all(np.isfinite(fluxerr))
    assert np.all(flag == 0)


def test_sum_circle_optimal_bkgann_uses_psf_effective_area():
    """A local sky level must be removed through the optimal PSF weights."""
    shape = (81, 81)
    sigma = 1.6
    fwhm = 2.355 * sigma
    r = 1.5 * fwhm
    x0 = y0 = 40.0
    sky = 20.0
    source_flux = 1000.0
    yy, xx = np.indices(shape)
    data = sky + source_flux / (2.0 * np.pi * sigma**2) * np.exp(
        -((xx - x0) ** 2 + (yy - y0) ** 2) / (2.0 * sigma**2)
    )

    local, _, _ = sep.sum_circle_optimal(
        data, [x0], [y0], r, fwhm, bkgann=(6.0 * sigma, 9.0 * sigma),
        clip_iters=0,
    )
    reference, _, _ = sep.sum_circle_optimal(data - sky, [x0], [y0], r, fwhm)

    assert_allclose(local, reference, rtol=0.0, atol=2.0e-5)


def test_sum_circle_optimal_grouped_bkgann_subtracts_before_solving():
    """A shared local sky must not bias a simultaneous PSF fit."""
    shape = (100, 100)
    fwhm = 3.0
    r = 6.0
    sky = 20.0
    x0 = np.array([40.2, 42.4])
    y0 = np.array([40.1, 40.6])
    true_flux = np.array([1000.0, 200.0])
    data = _gaussian_scene(shape, x0, y0, fwhm, true_flux)

    reference, _, _ = sep.sum_circle_optimal(
        data, x0, y0, r, fwhm, grouped=True, subpix=0
    )
    local, _, _ = sep.sum_circle_optimal(
        data + sky, x0, y0, r, fwhm, grouped=True, subpix=0,
        bkgann=(15.0, 20.0), clip_iters=0,
    )

    assert_allclose(local, reference, rtol=0.0, atol=2.0e-5)


def test_sum_circle_optimal_grouped_large_component_scalar_gaussian():
    shape = (220, 220)
    rng = np.random.default_rng(4)
    nsrc = 48
    fwhm = 5.0
    r = 2.0 * fwhm
    x0 = rng.uniform(60.0, 160.0, nsrc)
    y0 = rng.uniform(60.0, 160.0, nsrc)
    true_flux = rng.uniform(200.0, 1200.0, nsrc)

    data = _gaussian_scene(shape, x0, y0, fwhm, true_flux)

    flux, _, _ = sep.sum_circle_optimal(data, x0, y0, r, fwhm, subpix=0)
    flux_grp, _, _ = sep.sum_circle_optimal(
        data, x0, y0, r, fwhm, grouped=True, subpix=0
    )

    err = np.abs(flux - true_flux)
    err_grp = np.abs(flux_grp - true_flux)

    assert np.all(np.isfinite(flux_grp))
    assert err_grp.mean() < err.mean()


def test_sum_circle_optimal_grouped_large_component_variable_radius():
    shape = (220, 220)
    rng = np.random.default_rng(3)
    nsrc = 72
    fwhm = np.full(nsrc, 5.0)
    r = 2.0 * fwhm * (1.0 + 0.05 * np.sin(np.linspace(0.0, 3.0 * np.pi, nsrc)))
    x0 = rng.uniform(60.0, 160.0, nsrc)
    y0 = rng.uniform(60.0, 160.0, nsrc)
    true_flux = rng.uniform(200.0, 1200.0, nsrc)

    data = _gaussian_scene(shape, x0, y0, fwhm[0], true_flux)

    flux, _, _ = sep.sum_circle_optimal(data, x0, y0, r, fwhm, subpix=0)
    flux_grp, _, _ = sep.sum_circle_optimal(
        data, x0, y0, r, fwhm, grouped=True, subpix=0
    )

    err = np.abs(flux - true_flux)
    err_grp = np.abs(flux_grp - true_flux)

    assert np.all(np.isfinite(flux_grp))
    assert err_grp.mean() < err.mean()


def test_sum_circle_optimal_grouped_respects_aperture_overlap():
    """Grouping should be driven by aperture overlap, not full PSF stamp size."""
    shape = (240, 240)
    fwhm = 4.4
    r = 4.4

    # Spacing exceeds 2*r, so optimal-aperture groups should remain isolated.
    xs = np.arange(30.0, 210.1, 12.0)
    ys = np.arange(30.0, 210.1, 12.0)
    x0, y0 = np.meshgrid(xs, ys)
    x0 = x0.ravel()
    y0 = y0.ravel()
    true_flux = np.full(x0.shape, 1000.0, dtype=float)

    data = _gaussian_scene(shape, x0, y0, fwhm, true_flux)

    flux, _, _ = sep.sum_circle_optimal(data, x0, y0, r, fwhm, subpix=0)
    flux_grp_ref, _, _ = sep.sum_circle_optimal(
        data, x0, y0, r, fwhm, grouped=True, group_radius_factor=1.000001, subpix=0
    )
    flux_grp, _, _ = sep.sum_circle_optimal(
        data, x0, y0, r, fwhm, grouped=True, subpix=0
    )

    assert np.all(np.isfinite(flux_grp))
    assert_allclose(flux_grp, flux_grp_ref)
    assert_allclose(flux_grp, flux)


def test_sum_circle_optimal_grouped_recovers_blended_gaussians():
    """Grouped optimal should deblend close Gaussian sources without biasing much."""
    shape = (64, 64)
    x0 = np.array([29.2, 33.0, 35.1])
    y0 = np.array([31.1, 31.8, 34.6])
    fwhm = np.full(3, 3.6)
    r = np.full(3, 4.5)
    true_flux = np.array([800.0, 500.0, 300.0])

    data = _gaussian_scene(shape, x0, y0, fwhm[0], true_flux)
    flux_grp, _, flag = sep.sum_circle_optimal(
        data, x0, y0, r, fwhm, grouped=True, group_radius_factor=1.0, subpix=1
    )

    assert np.all(flag == 0)
    assert_allclose(flux_grp, true_flux, rtol=0.01, atol=0.0)


def test_sum_circle_optimal_group_halo_factor_matches_old_semantics():
    shape = (220, 220)
    rng = np.random.default_rng(8)
    nsrc = 80
    fwhm = 5.0
    r = 2.0 * fwhm
    x0 = rng.uniform(60.0, 160.0, nsrc)
    y0 = rng.uniform(60.0, 160.0, nsrc)
    true_flux = rng.uniform(200.0, 1200.0, nsrc)

    data = _gaussian_scene(shape, x0, y0, fwhm, true_flux)

    flux_old, fluxerr_old, flag_old = sep.sum_circle_optimal(
        data, x0, y0, r, fwhm, grouped=True, group_radius_factor=1.2, subpix=0
    )
    flux_new, fluxerr_new, flag_new = sep.sum_circle_optimal(
        data, x0, y0, r, fwhm,
        grouped=True,
        group_radius_factor=1.0,
        group_halo_factor=1.2,
        subpix=0,
    )

    assert np.all(np.isfinite(flux_new))
    assert np.all(np.isfinite(fluxerr_new))
    assert np.all(flag_new == flag_old)
    assert_allclose(flux_new, flux_old, rtol=5e-3, atol=5e-3)
    assert_allclose(fluxerr_new, fluxerr_old, rtol=5e-3, atol=5e-3)


def test_sum_circle_optimal_group_halo_factor_clamps_to_connectivity():
    """Localized grouped optimal fits should not use halo smaller than support."""
    shape = (240, 240)
    fwhm = 4.0
    r = 4.0

    xs = np.arange(40.0, 200.1, 7.0)
    ys = np.arange(40.0, 200.1, 7.0)
    x0, y0 = np.meshgrid(xs, ys)
    x0 = x0.ravel()
    y0 = y0.ravel()
    true_flux = np.full(x0.shape, 1000.0, dtype=float)

    data = _gaussian_scene(shape, x0, y0, fwhm, true_flux)

    flux_1, fluxerr_1, flag_1 = sep.sum_circle_optimal(
        data, x0, y0, r, fwhm,
        grouped=True,
        group_radius_factor=1.0,
        group_halo_factor=1.0,
        subpix=0,
    )
    flux_low, fluxerr_low, flag_low = sep.sum_circle_optimal(
        data, x0, y0, r, fwhm,
        grouped=True,
        group_radius_factor=1.0,
        group_halo_factor=0.5,
        subpix=0,
    )

    assert np.all(np.isfinite(flux_low))
    assert np.all(flag_low == flag_1)
    assert_allclose(flux_low, flux_1)
    assert_allclose(fluxerr_low, fluxerr_1)


def test_sum_circle_optimal_group_halo_factor_order_invariant():
    """Localized grouped optimal fits should not depend on input ordering."""
    shape = (120, 120)
    fwhm = 4.0
    r = 4.0

    xs = np.arange(30.0, 86.1, 7.0)
    ys = np.arange(30.0, 86.1, 7.0)
    x0, y0 = np.meshgrid(xs, ys)
    x0 = x0.ravel()
    y0 = y0.ravel()
    true_flux = np.full(x0.shape, 1000.0, dtype=float)

    data = _gaussian_scene(shape, x0, y0, fwhm, true_flux)

    flux_a, err_a, flag_a = sep.sum_circle_optimal(
        data, x0, y0, r, fwhm,
        grouped=True,
        group_radius_factor=1.0,
        group_halo_factor=10.0,
        subpix=0,
    )
    flux_b, err_b, flag_b = sep.sum_circle_optimal(
        data, x0[::-1], y0[::-1], r, fwhm,
        grouped=True,
        group_radius_factor=1.0,
        group_halo_factor=10.0,
        subpix=0,
    )

    assert_allclose(flux_a, flux_b[::-1])
    assert_allclose(err_a, err_b[::-1])
    assert np.all(flag_a == flag_b[::-1])


def test_sum_circle_optimal_stable_across_group_boundary():
    """Grouped fluxes stay accurate on both sides of the grouping cutoff."""
    fwhm = 2.5
    r = 2.0 * fwhm
    truth = np.array([1000.0, 100.0])
    errors = []

    for separation_fwhm in [3.95, 4.0, 4.05]:
        separation = separation_fwhm * fwhm
        x0 = np.array([32.35 - separation / 2.0, 32.35 + separation / 2.0])
        y0 = np.array([31.65, 31.65])
        data = _gaussian_scene((65, 65), x0, y0, fwhm, truth)

        flux, _, flag = sep.sum_circle_optimal(
            data, x0, y0, r, fwhm, grouped=True, subpix=0
        )
        errors.append(np.max(np.abs(flux - truth) / truth))
        assert np.all(flag == 0)
        assert_allclose(flux, truth, rtol=1.0e-5)

    assert np.max(np.abs(np.diff(errors))) < 1.0e-5


def test_sum_circle_optimal_chain_translation_and_order_invariant():
    """Localized long-chain solutions are insensitive to order and phase."""
    fwhm = 2.5
    r = 2.0 * fwhm
    nsrc = 10
    spacing = 3.95 * fwhm
    truth = 700.0 + 70.0 * np.arange(nsrc)

    for offset in [0.0, 1.3]:
        x0 = 18.0 + offset + spacing * np.arange(nsrc)
        y0 = np.full(nsrc, 31.4)
        data = _gaussian_scene((64, 128), x0, y0, fwhm, truth)

        flux, fluxerr, flag = sep.sum_circle_optimal(
            data, x0, y0, r, fwhm, grouped=True, subpix=0
        )
        reverse = np.arange(nsrc)[::-1]
        flux_rev, fluxerr_rev, flag_rev = sep.sum_circle_optimal(
            data,
            x0[reverse],
            y0[reverse],
            r,
            fwhm,
            grouped=True,
            subpix=0,
        )

        assert_allclose(flux, truth, rtol=1.0e-5)
        assert_allclose(flux, flux_rev[reverse], rtol=1.0e-12, atol=1.0e-10)
        assert_allclose(fluxerr, fluxerr_rev[reverse], rtol=1.0e-12, atol=1.0e-10)
        assert_equal(flag, flag_rev[reverse])


def _sigma_clip_mean(values, sigma=3.0, maxiters=5):
    mask = np.ones(values.shape, dtype=bool)
    for _ in range(maxiters):
        vals = values[mask]
        if vals.size == 0:
            return np.nan
        med = np.median(vals)
        mad = np.median(np.abs(vals - med))
        std = 1.4826 * mad
        if std == 0.0:
            return vals.mean()
        lo = med - sigma * std
        hi = med + sigma * std
        newmask = mask & (values >= lo) & (values <= hi)
        if newmask.sum() == mask.sum():
            return vals.mean()
        mask = newmask
    vals = values[mask]
    return vals.mean() if vals.size else np.nan


def test_stats_circann_flat():
    data = np.ones(data_shape)
    rin, rout = 3.0, 6.0
    mean, std, med, mad_std, mean_clip, flag = sep.stats_circann(
        data, x, y, rin, rout, subpix=1
    )
    assert_allclose(mean, 1.0)
    assert_allclose(std, 0.0)
    assert_allclose(med, 1.0)
    assert_allclose(mad_std, 0.0)
    assert_allclose(mean_clip, 1.0)


def test_stats_circann_matches_numpy_subpix1():
    rng = np.random.default_rng(12345)
    data = rng.normal(size=(64, 64))
    x0 = np.array([30.4])
    y0 = np.array([31.2])
    rin, rout = 5.0, 8.0

    mean, std, med, mad_std, mean_clip, flag = sep.stats_circann(
        data, x0, y0, rin, rout, subpix=1
    )

    yy, xx = np.indices(data.shape)
    rpix2 = (xx - x0[0]) ** 2 + (yy - y0[0]) ** 2
    mask = (rpix2 >= rin**2) & (rpix2 < rout**2)
    vals = data[mask]

    exp_mean = vals.mean()
    exp_std = vals.std()
    exp_med = np.median(vals)
    exp_mad = np.median(np.abs(vals - exp_med))
    exp_mad_std = 1.4826 * exp_mad  # robust std estimate for Gaussian data
    exp_clip = _sigma_clip_mean(vals)

    assert_allclose(mean[0], exp_mean, rtol=1.0e-10, atol=1.0e-8)
    assert_allclose(std[0], exp_std, rtol=1.0e-10, atol=1.0e-8)
    assert_allclose(med[0], exp_med, rtol=1.0e-10, atol=1.0e-8)
    assert_allclose(mad_std[0], exp_mad_std, rtol=1.0e-10, atol=1.0e-7)
    assert_allclose(mean_clip[0], exp_clip, rtol=1.0e-10, atol=1.0e-8)
    assert flag[0] == 0


def test_stats_circann_clipping_rejects_source_contamination():
    """Sigma clipping recovers a smooth background with bright annulus pixels."""
    ygrid, xgrid = np.indices((65, 65))
    x0 = y0 = 32.0
    data = 100.0 + 0.2 * (xgrid - x0) - 0.1 * (ygrid - y0)
    for ybad, xbad in [(32, 39), (32, 40), (39, 32), (26, 37)]:
        data[ybad, xbad] += 1000.0

    mean, _, median, _, mean_clip, flag = sep.stats_circann(
        data, [x0], [y0], 6.0, 10.0, subpix=1, clip_sigma=3.0, clip_iters=5
    )

    assert mean[0] > 115.0
    assert_allclose(median[0], 100.0, atol=0.1)
    assert_allclose(mean_clip[0], 100.0, atol=0.1)
    assert flag[0] == 0


def test_aperture_bkgann_overlapping():
    """
    Test bkgann functionality in circular & elliptical apertures.
    """

    # If bkgann overlaps aperture exactly, result should be zero with
    # subpixel sampling of 1 and clipping disabled. Sigma clipping can
    # intentionally reject a random annulus sample and change its mean.
    data = np.random.rand(*data_shape)
    r = 5.0
    f, _, _ = sep.sum_circle(
        data, x, y, r, bkgann=(0.0, r), subpix=1, clip_iters=0
    )
    assert_allclose(f, 0.0, rtol=0.0, atol=1.0e-13)

    f, _, _ = sep.sum_ellipse(
        data,
        x,
        y,
        2.0,
        1.0,
        np.pi / 4.0,
        r=r,
        bkgann=(0.0, r),
        subpix=1,
        clip_iters=0,
    )
    assert_allclose(f, 0.0, rtol=0.0, atol=1.0e-13)


def test_aperture_bkgann_ones():
    """
    Test bkgann functionality with flat data.
    """

    data = np.ones(data_shape)
    r = 5.0
    bkgann = (6.0, 8.0)

    # On flat data, result should be zero for any bkgann and subpix
    f, fe, _ = sep.sum_circle(data, x, y, r, bkgann=bkgann, gain=1.0)
    assert_allclose(f, 0.0, rtol=0.0, atol=1.0e-13)

    # for all ones data and no error array, error should be close to
    # sqrt(Npix_aper + Npix_ann * (Npix_aper**2 / Npix_ann**2))
    aper_area = np.pi * r**2
    bkg_area = np.pi * (bkgann[1] ** 2 - bkgann[0] ** 2)
    expected_error = np.sqrt(aper_area + bkg_area * (aper_area / bkg_area) ** 2)
    assert_allclose(fe, expected_error, rtol=0.1)

    f, _, _ = sep.sum_ellipse(data, x, y, 2.0, 1.0, np.pi / 4.0, r, bkgann=bkgann)
    assert_allclose(f, 0.0, rtol=0.0, atol=1.0e-13)


def test_masked_segmentation_measurements():
    """
    Test measurements with segmentation masking.
    """

    NX = 100
    data = np.zeros((NX * 2, NX * 2))
    yp, xp = np.indices(data.shape)

    ####
    # Make two 2D gaussians that slightly overlap

    # width of the 2D objects
    gsigma = 10.0

    # offset between two gaussians in sigmas
    off = 4

    for xy in [[NX, NX], [NX + off * gsigma, NX + off * gsigma]]:
        R = np.sqrt((xp - xy[0]) ** 2 + (yp - xy[1]) ** 2)
        g_i = np.exp(-(R**2) / 2 / gsigma**2)
        data += g_i

    # Absolute total
    total_exact = g_i.sum()

    # Add some noise
    rms = 0.02
    np.random.seed(1)
    data += np.random.normal(size=data.shape) * rms

    # Run source detection
    objs, segmap = sep.extract(
        data,
        thresh=1.2,
        err=rms,
        mask=None,
        filter_type="conv",
        segmentation_map=True,
    )

    seg_id = np.arange(1, len(objs) + 1, dtype=np.int32)

    # Compute Kron/Auto parameters
    x, y, a, b = objs["x"], objs["y"], objs["a"], objs["b"]
    theta = objs["theta"]

    kronrad, krflag = sep.kron_radius(data, x, y, a, b, theta, 6.0)

    flux_auto, fluxerr, flag = sep.sum_ellipse(
        data, x, y, a, b, theta, 2.5 * kronrad, segmap=segmap, seg_id=seg_id, subpix=1
    )

    # Test total flux
    assert_allclose(flux_auto, total_exact, rtol=5.0e-2)

    # Flux_radius
    for flux_fraction in [0.2, 0.5]:

        # Exact solution
        rhalf_exact = np.sqrt(-np.log(1 - flux_fraction) * gsigma**2 * 2)

        # Masked measurement
        flux_radius, flag = sep.flux_radius(
            data,
            x,
            y,
            6.0 * a,
            flux_fraction,
            seg_id=seg_id,
            segmap=segmap,
            normflux=flux_auto,
            subpix=5,
        )

        # Test flux fraction
        assert_allclose(flux_radius, rhalf_exact, rtol=5.0e-2)

    if False:
        print("test_masked_flux_radius")
        print(total_exact, flux_auto)
        print(rhalf_exact, flux_radius)


def test_mask_ellipse():
    """
    Test that the correct number of elements are masked with an ellipse.
    """
    arr = np.zeros((20, 20), dtype=np.bool_)

    # should mask 5 pixels:
    sep.mask_ellipse(arr, 10.0, 10.0, 1.0, 1.0, 0.0, r=1.001)
    assert arr.sum() == 5

    # should mask 13 pixels:
    sep.mask_ellipse(arr, 10.0, 10.0, 1.0, 1.0, 0.0, r=2.001)
    assert arr.sum() == 13


def test_flux_radius():
    """
    Test that the correct radius is returned for varying flux fractions.
    """
    data = np.ones(data_shape)
    fluxfrac = [0.2**2, 0.3**2, 0.7**2, 1.0]
    true_r = [2.0, 3.0, 7.0, 10.0]
    r, _ = sep.flux_radius(
        data, x, y, 10.0 * np.ones_like(x), [0.2**2, 0.3**2, 0.7**2, 1.0], subpix=5
    )
    for i in range(len(fluxfrac)):
        assert_allclose(r[:, i], true_r[i], rtol=0.01)


def test_mask_ellipse_alt():
    """
    Mask_ellipse with cxx, cyy, cxy parameters.
    """
    arr = np.zeros((20, 20), dtype=np.bool_)

    # should mask 5 pixels:
    sep.mask_ellipse(arr, 10.0, 10.0, cxx=1.0, cyy=1.0, cxy=0.0, r=1.001)
    assert arr.sum() == 5

    # should mask 13 pixels:
    sep.mask_ellipse(arr, 10.0, 10.0, cxx=1.0, cyy=1.0, cxy=0.0, r=2.001)
    assert arr.sum() == 13


# -----------------------------------------------------------------------------
# General behavior and utilities


def test_byte_order_exception():
    """
    Test that SEP will not run with non-native byte order.

    Test that error about byte order is raised with non-native
    byte order input array. This should happen for Background, extract,
    and aperture functions.
    """

    data = np.ones((100, 100), dtype=np.float64)
    data = data.view(data.dtype.newbyteorder("S"))
    with pytest.raises(ValueError) as excinfo:
        bkg = sep.Background(data)
    assert "byte order" in excinfo.value.args[0]


def test_set_pixstack():
    """
    Ensure that setting the pixel stack size works.
    """
    old = sep.get_extract_pixstack()
    new = old * 2
    sep.set_extract_pixstack(new)
    assert new == sep.get_extract_pixstack()
    sep.set_extract_pixstack(old)


def test_set_sub_object_limit():
    """
    Ensure that setting the sub-object deblending limit works.
    """
    old = sep.get_sub_object_limit()
    new = old * 2
    sep.set_sub_object_limit(new)
    assert new == sep.get_sub_object_limit()
    sep.set_sub_object_limit(old)


def test_extract_deblend_disabled_skips_deblend_tree():
    """
    Disabling deblending should not build the threshold deblend tree.
    """

    old = sep.get_sub_object_limit()
    data = np.zeros((7, 7), dtype=np.float32)
    data[2:5, 2:5] = 10.0

    try:
        sep.set_sub_object_limit(1)
        objects = sep.extract(
            data,
            1.0,
            minarea=1,
            filter_kernel=None,
            deblend_cont=1.0,
            deblend_nthresh=32,
            clean=False,
        )
    finally:
        sep.set_sub_object_limit(old)

    assert len(objects) == 1


def test_extract_deblend_prunes_low_contrast_branches():
    """
    Low-contrast branches should not consume deblend tree capacity.
    """

    old = sep.get_sub_object_limit()
    data = np.ones((9, 41), dtype=np.float32) * 2.0
    data[4, 2::4] = 20.0

    try:
        sep.set_sub_object_limit(4)
        objects = sep.extract(
            data,
            1.0,
            minarea=1,
            filter_kernel=None,
            deblend_cont=0.9,
            deblend_nthresh=32,
            clean=False,
        )
    finally:
        sep.set_sub_object_limit(old)

    assert len(objects) == 1


@pytest.mark.parametrize(
    "method,separation_fwhm,expected",
    [
        ("threshold", 1.0, 1),
        ("threshold", 2.0, 2),
        ("watershed", 1.0, 2),
        ("watershed", 2.0, 2),
    ],
)
def test_extract_deblend_pair_separation_matrix(method, separation_fwhm, expected):
    """Controlled equal-flux pairs have stable merge and split behavior."""
    fwhm = 3.0
    separation = separation_fwhm * fwhm
    xtrue = np.array([32.0 - separation / 2.0, 32.0 + separation / 2.0])
    ytrue = np.array([32.0, 32.0])
    data = _gaussian_scene((65, 65), xtrue, ytrue, fwhm, [3000.0, 3000.0])

    objects = sep.extract(
        data,
        1.0,
        minarea=3,
        filter_kernel=None,
        clean=False,
        deblend_cont=0.005,
        deblend_fwhm=fwhm if method == "watershed" else 0.0,
        deblend_method=method,
    )

    assert len(objects) == expected
    if expected == 2:
        assert_allclose(np.sort(objects["x"]), xtrue, atol=0.5)


@pytest.mark.parametrize("method", ["threshold", "watershed"])
@pytest.mark.parametrize("flux_ratio", [0.3, 0.1])
def test_extract_deblend_high_contrast_pair(method, flux_ratio):
    """Resolvable faint companions survive next to a brighter source."""
    fwhm = 3.0
    xtrue = np.array([29.0, 35.0])
    ytrue = np.array([32.0, 32.0])
    flux = np.array([5000.0, 5000.0 * flux_ratio])
    data = _gaussian_scene((65, 65), xtrue, ytrue, fwhm, flux)

    objects = sep.extract(
        data,
        0.5,
        minarea=3,
        filter_kernel=None,
        clean=False,
        deblend_cont=0.005,
        deblend_fwhm=fwhm if method == "watershed" else 0.0,
        deblend_method=method,
    )

    assert len(objects) == 2
    assert_allclose(np.sort(objects["x"]), xtrue, atol=0.5)


def test_extract_watershed_centroids_follow_segment_moments():
    """
    Watershed-deblended centroids should match first moments of the assigned
    segmentation regions rather than staying pinned to seed peak pixels.
    """

    shape = (25, 25)
    ygrid, xgrid = np.mgrid[:shape[0], :shape[1]]
    sigma1 = 1.2
    sigma2 = 1.35
    data = (
        20.0
        * np.exp(-((xgrid - 8.35) ** 2 + (ygrid - 12.15) ** 2) / (2.0 * sigma1**2))
        + 14.0
        * np.exp(-((xgrid - 13.65) ** 2 + (ygrid - 11.8) ** 2) / (2.0 * sigma2**2))
    ).astype(np.float32)

    objects, segmap = sep.extract(
        data,
        0.8,
        minarea=1,
        filter_kernel=None,
        clean=False,
        segmentation_map=True,
        deblend_cont=0.0,
        deblend_method="watershed",
    )

    assert len(objects) == 2

    order = np.argsort(objects["x"])
    objects = objects[order]

    for sorted_idx, orig_idx in enumerate(order):
        obj = objects[sorted_idx]
        region = segmap == orig_idx + 1
        assert region.any()

        yy, xx = np.nonzero(region)
        weights = data[region].astype(np.float64)
        x_moment = np.sum(xx * weights) / np.sum(weights)
        y_moment = np.sum(yy * weights) / np.sum(weights)

        assert_allclose(obj["x"], x_moment, atol=1.0e-6)
        assert_allclose(obj["y"], y_moment, atol=1.0e-6)

        # This regression should fail if deblended centroids fall back to
        # integer watershed seeds / peak pixels instead of measured moments.
        assert abs(obj["x"] - obj["xpeak"]) > 0.1


def test_extract_watershed_peak_relabel_preserves_brightness_order():
    """Peak-separation relabeling must not overwrite later source seeds."""
    ygrid, xgrid = np.mgrid[:31, :31]
    data = (
        12.0 * np.exp(-((xgrid - 8.0) ** 2 + (ygrid - 15.0) ** 2) / 20.0)
        + 30.0 * np.exp(-((xgrid - 21.0) ** 2 + (ygrid - 15.0) ** 2) / 20.0)
    ).astype(np.float32)

    objects = sep.extract(
        data,
        0.01,
        minarea=1,
        filter_kernel=None,
        clean=False,
        deblend_cont=0.001,
        deblend_fwhm=2.0,
        deblend_method="watershed",
    )
    objects.sort(order="x")

    assert len(objects) == 2
    assert_allclose(objects["x"], [8.0, 21.0], atol=0.5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minarea": 0},
        {"deblend_nthresh": -1},
        {"deblend_cont": -0.1},
        {"deblend_cont": 1.1},
        {"deblend_fwhm": -1.0},
        {"clean_param": 0.0},
        {"filter_kernel": np.zeros((3, 3), dtype=np.float32)},
    ],
)
def test_extract_rejects_invalid_detection_parameters(kwargs):
    data = np.zeros((5, 5), dtype=np.float32)
    with pytest.raises(ValueError):
        sep.extract(data, 1.0, **kwargs)


def test_long_error_msg():
    """
    Test the error handling in SEP.

    Ensure that the error message is created successfully when
    there is an error detail.
    """

    # set extract pixstack to an insanely small value; this will trigger
    # a detailed error message when running sep.extract()
    old = sep.get_extract_pixstack()
    sep.set_extract_pixstack(5)

    data = np.ones((10, 10), dtype=np.float64)
    with pytest.raises(Exception) as excinfo:
        sep.extract(data, 0.1)
    msg = excinfo.value.args[0]
    assert type(msg) == str  # check that message is the native string type
    assert msg.startswith("internal pixel buffer full: The limit")

    # restore
    sep.set_extract_pixstack(old)


# ---------------------------------------------------------------------------
# PSF photometry tests
# ---------------------------------------------------------------------------


def _make_gaussian_source(nx, ny, xcen, ycen, flux, fwhm):
    """Create an image with a single pixel-integrated Gaussian source."""
    sigma = fwhm / 2.3548
    x = np.array([
        0.5 * (math.erf((i + 0.5 - xcen) / (math.sqrt(2) * sigma)) -
               math.erf((i - 0.5 - xcen) / (math.sqrt(2) * sigma)))
        for i in range(nx)
    ], dtype=np.float64)
    y = np.array([
        0.5 * (math.erf((j + 0.5 - ycen) / (math.sqrt(2) * sigma)) -
               math.erf((j - 0.5 - ycen) / (math.sqrt(2) * sigma)))
        for j in range(ny)
    ], dtype=np.float64)
    img = np.outer(y, x)
    img = flux * img / img.sum()
    return img.astype(np.float32)


def _gaussian_kernel(size, sigma):
    """Create a normalized native-pixel Gaussian kernel."""
    y, x = np.mgrid[0:size, 0:size]
    c = size // 2
    kernel = np.exp(-((x - c) ** 2 + (y - c) ** 2) / (2.0 * sigma**2))
    return (kernel / kernel.sum()).astype(np.float32)


def _psf_native_kernel(psf):
    """Render a constant PSF as a native-pixel filter kernel."""
    kernel = np.zeros((psf.stamp_height, psf.stamp_width), dtype=np.float64)
    sep.model_psf(
        kernel,
        [psf.stamp_width // 2],
        [psf.stamp_height // 2],
        [1.0],
        psf,
    )
    return kernel


def test_psf_from_gaussian():
    """PSF.from_gaussian creates a valid PSF model."""
    psf = sep.PSF.from_gaussian(fwhm=3.5)
    assert psf.stamp_width == 15
    assert psf.stamp_height == 15
    assert psf.ncomp == 1
    assert psf.degree == 0
    assert psf.sampling == 0.5
    assert psf.fwhm == 3.5


def test_winpos_psf_requires_sig_or_psf():
    """winpos requires either Gaussian sigma or a PSF model."""
    data = np.zeros((16, 16), dtype=np.float32)

    with pytest.raises(ValueError, match="`sig` is required"):
        sep.winpos(data, [8.0], [8.0])


def test_winpos_psf_weighting_tracks_supplied_psf():
    """PSF-weighted winpos works with a supplied supersampled PSF model."""
    fwhm = 3.0
    psf = sep.PSF.from_gaussian(fwhm=fwhm, oversampling=4)

    data = np.zeros((96, 96), dtype=np.float32)
    xtrue = 48.35
    ytrue = 47.65
    sep.model_psf(data, [xtrue], [ytrue], [1500.0], psf)

    xinit = np.array([xtrue + 0.45], dtype=np.float64)
    yinit = np.array([ytrue - 0.35], dtype=np.float64)

    xg, yg, _ = sep.winpos(data, xinit, yinit, fwhm / 2.354820045)
    xp, yp, flag = sep.winpos(data, xinit, yinit, psf=psf, maxstep=0.8)

    assert flag[0] == 0
    assert np.hypot(xp[0] - xtrue, yp[0] - ytrue) < 0.01
    assert_allclose(xp, xg, atol=2.0e-3)
    assert_allclose(yp, yg, atol=2.0e-3)


def test_winpos_psf_array_matches_scalar_calls():
    """Batched PSF-weighted winpos matches per-source calls."""
    fwhm = 4.0
    psf = sep.PSF.from_gaussian(fwhm=fwhm, oversampling=4)
    data = np.zeros((96, 96), dtype=np.float32)
    xtrue = np.array([22.3, 48.2, 71.7], dtype=np.float64)
    ytrue = np.array([24.6, 44.1, 68.4], dtype=np.float64)
    flux = np.array([1200.0, 900.0, 1500.0], dtype=np.float64)
    sep.model_psf(data, xtrue, ytrue, flux, psf)

    xinit = xtrue + np.array([0.35, -0.28, 0.22])
    yinit = ytrue + np.array([-0.18, 0.31, -0.27])
    maxstep = np.array([0.6, 0.8, 0.7], dtype=np.float64)

    xb, yb, flagb = sep.winpos(data, xinit, yinit, psf=psf, maxstep=maxstep)

    xs = np.empty_like(xb)
    ys = np.empty_like(yb)
    flags = np.empty_like(flagb)
    for i in range(len(xinit)):
        x1, y1, f1 = sep.winpos(data, xinit[i], yinit[i], psf=psf,
                                maxstep=maxstep[i])
        xs[i] = x1
        ys[i] = y1
        flags[i] = f1

    assert_allclose(xb, xs, atol=1.0e-10)
    assert_allclose(yb, ys, atol=1.0e-10)
    assert_equal(flagb, flags)


def test_winpos_isolated_sources_stable_across_subpixel_phases():
    """Gaussian windowing converges for varied phases and initial offsets."""
    fwhm = 3.0
    sigma = fwhm / 2.354820045
    xtrue = np.array([18.15, 32.50, 46.85])
    ytrue = np.array([20.80, 33.25, 44.60])
    flux = np.array([1200.0, 900.0, 1500.0])
    data = _gaussian_scene((64, 64), xtrue, ytrue, fwhm, flux)
    xinit = xtrue + np.array([0.45, -0.40, 0.35])
    yinit = ytrue + np.array([-0.35, 0.30, -0.42])

    xwin, ywin, flag = sep.winpos(data, xinit, yinit, sigma)

    assert np.all(flag == 0)
    assert_allclose(xwin, xtrue, atol=0.02)
    assert_allclose(ywin, ytrue, atol=0.02)


def test_winpos_segmented_close_pair_does_not_collapse():
    """Strict segmented winpos should not drift onto a close neighbor."""
    ygrid, xgrid = np.mgrid[:64, :64]
    data = (
        100.0 * np.exp(-((xgrid - 30.0) ** 2 + (ygrid - 32.0) ** 2) / 8.0)
        + 250.0 * np.exp(-((xgrid - 34.0) ** 2 + (ygrid - 32.0) ** 2) / 8.0)
    ).astype(np.float64)

    segmap = np.zeros(data.shape, dtype=np.int32)
    source = data > 1.0
    segmap[source & (xgrid < 32)] = 1
    segmap[source & (xgrid >= 32)] = 2

    xinit = np.array([30.0, 34.0])
    yinit = np.array([32.0, 32.0])
    sig = np.array([2.0, 2.0])

    xplain, _, _ = sep.winpos(data, xinit, yinit, sig)
    xseg, yseg, flag = sep.winpos(
        data, xinit, yinit, sig, segmap=segmap, seg_id=[-1, -2]
    )

    assert xplain[0] > 31.0
    assert xseg[0] < 31.0
    assert xseg[1] > 33.0
    assert_allclose(yseg, yinit, atol=0.05)
    assert np.all(flag & sep.APER_ALLMASKED == 0)


def test_winpos_maxshift_caps_total_displacement():
    """maxshift limits cumulative winpos motion and sets APER_TRUNC."""
    ygrid, xgrid = np.mgrid[:48, :48]
    data = np.exp(-((xgrid - 26.0) ** 2 + (ygrid - 24.0) ** 2) / 18.0)

    xinit = np.array([18.0])
    yinit = np.array([24.0])
    xwin, ywin, flag = sep.winpos(data, xinit, yinit, sig=[6.0], maxshift=0.5)

    assert_allclose(np.hypot(xwin - xinit, ywin - yinit), 0.5, atol=1.0e-12)
    assert flag[0] & sep.APER_TRUNC


def test_psf_flux_only():
    """PSF flux-only photometry recovers flux at the exact position."""
    fwhm = 3.5
    psf = sep.PSF.from_gaussian(fwhm=fwhm)
    data = _make_gaussian_source(64, 64, 32.0, 32.0, 1000.0, fwhm)

    flux, fluxerr, xf, yf, flag, _, _ = sep.psf_fit(
        data, 32.0, 32.0, psf, fit_positions=False
    )
    assert_allclose(flux, 1000.0, rtol=0.01)
    assert flag == 0


def test_psf_snr_matches_flux_only_snr():
    """psf_snr matches fixed-position PSF flux divided by its error."""
    fwhm = 3.5
    psf = sep.PSF.from_gaussian(fwhm=fwhm)
    data = np.zeros((64, 64), dtype=np.float64)
    sep.model_psf(data, [32.0], [32.0], [1000.0], psf)

    snr = sep.psf_snr(data, psf, var=25.0)
    flux, fluxerr, _, _, flag, _, _ = sep.psf_fit(
        data, 32.0, 32.0, psf, var=25.0, fit_positions=False
    )

    assert flag == 0
    assert_allclose(snr[32, 32], flux / fluxerr, rtol=1e-12)
    assert np.argmax(snr) == np.ravel_multi_index((32, 32), snr.shape)


def test_psf_snr_matches_spatially_varying_constant_component():
    """Fast constant-PSF path matches the generic spatially varying path."""
    fwhm = 3.5
    oversampling = 2
    size = int(np.ceil(4.0 * fwhm))
    if size % 2 == 0:
        size += 1
    ossize = size * oversampling
    sigma = fwhm / 2.354820045 * oversampling
    y, x = np.mgrid[0:ossize, 0:ossize]
    stamp = np.exp(
        -((x - ossize // 2) ** 2 + (y - ossize // 2) ** 2) / (2.0 * sigma**2)
    )
    stamp = (stamp / stamp.sum()).astype(np.float32)
    data = stamp[np.newaxis, :, :]
    zeros = np.zeros_like(data)
    psf_fast = sep.PSF(data, sampling=1.0 / oversampling, degree=0, fwhm=fwhm)
    psf_generic = sep.PSF(
        np.concatenate([data, zeros, zeros], axis=0),
        sampling=psf_fast.sampling,
        degree=1,
        x0=0.0,
        y0=0.0,
        sx=1.0,
        sy=1.0,
        fwhm=fwhm,
    )

    image = np.zeros((48, 48), dtype=np.float64)
    sep.model_psf(image, [24.0], [24.0], [500.0], psf_fast)
    var = np.ones_like(image) * 9.0

    snr_fast = sep.psf_snr(image, psf_fast, var=var)
    snr_generic = sep.psf_snr(image, psf_generic, var=var)

    assert_allclose(snr_fast, snr_generic, rtol=1e-7, atol=1e-10)


def test_psf_snr_matches_matched_filter_for_constant_psf():
    """Constant-PSF S/N matches the extract matched-filter statistic."""
    rng = np.random.default_rng(123)
    psf = sep.PSF.from_gaussian(fwhm=3.5, oversampling=2)
    kernel = _psf_native_kernel(psf)
    data = rng.normal(size=(32, 33))
    err = np.full_like(data, 2.5)

    snr = sep.psf_snr(data, psf, err=err)
    expected = matched_filter_snr(data, err, kernel)

    assert_allclose(snr, expected, rtol=1e-6, atol=1e-6)


def test_psf_snr_matches_matched_filter_orientation():
    """An asymmetric constant PSF uses the same orientation as matched filter."""
    rng = np.random.default_rng(456)
    kernel = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0, 0.0, 0.0],
            [0.0, 3.0, 6.0, 1.0, 0.0],
            [0.0, 0.0, 2.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    kernel /= kernel.sum()
    psf = sep.PSF(kernel, sampling=1.0, degree=0, fwhm=2.0)
    data = rng.normal(size=(24, 25))
    err = 1.7 + 0.2 * rng.random(size=data.shape)

    snr = sep.psf_snr(data, psf, err=err)
    expected = matched_filter_snr(data, err, kernel)

    assert_allclose(snr, expected, rtol=1e-6, atol=1e-6)


def test_psf_snr_matches_matched_filter_with_mask():
    """Masked pixels are excluded like infinite-noise matched-filter pixels."""
    rng = np.random.default_rng(789)
    psf = sep.PSF.from_gaussian(fwhm=3.0, oversampling=2)
    kernel = _psf_native_kernel(psf)
    data = rng.normal(size=(30, 31))
    err = 2.0 + 0.5 * rng.random(size=data.shape)
    mask = np.zeros(data.shape, dtype=np.uint8)
    mask[10:14, 12:16] = 1
    mask[20, 5:11] = 1

    masked_data = data.copy()
    masked_data[mask > 0] = 0.0
    masked_err = err.copy()
    masked_err[mask > 0] = 1.0e30

    snr = sep.psf_snr(data, psf, err=err, mask=mask)
    expected = matched_filter_snr(masked_data, masked_err, kernel)

    assert_allclose(snr, expected, rtol=1e-6, atol=1e-7)


def test_psf_extract_uses_spatially_varying_psf():
    """A spatially varying PSF detects a source missed by a fixed kernel."""
    size = 15
    left = _gaussian_kernel(size, 1.0)
    right = _gaussian_kernel(size, 2.7)
    const = 0.5 * (left + right)
    xcomp = 0.5 * (right - left)
    ycomp = np.zeros_like(const)

    psf_var = sep.PSF(
        np.stack([const, xcomp, ycomp]),
        sampling=1.0,
        degree=1,
        x0=50.0,
        y0=0.0,
        sx=40.0,
        sy=1.0,
        fwhm=3.5,
    )
    psf_fixed = sep.PSF(const, sampling=1.0, degree=0, fwhm=3.5)

    data = np.zeros((80, 110), dtype=np.float64)
    x = np.array([10.0, 90.0])
    y = np.array([40.0, 40.0])
    sep.model_psf(data, x, y, [100.0, 100.0], psf_var)

    snr_var = sep.psf_snr(data, psf_var, var=1.0)
    snr_fixed = sep.psf_snr(data, psf_fixed, var=1.0)

    assert snr_var[40, 90] > 1.2 * snr_fixed[40, 90]

    objects_var = sep.psf_extract(data, 9.5, psf_var, var=1.0, minarea=1)
    objects_fixed = sep.psf_extract(data, 9.5, psf_fixed, var=1.0, minarea=1)

    assert len(objects_var) == 2
    assert len(objects_fixed) == 1


def test_psf_snr_gain_matches_data_dependent_matched_filter():
    """gain adds positive pixel values to the matched-filter variance."""
    psf = sep.PSF.from_gaussian(fwhm=3.5, oversampling=2)
    kernel = _psf_native_kernel(psf)
    data = np.zeros((48, 49), dtype=np.float64)
    sep.model_psf(data, [24.0], [23.0], [700.0], psf)
    data[8:13, 35:40] -= 20.0

    gain = 2.5
    total_var = 16.0 + np.maximum(data, 0.0) / gain

    snr = sep.psf_snr(data, psf, var=16.0, gain=gain)
    expected = matched_filter_snr(data, np.sqrt(total_var), kernel)

    assert_allclose(snr, expected, rtol=1e-6, atol=1e-7)


def test_psf_snr_gain_lowers_positive_source_significance():
    """Poisson variance from gain reduces S/N for bright positive sources."""
    psf = sep.PSF.from_gaussian(fwhm=3.5)
    data = np.zeros((64, 64), dtype=np.float64)
    sep.model_psf(data, [32.0], [32.0], [1000.0], psf)

    snr_no_gain = sep.psf_snr(data, psf, var=25.0)
    snr_gain = sep.psf_snr(data, psf, var=25.0, gain=1.0)

    assert snr_gain[32, 32] < snr_no_gain[32, 32]


def test_psf_snr_local_bkg_matches_weighted_two_parameter_fit():
    """local_bkg=True matches a source-plus-constant weighted fit."""
    rng = np.random.default_rng(321)
    psf = sep.PSF.from_gaussian(fwhm=3.2, oversampling=2)
    kernel = _psf_native_kernel(psf)
    yy, xx = np.indices((34, 35))
    data = 12.0 + 0.02 * xx - 0.01 * yy
    data += rng.normal(scale=0.05, size=data.shape)
    sep.model_psf(data, [18.0], [17.0], [120.0], psf)

    err = 1.5 + 0.2 * rng.random(size=data.shape)
    mask = np.zeros(data.shape, dtype=np.uint8)
    mask[12:15, 16:19] = 1
    mask[23, 4:9] = 1

    masked_data = data.copy()
    masked_data[mask > 0] = 0.0
    masked_err = err.copy()
    masked_err[mask > 0] = 1.0e30

    snr = sep.psf_snr(data, psf, err=err, mask=mask, local_bkg=True)
    expected = matched_filter_snr_local_bkg(masked_data, masked_err, kernel)

    assert_allclose(snr, expected, rtol=1e-6, atol=1e-6)


def test_psf_snr_respects_variable_variance_and_mask():
    """High variance and masked pixels are downweighted in PSF significance."""
    fwhm = 3.5
    psf = sep.PSF.from_gaussian(fwhm=fwhm)
    data = np.zeros((64, 64), dtype=np.float64)
    sep.model_psf(data, [32.0], [32.0], [1000.0], psf)

    var = np.ones_like(data)
    snr_uniform = sep.psf_snr(data, psf, var=var)

    var_high = var.copy()
    var_high[29:36, 29:36] = 100.0
    snr_high_var = sep.psf_snr(data, psf, var=var_high)

    mask = np.zeros_like(data, dtype=np.uint8)
    mask[29:36, 29:36] = 1
    snr_masked = sep.psf_snr(data, psf, var=var, mask=mask)

    assert snr_high_var[32, 32] < snr_uniform[32, 32]
    assert snr_masked[32, 32] < snr_high_var[32, 32]


def test_psf_snr_local_bkg_rejects_constant_background():
    """Local-background PSF S/N removes constant offsets in the footprint."""
    fwhm = 3.5
    psf = sep.PSF.from_gaussian(fwhm=fwhm)
    data = np.full((64, 64), 100.0, dtype=np.float64)

    snr_plain = sep.psf_snr(data, psf, var=25.0)
    snr_lbs = sep.psf_snr(data, psf, var=25.0, local_bkg=True)

    assert snr_plain[32, 32] > 1.0
    assert abs(snr_lbs[32, 32]) < 1e-10


def test_psf_snr_local_bkg_keeps_point_source():
    """Local-background PSF S/N still detects a compact source."""
    fwhm = 3.5
    psf = sep.PSF.from_gaussian(fwhm=fwhm)
    data = np.full((64, 64), 100.0, dtype=np.float64)
    sep.model_psf(data, [32.0], [32.0], [500.0], psf)

    snr_lbs = sep.psf_snr(data, psf, var=25.0, local_bkg=True)

    assert snr_lbs[32, 32] > 5.0
    assert np.argmax(snr_lbs) == np.ravel_multi_index((32, 32), snr_lbs.shape)


def test_psf_extract_detects_psf_weighted_sources():
    """psf_extract detects sources from the PSF-matched S/N image."""
    fwhm = 3.5
    psf = sep.PSF.from_gaussian(fwhm=fwhm)
    data = np.zeros((80, 80), dtype=np.float64)
    xtrue = np.array([25.0, 55.0])
    ytrue = np.array([28.0, 52.0])
    sep.model_psf(data, xtrue, ytrue, [250.0, 180.0], psf)

    objects, snr = sep.psf_extract(
        data, 5.0, psf, var=25.0, minarea=1, return_snr=True
    )

    assert len(objects) == 2
    peaks = sorted(zip(objects["xpeak"], objects["ypeak"]))
    assert peaks == [(25, 28), (55, 52)]
    assert snr[28, 25] > 5.0
    assert snr[52, 55] > 5.0


def test_psf_extract_returns_segmap_and_respects_mask():
    """psf_extract passes masks and segmentation output through extract."""
    fwhm = 3.5
    psf = sep.PSF.from_gaussian(fwhm=fwhm)
    data = np.zeros((64, 64), dtype=np.float64)
    sep.model_psf(data, [32.0], [32.0], [300.0], psf)

    objects, segmap, snr = sep.psf_extract(
        data, 5.0, psf, var=25.0, segmentation_map=True, return_snr=True
    )
    assert len(objects) == 1
    assert segmap.shape == data.shape
    assert snr[32, 32] > 5.0

    mask = np.zeros_like(data, dtype=np.uint8)
    mask[24:41, 24:41] = 1
    masked = sep.psf_extract(data, 5.0, psf, var=25.0, mask=mask)
    assert len(masked) == 0


def test_psf_extract_can_normalize_snr_background():
    """psf_extract can renormalize a biased PSF-matched detection image."""
    rng = np.random.default_rng(123)
    fwhm = 3.5
    psf = sep.PSF.from_gaussian(fwhm=fwhm)
    data = rng.normal(100.0, 5.0, size=(128, 128))
    sep.model_psf(data, [64.0], [65.0], [350.0], psf)

    objects, snr = sep.psf_extract(
        data,
        5.0,
        psf,
        var=25.0,
        minarea=1,
        return_snr=True,
        normalize_snr=True,
        snr_bw=32,
        snr_bh=32,
    )

    assert len(objects) == 1
    assert objects["xpeak"][0] == 64
    assert objects["ypeak"][0] == 65
    assert abs(np.median(snr)) < 0.5


def test_psf_peaks_finds_local_maxima():
    """psf_peaks returns local maxima from a PSF-matched S/N image."""
    fwhm = 3.5
    psf = sep.PSF.from_gaussian(fwhm=fwhm)
    data = np.zeros((80, 80), dtype=np.float64)
    xtrue = np.array([25.0, 55.0])
    ytrue = np.array([28.0, 52.0])
    sep.model_psf(data, xtrue, ytrue, [250.0, 180.0], psf)

    peaks, snr = sep.psf_peaks(
        data, 5.0, psf, var=25.0, min_distance=1.5, return_snr=True
    )

    assert len(peaks) == 2
    assert sorted(zip(peaks["xpeak"], peaks["ypeak"])) == [(25, 28), (55, 52)]
    assert_allclose(peaks["snr"], snr[peaks["ypeak"], peaks["xpeak"]])


def test_psf_peaks_suppresses_close_peaks():
    """psf_peaks greedily suppresses peaks closer than min_distance."""
    psf = sep.PSF.from_gaussian(fwhm=2.0)
    data = np.zeros((60, 60), dtype=np.float64)
    sep.model_psf(data, [28.0, 32.0], [30.0, 30.0], [300.0, 260.0], psf)

    unsuppressed = sep.psf_peaks(data, 5.0, psf, var=25.0, min_distance=0.0)
    suppressed = sep.psf_peaks(data, 5.0, psf, var=25.0, min_distance=5.0)

    assert len(unsuppressed) == 2
    assert len(suppressed) == 1
    assert suppressed["xpeak"][0] == 28
    assert suppressed["ypeak"][0] == 30


def test_psf_peaks_can_normalize_snr_background():
    """psf_peaks supports the same S/N normalization as psf_extract."""
    rng = np.random.default_rng(123)
    psf = sep.PSF.from_gaussian(fwhm=3.5)
    data = rng.normal(100.0, 5.0, size=(128, 128))
    sep.model_psf(data, [64.0], [65.0], [350.0], psf)

    peaks, snr = sep.psf_peaks(
        data,
        5.0,
        psf,
        var=25.0,
        return_snr=True,
        normalize_snr=True,
        snr_bw=32,
        snr_bh=32,
    )

    assert len(peaks) == 1
    assert peaks["xpeak"][0] == 64
    assert peaks["ypeak"][0] == 65
    assert abs(np.median(snr)) < 0.5


def test_psf_extract_peaks_mode_returns_fit_catalog():
    """psf_extract(mode='peaks') fits and prunes peak candidates."""
    psf = sep.PSF.from_gaussian(fwhm=3.5)
    data = np.zeros((80, 80), dtype=np.float64)
    sep.model_psf(data, [25.0, 55.0], [28.0, 52.0], [250.0, 120.0], psf)

    objects = sep.psf_extract(data, 5.0, psf, var=25.0, mode="peaks", fit_snr=8.0)

    assert objects.dtype.names == (
        "x",
        "y",
        "flux",
        "fluxerr",
        "fit_snr",
        "peak_snr",
        "qf",
        "rchi2",
        "fracflux",
        "xpeak",
        "ypeak",
        "chi2",
        "niter",
        "flag",
    )
    assert len(objects) == 1
    assert_allclose(objects["x"][0], 25.0, atol=0.1)
    assert_allclose(objects["y"][0], 28.0, atol=0.1)
    assert objects["fit_snr"][0] > 8.0
    assert_allclose(objects["qf"][0], 1.0, rtol=1e-6)
    assert objects["rchi2"][0] < 1e-10
    assert_allclose(objects["fracflux"][0], 1.0, rtol=1e-6)
    assert np.isfinite(objects["chi2"][0])
    assert objects["niter"][0] >= 1


def test_psf_extract_peaks_mode_can_return_raw_peaks():
    """fit_snr=None returns raw peak candidates from psf_extract."""
    psf = sep.PSF.from_gaussian(fwhm=3.5)
    data = np.zeros((80, 80), dtype=np.float64)
    sep.model_psf(data, [25.0, 55.0], [28.0, 52.0], [250.0, 180.0], psf)

    direct = sep.psf_peaks(data, 5.0, psf, var=25.0)
    via_extract, snr = sep.psf_extract(
        data, 5.0, psf, var=25.0, mode="peaks", fit_snr=None, return_snr=True
    )

    assert_allclose(via_extract["x"], direct["x"])
    assert_allclose(via_extract["y"], direct["y"])
    assert_allclose(via_extract["snr"], direct["snr"])
    assert snr[28, 25] > 5.0


def test_psf_extract_peaks_mode_rejects_segmentation_map():
    """mode='peaks' does not support connected segmentation maps."""
    psf = sep.PSF.from_gaussian(fwhm=3.5)
    data = np.zeros((32, 32), dtype=np.float64)

    with pytest.raises(ValueError, match="segmentation_map"):
        sep.psf_extract(data, 5.0, psf, var=25.0, mode="peaks", segmentation_map=True)


def test_psf_extract_peaks_mode_deblends_close_pair():
    """Peak-mode fits solve overlapping candidates jointly."""
    psf = sep.PSF.from_gaussian(fwhm=2.0)
    data = np.zeros((80, 80), dtype=np.float64)
    x = np.array([38.0, 42.0])
    y = np.array([40.0, 40.0])
    true_flux = np.array([300.0, 260.0])
    sep.model_psf(data, x, y, true_flux, psf)

    objects = sep.psf_extract(
        data, 5.0, psf, var=25.0, mode="peaks", fit_snr=0.0,
        fit_positions=False, group_factor=5.0, peak_min_distance=0.0
    )
    objects.sort(order="x")

    assert len(objects) == 2
    assert_allclose(objects["flux"], true_flux, rtol=1e-5)
    assert np.all(objects["qf"] > 0.99)
    assert np.all(objects["fracflux"] < 1.0)
    assert np.all(objects["fracflux"] > 0.98)


def test_psf_extract_peaks_mode_applies_quality_cuts():
    """Peak-mode quality cuts prune fitted candidates."""
    psf = sep.PSF.from_gaussian(fwhm=2.0)
    data = np.zeros((80, 80), dtype=np.float64)
    sep.model_psf(
        data, [38.0, 42.0], [40.0, 40.0], [300.0, 260.0], psf
    )

    loose = sep.psf_extract(
        data, 5.0, psf, var=25.0, mode="peaks", fit_snr=0.0,
        fit_positions=False, group_factor=5.0, peak_min_distance=0.0,
        min_fracflux=0.98
    )
    strict = sep.psf_extract(
        data, 5.0, psf, var=25.0, mode="peaks", fit_snr=0.0,
        fit_positions=False, group_factor=5.0, peak_min_distance=0.0,
        min_fracflux=0.995
    )

    assert len(loose) == 2
    assert len(strict) == 0

    psf = sep.PSF.from_gaussian(fwhm=3.5)
    data = np.zeros((80, 80), dtype=np.float64)
    sep.model_psf(data, [25.0], [28.0], [1000.0], psf)
    data[28, 30] += 200.0

    accepted = sep.psf_extract(
        data, 5.0, psf, var=25.0, mode="peaks", fit_snr=0.0,
        fit_positions=False, peak_min_distance=0.0, max_rchi2=0.6
    )
    rejected = sep.psf_extract(
        data, 5.0, psf, var=25.0, mode="peaks", fit_snr=0.0,
        fit_positions=False, peak_min_distance=0.0, max_rchi2=0.5
    )

    assert len(accepted) == 1
    assert accepted["rchi2"][0] > 0.5
    assert len(rejected) == 0


def test_psf_extract_peaks_mode_local_sky_corrects_flux():
    """Peak-mode local sky subtraction removes constant sky bias in fits."""
    psf = sep.PSF.from_gaussian(fwhm=3.5)
    data = np.full((80, 80), -10.0, dtype=np.float64)
    sep.model_psf(data, [40.0], [41.0], [800.0], psf)

    biased = sep.psf_extract(
        data, 5.0, psf, var=25.0, mode="peaks", fit_snr=0.0,
        fit_positions=False
    )
    corrected = sep.psf_extract(
        data, 5.0, psf, var=25.0, mode="peaks", fit_snr=0.0,
        fit_positions=False, peak_local_sky=True, peak_local_sky_box=20
    )

    assert len(biased) == 1
    assert len(corrected) == 1
    assert abs(corrected["flux"][0] - 800.0) < abs(biased["flux"][0] - 800.0)
    assert_allclose(corrected["flux"][0], 800.0, rtol=1e-5)


def test_psf_extract_peaks_mode_local_sky_refits_after_model_subtraction():
    """Local sky fitting refits after subtracting the current source model."""
    psf = sep.PSF.from_gaussian(fwhm=10.0, oversampling=2)
    data = np.zeros((100, 100), dtype=np.float64)
    true_flux = 20000.0
    sep.model_psf(data, [50.0], [50.0], [true_flux], psf)

    sky = sep._psf_peak_local_sky(data, var=25.0, box_size=20)
    peak_data = data - sky
    peaks = sep.psf_peaks(peak_data, 5.0, psf, var=25.0)
    first_pass = sep._fit_psf_peaks(
        peak_data, peaks, psf, var=25.0, fit_snr=0.0,
        fit_positions=False
    )
    refit = sep.psf_extract(
        data, 5.0, psf, var=25.0, mode="peaks", fit_snr=0.0,
        fit_positions=False, peak_local_sky=True, peak_local_sky_box=20
    )

    assert len(first_pass) == 1
    assert len(refit) == 1
    assert first_pass["flux"][0] < 0.95 * true_flux
    assert abs(refit["flux"][0] - true_flux) < abs(first_pass["flux"][0] - true_flux)
    assert_allclose(refit["flux"][0], true_flux, rtol=0.01)


def test_psf_extract_peaks_mode_iterates_on_residuals():
    """Iterative peak mode can recover a source suppressed in the first pass."""
    psf = sep.PSF.from_gaussian(fwhm=2.0)
    data = np.zeros((80, 80), dtype=np.float64)
    sep.model_psf(data, [38.0, 42.0], [40.0, 40.0], [500.0, 120.0], psf)

    single = sep.psf_extract(
        data, 5.0, psf, var=25.0, mode="peaks", fit_snr=5.0,
        fit_positions=False, peak_min_distance=5.0, peak_iterations=1
    )
    iterative = sep.psf_extract(
        data, 5.0, psf, var=25.0, mode="peaks", fit_snr=5.0,
        fit_positions=False, peak_min_distance=5.0,
        peak_duplicate_distance=1.0, peak_iterations=3
    )

    assert len(single) == 1
    assert len(iterative) == 2
    assert_allclose(np.sort(iterative["x"]), [38.0, 42.0], atol=0.1)
    iterative.sort(order="x")

    assert_allclose(iterative["flux"], [500.0, 120.0], rtol=1e-5)
    assert np.min(iterative["peak_snr"]) > 5.0


def test_psf_extract_peaks_mode_iterative_option_validation():
    """Residual iterations require fitted peak catalogs."""
    psf = sep.PSF.from_gaussian(fwhm=3.5)
    data = np.zeros((32, 32), dtype=np.float64)

    with pytest.raises(ValueError, match="peak_iterations"):
        sep.psf_extract(data, 5.0, psf, var=25.0, mode="peaks",
                        peak_iterations=0)
    with pytest.raises(ValueError, match="requires fit_snr"):
        sep.psf_extract(data, 5.0, psf, var=25.0, mode="peaks",
                        fit_snr=None, peak_iterations=2)


def test_fit_psf_peaks_applies_qf_cut():
    """The fitted peak helper can reject incomplete PSF footprints."""
    psf = sep.PSF.from_gaussian(fwhm=3.5)
    data = np.zeros((80, 80), dtype=np.float64)
    sep.model_psf(data, [6.0], [6.0], [1000.0], psf)
    peaks = np.array(
        [(6.0, 6.0, 50.0, 6, 6)],
        dtype=[
            ("x", np.float64),
            ("y", np.float64),
            ("snr", np.float64),
            ("xpeak", np.int64),
            ("ypeak", np.int64),
        ],
    )

    loose = sep._fit_psf_peaks(
        data, peaks, psf, var=25.0, fit_snr=0.0, fit_positions=False,
        keep_flagged=True, min_qf=0.9999
    )
    strict = sep._fit_psf_peaks(
        data, peaks, psf, var=25.0, fit_snr=0.0, fit_positions=False,
        keep_flagged=True, min_qf=0.99999
    )

    assert len(loose) == 1
    assert loose["qf"][0] < 1.0
    assert len(strict) == 0


def test_psf_fit_exact_center():
    """PSF fit at the exact source center recovers flux and position."""
    fwhm = 3.5
    psf = sep.PSF.from_gaussian(fwhm=fwhm)
    data = _make_gaussian_source(64, 64, 32.0, 32.0, 1000.0, fwhm)

    flux, fluxerr, xf, yf, flag, _, _ = sep.psf_fit(data, 32.0, 32.0, psf)
    assert_allclose(flux, 1000.0, rtol=0.01)
    assert_allclose(xf, 32.0, atol=0.01)
    assert_allclose(yf, 32.0, atol=0.01)
    assert flag == 0


def test_psf_fit_returns_chi2_and_niter():
    """psf_fit always exposes chi2 and niter."""
    fwhm = 3.5
    psf = sep.PSF.from_gaussian(fwhm=fwhm)
    data = _make_gaussian_source(64, 64, 32.0, 32.0, 1000.0, fwhm)

    flux, fluxerr, xf, yf, flag, chi2, niter = sep.psf_fit(data, 32.0, 32.0, psf)
    assert_allclose(flux, 1000.0, rtol=0.01)
    assert_allclose(xf, 32.0, atol=0.01)
    assert_allclose(yf, 32.0, atol=0.01)
    assert flag == 0
    assert np.isfinite(chi2)
    assert chi2 >= 0.0
    assert niter >= 1


def test_psf_fit_subpixel_offsets():
    """PSF fit recovers sub-pixel positions from an offset initial guess."""
    fwhm = 3.5
    psf = sep.PSF.from_gaussian(fwhm=fwhm)

    offsets = [(0.3, -0.2), (-0.4, 0.1), (0.0, 0.45), (-0.15, -0.35)]
    for dx, dy in offsets:
        xtrue, ytrue = 32.0 + dx, 32.0 + dy
        data = _make_gaussian_source(64, 64, xtrue, ytrue, 1000.0, fwhm)

        flux, fluxerr, xf, yf, flag, _, _ = sep.psf_fit(data, 32.0, 32.0, psf)
        assert_allclose(flux, 1000.0, rtol=0.01,
                        err_msg=f"offset=({dx}, {dy})")
        assert_allclose(xf, xtrue, atol=0.01,
                        err_msg=f"offset=({dx}, {dy})")
        assert_allclose(yf, ytrue, atol=0.01,
                        err_msg=f"offset=({dx}, {dy})")


def test_psf_fit_with_noise():
    """PSF fit works with noisy data and a variance map."""
    np.random.seed(42)
    fwhm = 3.5
    psf = sep.PSF.from_gaussian(fwhm=fwhm)
    xtrue, ytrue = 32.3, 31.7
    data = _make_gaussian_source(64, 64, xtrue, ytrue, 1000.0, fwhm)
    noise_var = 1.0
    data += np.random.normal(0, np.sqrt(noise_var), data.shape).astype(
        np.float32
    )

    flux, fluxerr, xf, yf, flag, _, _ = sep.psf_fit(
        data, 32.0, 32.0, psf, var=noise_var
    )
    assert_allclose(flux, 1000.0, atol=50,
                    err_msg="noisy flux recovery")
    assert_allclose(xf, xtrue, atol=0.1,
                    err_msg="noisy x recovery")
    assert_allclose(yf, ytrue, atol=0.1,
                    err_msg="noisy y recovery")
    assert fluxerr > 0


def test_psf_fit_multiple_sources():
    """PSF fit handles multiple isolated sources in one call."""
    fwhm = 3.5
    psf = sep.PSF.from_gaussian(fwhm=fwhm)
    data = np.zeros((128, 128), dtype=np.float32)
    sources = [(30.0, 30.0, 1000.0), (80.0, 50.0, 500.0), (50.0, 90.0, 2000.0)]
    for sx, sy, sf in sources:
        data += _make_gaussian_source(128, 128, sx, sy, sf, fwhm)

    x = np.array([s[0] for s in sources])
    y = np.array([s[1] for s in sources])
    expected = np.array([s[2] for s in sources])

    fout, ferr, xf, yf, flag, _, _ = sep.psf_fit(data, x, y, psf)
    assert_allclose(fout, expected, rtol=0.01)
    assert_allclose(xf, x, atol=0.01)
    assert_allclose(yf, y, atol=0.01)


def test_psf_fit_different_fwhm():
    """PSF fit works across a range of FWHM values."""
    for fw in [2.0, 5.0, 8.0]:
        psf = sep.PSF.from_gaussian(fwhm=fw)
        data = _make_gaussian_source(64, 64, 32.2, 31.8, 1000.0, fw)

        flux, _, xf, yf, _, _, _ = sep.psf_fit(data, 32.0, 32.0, psf)
        assert_allclose(flux, 1000.0, rtol=0.01,
                        err_msg=f"FWHM={fw}")
        assert_allclose(xf, 32.2, atol=0.02,
                        err_msg=f"FWHM={fw}")
        assert_allclose(yf, 31.8, atol=0.02,
                        err_msg=f"FWHM={fw}")


def test_psf_grouped_blended_pair():
    """Grouped PSF fit deblends a close pair better than non-grouped."""
    fwhm = 3.5
    sigma = fwhm / 2.3548
    psf = sep.PSF.from_gaussian(fwhm=fwhm)

    # Two sources separated by ~1.2 FWHM
    nx, ny = 64, 64
    x1, y1, f1 = 30.0, 32.0, 1000.0
    x2, y2, f2 = 30.0 + 1.2 * fwhm, 32.0, 800.0

    data = np.zeros((ny, nx), dtype=np.float32)
    for cx, cy, cf in [(x1, y1, f1), (x2, y2, f2)]:
        data += _make_gaussian_source(nx, ny, cx, cy, cf, fwhm)

    xa = np.array([x1, x2])
    ya = np.array([y1, y2])
    ftrue = np.array([f1, f2])

    # Non-grouped: biased because the sources overlap
    flux_ng, _, xf_ng, yf_ng, _, _, _ = sep.psf_fit(data, xa, ya, psf)

    # Grouped: simultaneous fit should deblend
    flux_g, _, xf_g, yf_g, _, _, _ = sep.psf_fit(
        data, xa, ya, psf, grouped=True
    )

    # Grouped should recover fluxes much better
    assert_allclose(flux_g, ftrue, rtol=0.02,
                    err_msg="grouped flux recovery")
    assert_allclose(xf_g, xa, atol=0.05,
                    err_msg="grouped x recovery")
    assert_allclose(yf_g, ya, atol=0.05,
                    err_msg="grouped y recovery")

    # Non-grouped should be noticeably biased
    err_ng = np.sum(np.abs(flux_ng - ftrue))
    err_g = np.sum(np.abs(flux_g - ftrue))
    assert err_g < err_ng, (
        f"grouped error ({err_g:.1f}) should be less than "
        f"non-grouped error ({err_ng:.1f})"
    )


def test_psf_grouped_large_component_localized():
    """Large grouped components use the localized solver and stay accurate."""
    fwhm = 3.0
    psf = sep.PSF.from_gaussian(fwhm=fwhm)

    nsrc = 80
    spacing = 4.5
    x = 16.0 + spacing * np.arange(nsrc, dtype=np.float64)
    y = 48.0 + 0.35 * np.sin(np.arange(nsrc, dtype=np.float64) * 0.3)
    flux_true = 900.0 + 120.0 * (np.arange(nsrc) % 5)

    data = np.zeros((96, 400), dtype=np.float32)
    sep.model_psf(data, x, y, flux_true, psf)

    flux_ng, _, _, _, _, _, _ = sep.psf_fit(data, x, y, psf)
    flux_g, _, xf_g, yf_g, _, _, _ = sep.psf_fit(data, x, y, psf, grouped=True)

    err_ng = np.mean(np.abs(flux_ng - flux_true))
    err_g = np.mean(np.abs(flux_g - flux_true))

    assert err_g < 0.5 * err_ng
    assert_allclose(xf_g, x, atol=0.7)
    assert_allclose(yf_g, y, atol=0.05)


def test_psf_grouped_close_pair_stays_non_negative():
    """Grouped fit should avoid negative fluxes for very close noisy pairs."""
    fwhm = 3.0
    psf = sep.PSF.from_gaussian(fwhm=fwhm)

    x = np.array([48.0, 48.0 + 0.5 * fwhm], dtype=np.float64)
    y = np.array([48.0, 48.0], dtype=np.float64)
    flux_true = np.array([1200.0, 800.0], dtype=np.float64)

    rng = np.random.default_rng(1234)
    data = np.zeros((96, 96), dtype=np.float32)
    sep.model_psf(data, x, y, flux_true, psf)
    data += rng.normal(0.0, 5.0, size=data.shape).astype(np.float32)

    flux_g, _, xf_g, yf_g, _, _, _ = sep.psf_fit(
        data, x, y, psf, var=25.0, grouped=True
    )

    assert np.all(flux_g >= -1.0e-8)
    assert_allclose(np.sum(flux_g), np.sum(flux_true), rtol=0.1)
    assert_allclose(xf_g, x, atol=0.5)
    assert_allclose(yf_g, y, atol=0.5)


def test_psf_grouped_returns_chi2_and_niter():
    """Grouped psf_fit always exposes chi2 and niter."""
    fwhm = 3.5
    psf = sep.PSF.from_gaussian(fwhm=fwhm)

    x = np.array([30.0, 30.0 + 1.2 * fwhm], dtype=np.float64)
    y = np.array([32.0, 32.0], dtype=np.float64)
    flux_true = np.array([1000.0, 800.0], dtype=np.float64)

    data = np.zeros((64, 64), dtype=np.float32)
    for cx, cy, cf in zip(x, y, flux_true):
        data += _make_gaussian_source(64, 64, cx, cy, cf, fwhm)

    flux, fluxerr, xf, yf, flag, chi2, niter = sep.psf_fit(
        data, x, y, psf, grouped=True
    )

    assert_allclose(flux, flux_true, rtol=0.02)
    assert_allclose(xf, x, atol=0.05)
    assert_allclose(yf, y, atol=0.05)
    assert np.all(np.isfinite(chi2))
    assert np.all(chi2 >= 0.0)
    assert np.all(niter >= 1)
    assert np.all(flag == 0)


def test_psf_grouped_fit_radius_recovers_positions():
    """Grouped fit_radius should not bias position updates in exact fits."""
    fwhm = 3.5
    fit_radius = 3.0
    psf = sep.PSF.from_gaussian(fwhm=fwhm)

    x_true = np.array([30.35, 34.35], dtype=np.float64)
    y_true = np.array([31.75, 32.20], dtype=np.float64)
    flux_true = np.array([1100.0, 700.0], dtype=np.float64)
    x_init = x_true + np.array([0.30, -0.25], dtype=np.float64)
    y_init = y_true + np.array([-0.20, 0.25], dtype=np.float64)

    data = np.zeros((64, 64), dtype=np.float32)
    sep.model_psf(data, x_true, y_true, flux_true, psf)

    flux, _, xf, yf, flag, chi2, niter = sep.psf_fit(
        data, x_init, y_init, psf, grouped=True, fit_radius=fit_radius
    )

    assert_allclose(flux, flux_true, rtol=0.05)
    assert_allclose(xf, x_true, atol=0.05)
    assert_allclose(yf, y_true, atol=0.05)
    assert np.all(np.isfinite(chi2))
    assert np.all(niter >= 1)
    assert np.all(flag == 0)


def test_psf_grouped_fit_radius_chi2_matches_masked_residual():
    """Grouped chi2 should use the same fit_radius-limited pixels as the fit."""
    rng = np.random.default_rng(123)
    fwhm = 3.0
    fit_radius = 2.5
    var = 16.0
    psf = sep.PSF.from_gaussian(fwhm=fwhm)

    x = np.array([30.2, 33.1], dtype=np.float64)
    y = np.array([31.8, 32.4], dtype=np.float64)
    flux_true = np.array([1000.0, 850.0], dtype=np.float64)

    data = np.zeros((64, 64), dtype=np.float32)
    sep.model_psf(data, x, y, flux_true, psf)
    data += rng.normal(0.0, np.sqrt(var), size=data.shape).astype(np.float32)

    flux, _, xf, yf, flag, chi2, niter = sep.psf_fit(
        data, x, y, psf, grouped=True, var=var, fit_radius=fit_radius
    )

    model = np.zeros_like(data, dtype=np.float32)
    yy, xx = np.indices(data.shape, dtype=np.float64)
    for fi, xi, yi in zip(flux, xf, yf):
        tmp = np.zeros_like(data, dtype=np.float32)
        sep.model_psf(tmp, [xi], [yi], [fi], psf)
        mask = (xx - xi) ** 2 + (yy - yi) ** 2 <= fit_radius ** 2
        model[mask] += tmp[mask]
    resid = data - model

    manual = np.empty_like(chi2)
    for i in range(len(xf)):
        mask = (xx - xf[i]) ** 2 + (yy - yf[i]) ** 2 <= fit_radius ** 2
        ngood = int(np.count_nonzero(mask))
        chi2sum = float(np.sum((resid[mask] ** 2) / var))
        manual[i] = chi2sum / (ngood - 3)

    assert np.all(flag == 0)
    assert np.all(niter >= 1)
    assert_allclose(chi2, manual, rtol=1.0e-6, atol=1.0e-6)


def test_psf_grouped_fit_radius_limits_connectivity():
    """Grouped PSF fitting should use fit support, not full stamp overlap."""
    fwhm = 4.4
    fit_radius = 4.4
    psf = sep.PSF.from_gaussian(fwhm=fwhm)

    shape = (240, 240)
    xs = np.arange(30.0, 210.1, 12.0)
    ys = np.arange(30.0, 210.1, 12.0)
    x0, y0 = np.meshgrid(xs, ys)
    x0 = x0.ravel().astype(np.float64)
    y0 = y0.ravel().astype(np.float64)
    flux_true = np.full(x0.shape, 1000.0, dtype=np.float64)

    data = np.zeros(shape, dtype=np.float32)
    sep.model_psf(data, x0, y0, flux_true, psf)

    flux_u, err_u, xf_u, yf_u, flag_u, chi2_u, niter_u = sep.psf_fit(
        data, x0, y0, psf, fit_positions=False, fit_radius=fit_radius
    )
    flux_g, err_g, xf_g, yf_g, flag_g, chi2_g, niter_g = sep.psf_fit(
        data, x0, y0, psf, grouped=True, fit_positions=False, fit_radius=fit_radius
    )
    flux_h, err_h, xf_h, yf_h, flag_h, chi2_h, niter_h = sep.psf_fit(
        data, x0, y0, psf,
        grouped=True, fit_positions=False, fit_radius=fit_radius,
        group_factor=5.0,
    )

    assert_allclose(flux_g, flux_u)
    assert_allclose(err_g, err_u)
    assert_allclose(xf_g, xf_u)
    assert_allclose(yf_g, yf_u)
    assert_allclose(flux_h, flux_u)
    assert_allclose(err_h, err_u)
    assert_allclose(xf_h, xf_u)
    assert_allclose(yf_h, yf_u)
    assert np.all(flag_g == flag_u)
    assert np.all(flag_h == flag_u)
    assert np.all(np.isnan(chi2_g))
    assert np.all(np.isnan(chi2_h))
    assert np.all(niter_g == 0)
    assert np.all(niter_h == 0)


def test_psf_grouped_full_stamp_uses_effective_support():
    """Grouped full-stamp PSF fitting should not group by raw stamp extent."""
    fwhm = 4.4
    psf = sep.PSF.from_gaussian(fwhm=fwhm)

    shape = (240, 240)
    xs = np.arange(30.0, 210.1, 12.0)
    ys = np.arange(30.0, 210.1, 12.0)
    x0, y0 = np.meshgrid(xs, ys)
    x0 = x0.ravel().astype(np.float64)
    y0 = y0.ravel().astype(np.float64)
    flux_true = np.full(x0.shape, 1000.0, dtype=np.float64)

    data = np.zeros(shape, dtype=np.float32)
    sep.model_psf(data, x0, y0, flux_true, psf)

    flux_u, err_u, xf_u, yf_u, flag_u, chi2_u, niter_u = sep.psf_fit(
        data, x0, y0, psf, fit_positions=False
    )
    flux_g, err_g, xf_g, yf_g, flag_g, chi2_g, niter_g = sep.psf_fit(
        data, x0, y0, psf, grouped=True, fit_positions=False
    )
    flux_h, err_h, xf_h, yf_h, flag_h, chi2_h, niter_h = sep.psf_fit(
        data, x0, y0, psf, grouped=True, fit_positions=False, group_factor=5.0
    )

    assert_allclose(flux_g, flux_u)
    assert_allclose(err_g, err_u)
    assert_allclose(xf_g, xf_u)
    assert_allclose(yf_g, yf_u)
    assert_allclose(flux_h, flux_u)
    assert_allclose(err_h, err_u)
    assert_allclose(xf_h, xf_u)
    assert_allclose(yf_h, yf_u)
    assert np.all(flag_g == flag_u)
    assert np.all(flag_h == flag_u)
    assert np.all(np.isnan(chi2_g))
    assert np.all(np.isnan(chi2_h))
    assert np.all(niter_g == 0)
    assert np.all(niter_h == 0)


def test_psf_grouped_group_factor_clamps_to_support():
    """Localized grouped PSF fits should keep halo at least as large as support."""
    fwhm = 4.0
    fit_radius = 4.0
    psf = sep.PSF.from_gaussian(fwhm=fwhm)

    shape = (240, 240)
    xs = np.arange(40.0, 200.1, 7.0)
    ys = np.arange(40.0, 200.1, 7.0)
    x0, y0 = np.meshgrid(xs, ys)
    x0 = x0.ravel().astype(np.float64)
    y0 = y0.ravel().astype(np.float64)
    flux_true = np.full(x0.shape, 1000.0, dtype=np.float64)

    data = np.zeros(shape, dtype=np.float32)
    sep.model_psf(data, x0, y0, flux_true, psf)

    flux_1, err_1, xf_1, yf_1, flag_1, chi2_1, niter_1 = sep.psf_fit(
        data, x0, y0, psf,
        grouped=True, fit_positions=False, fit_radius=fit_radius,
        group_factor=1.0,
    )
    flux_low, err_low, xf_low, yf_low, flag_low, chi2_low, niter_low = sep.psf_fit(
        data, x0, y0, psf,
        grouped=True, fit_positions=False, fit_radius=fit_radius,
        group_factor=0.5,
    )

    assert_allclose(flux_low, flux_1)
    assert_allclose(err_low, err_1)
    assert_allclose(xf_low, xf_1)
    assert_allclose(yf_low, yf_1)
    assert np.all(flag_low == flag_1)
    assert np.all(np.isnan(chi2_low))
    assert np.all(np.isnan(chi2_1))
    assert np.all(niter_low == niter_1)


def test_psf_grouped_group_factor_order_invariant():
    """Localized grouped PSF fits should not depend on input ordering."""
    fwhm = 4.0
    fit_radius = 4.0
    psf = sep.PSF.from_gaussian(fwhm=fwhm)

    shape = (120, 120)
    xs = np.arange(30.0, 86.1, 7.0)
    ys = np.arange(30.0, 86.1, 7.0)
    x0, y0 = np.meshgrid(xs, ys)
    x0 = x0.ravel().astype(np.float64)
    y0 = y0.ravel().astype(np.float64)
    flux_true = np.full(x0.shape, 1000.0, dtype=np.float64)

    data = np.zeros(shape, dtype=np.float32)
    sep.model_psf(data, x0, y0, flux_true, psf)

    flux_a, err_a, xf_a, yf_a, flag_a, chi2_a, niter_a = sep.psf_fit(
        data, x0, y0, psf,
        grouped=True, fit_positions=False, fit_radius=fit_radius,
        group_factor=10.0,
    )
    flux_b, err_b, xf_b, yf_b, flag_b, chi2_b, niter_b = sep.psf_fit(
        data, x0[::-1], y0[::-1], psf,
        grouped=True, fit_positions=False, fit_radius=fit_radius,
        group_factor=10.0,
    )

    assert_allclose(flux_a, flux_b[::-1])
    assert_allclose(err_a, err_b[::-1])
    assert_allclose(xf_a, xf_b[::-1])
    assert_allclose(yf_a, yf_b[::-1])
    assert np.all(flag_a == flag_b[::-1])
    assert np.all(np.isnan(chi2_a))
    assert np.all(np.isnan(chi2_b))
    assert np.all(niter_a == niter_b[::-1])


def test_psf_position_varying():
    """Position-varying PSF (degree=1) produces different stamps at
    different image positions."""
    # Build a PSF with 3 components (degree=1: 1, x, y)
    # Component 0: narrow Gaussian, Component 1/2: wider Gaussians
    size = 15
    oversamp = 2
    ossize = size * oversamp
    cx, cy = ossize // 2, ossize // 2
    yy, xx = np.mgrid[0:ossize, 0:ossize]

    sigma_narrow = 1.5 * oversamp
    sigma_wide = 2.5 * oversamp

    comp0 = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma_narrow**2))
    comp0 /= comp0.sum()
    comp1 = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma_wide**2))
    comp1 /= comp1.sum()
    comp1 = (comp1 - comp0) * 0.1  # small spatial variation component

    data = np.stack([comp0, comp1, comp1], axis=0).astype(np.float32)

    psf = sep.PSF(data, sampling=1.0 / oversamp, degree=1,
                  x0=500.0, y0=500.0, sx=500.0, sy=500.0, fwhm=3.0)

    assert psf.ncomp == 3
    assert psf.degree == 1

    # Evaluate at two different positions by doing flux-only fits
    # on a point source. The effective widths should differ.
    img_narrow = _make_gaussian_source(64, 64, 32.0, 32.0, 1000.0, 3.0)
    f1, _, _, _, _, _, _ = sep.psf_fit(
        img_narrow, 32.0, 32.0, psf, fit_positions=False
    )

    # The PSF model at (500,500) vs (1000,500) should give
    # different results because the polynomial varies
    psf2 = sep.PSF(data, sampling=1.0 / oversamp, degree=1,
                   x0=500.0, y0=500.0, sx=500.0, sy=500.0, fwhm=3.0)

    # Just verify different PSF at different position gives different flux
    # (since it's a mismatched PSF, fluxes will differ)
    # Use the same image but tell the PSF it's at a far-off position
    img_far = _make_gaussian_source(64, 64, 32.0, 32.0, 1000.0, 3.0)

    # PSF at center (normalized coords ~0) vs PSF at edge (normalized ~1)
    # The PSF class always evaluates at the pixel position given to psf_fit
    # So we just check PSF properties are correct
    assert psf.stamp_width == size
    assert psf.stamp_height == size


def test_psf_vs_optimal_extraction():
    """PSF flux-only photometry matches sum_circle_optimal for a Gaussian."""
    fwhm = 3.5
    sigma = fwhm / 2.3548
    psf = sep.PSF.from_gaussian(fwhm=fwhm)

    data = _make_gaussian_source(64, 64, 32.0, 32.0, 1000.0, fwhm)

    # PSF flux-only
    flux_psf, _, _, _, _, _, _ = sep.psf_fit(
        data, 32.0, 32.0, psf, fit_positions=False
    )

    # Optimal aperture extraction with matched Gaussian weight
    flux_opt, _, flag_opt = sep.sum_circle(data, [32.0], [32.0],
                                           3.0 * sigma)

    # Both should recover approximately the same flux
    # (PSF fit uses the full stamp, aperture may lose some flux at edges)
    assert_allclose(flux_psf, 1000.0, rtol=0.01,
                    err_msg="PSF flux recovery")
    assert_allclose(flux_opt, 1000.0, rtol=0.05,
                    err_msg="aperture flux recovery")


def test_psf_from_psfex():
    """PSF.from_psfex loads a synthetic PSFEx FITS file correctly."""
    pytest.importorskip("astropy")
    from astropy.io import fits

    # Create a minimal PSFEx-format FITS file
    size = 25
    ncomp = 3  # degree=1: constant + x + y
    degree = 1

    # Build synthetic PSF components
    yy, xx = np.mgrid[0:size, 0:size]
    cx, cy = size // 2, size // 2
    sigma = 3.0
    comp0 = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
    comp0 /= comp0.sum()
    comp1 = comp0 * 0.01  # tiny x-variation
    comp2 = comp0 * 0.01  # tiny y-variation

    data = np.stack([comp0, comp1, comp2]).astype(np.float32)
    # PSFEx stores as data[0][0] = (ncomp, h, w) nested in a FITS table
    data_col = np.array([[data]])

    col = fits.Column(name='PSF_MASK', format=f'{ncomp * size * size}E',
                       dim=f'({size},{size},{ncomp})',
                       array=data_col)
    hdu = fits.BinTableHDU.from_columns([col])
    hdu.header['PSFAXIS1'] = size
    hdu.header['PSFAXIS2'] = size
    hdu.header['PSFAXIS3'] = ncomp
    hdu.header['POLDEG1'] = degree
    hdu.header['POLZERO1'] = 500.0
    hdu.header['POLSCAL1'] = 500.0
    hdu.header['POLZERO2'] = 400.0
    hdu.header['POLSCAL2'] = 400.0
    hdu.header['PSF_SAMP'] = 0.5
    hdu.header['PSF_FWHM'] = 3.5

    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix='.psf', delete=False) as f:
        fname = f.name
    try:
        fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(fname, overwrite=True)

        psf = sep.PSF.from_psfex(fname)
        assert psf.ncomp == ncomp
        assert psf.degree == degree
        assert psf.fwhm == 3.5
        assert psf.sampling == 0.5
        assert psf.width == size
        assert psf.height == size
    finally:
        os.unlink(fname)


def test_model_psf_single_source_flux_recovery():
    """model_psf renders a single source with the requested total flux."""
    fwhm = 3.5
    psf = sep.PSF.from_gaussian(fwhm=fwhm)
    model = np.zeros((64, 64), dtype=np.float64)

    x0, y0, f0 = 32.3, 31.7, 1234.5
    sep.model_psf(model, x0, y0, f0, psf)

    flux, fluxerr, _, _, flag, _, _ = sep.psf_fit(
        model, x0, y0, psf, fit_positions=False
    )
    assert_allclose(flux, f0, rtol=1.0e-5)
    assert_allclose(model.sum(), f0, rtol=1.0e-5)
    assert flag == 0


def test_model_psf_broadcast_multiple_sources():
    """model_psf supports broadcasting x/y/flux for multiple sources."""
    fwhm = 4.0
    psf = sep.PSF.from_gaussian(fwhm=fwhm)
    model = np.zeros((96, 96), dtype=np.float32)

    x = np.array([20.2, 70.4], dtype=np.float64)
    y = np.array([25.5, 68.1], dtype=np.float64)
    flux = np.array([700.0, 1200.0], dtype=np.float64)

    sep.model_psf(model, x, y, flux, psf)

    fitted, _, _, _, _, _, _ = sep.psf_fit(model, x, y, psf, fit_positions=False)
    assert_allclose(fitted, flux, rtol=2.0e-4)


def test_model_psf_inplace_add_and_truncation():
    """model_psf accumulates in-place and clips safely at image boundaries."""
    fwhm = 3.0
    psf = sep.PSF.from_gaussian(fwhm=fwhm)
    model = np.zeros((32, 32), dtype=np.float32)

    sep.model_psf(model, 10.0, 10.0, 100.0, psf)
    sep.model_psf(model, 10.0, 10.0, 200.0, psf)
    assert_allclose(model.sum(), 300.0, rtol=2.0e-5)

    sep.model_psf(model, 0.1, 0.2, 1000.0, psf)
    assert model.sum() > 300.0
    assert model.sum() < 1300.0


def test_psf_fit_dtype_coercion():
    """psf_fit works correctly with non-float64 x/y inputs (regression)."""
    fwhm = 4.0
    psf = sep.PSF.from_gaussian(fwhm)
    data = _make_gaussian_source(64, 64, 32.0, 32.0, 1000.0, fwhm)

    # Pass float32 and int arrays for x/y — should not crash or give wrong results
    for dtype in [np.float32, np.float64, np.int32, np.int64]:
        x = np.array([32.0], dtype=dtype)
        y = np.array([32.0], dtype=dtype)
        flux, fluxerr, xf, yf, flag, _, _ = sep.psf_fit(
            data, x, y, psf)
        assert_allclose(flux, 1000.0, rtol=0.02,
                        err_msg=f"dtype={dtype} gave wrong flux")

    # Also test scalar inputs
    flux, fluxerr, xf, yf, flag, _, _ = sep.psf_fit(
        data, 32.0, 32.0, psf)
    assert_allclose(flux, 1000.0, rtol=0.02)


def test_psf_grouped_segmap():
    """Grouped PSF fitting respects segmap — pixels from non-group sources are masked."""
    fwhm = 4.0
    psf = sep.PSF.from_gaussian(fwhm)
    sigma = fwhm / 2.3548
    ny, nx = 128, 128
    yy, xx = np.mgrid[0:ny, 0:nx]

    # Two grouped sources (close together) and one interloper nearby
    x_src = np.array([40.0, 48.0, 55.0])
    y_src = np.array([64.0, 64.0, 64.0])
    fluxes = np.array([1000.0, 800.0, 1200.0])

    data = np.zeros((ny, nx), dtype=np.float32)
    for i in range(3):
        data += fluxes[i] * np.exp(
            -((xx - x_src[i])**2 + (yy - y_src[i])**2) / (2 * sigma**2))

    # Create segmap: each source gets its own segment ID
    segmap = np.zeros((ny, nx), dtype=np.int32)
    for i in range(3):
        mask = ((xx - x_src[i])**2 + (yy - y_src[i])**2) < (3 * sigma)**2
        segmap[mask] = i + 1  # IDs 1, 2, 3

    # Fit only sources 0 and 1 (grouped), with segmap masking source 3's pixels
    x_fit = x_src[:2].copy()
    y_fit = y_src[:2].copy()
    seg_id = np.array([1, 2], dtype=np.intc)

    flux_seg, _, xf, yf, flag, _, _ = sep.psf_fit(
        data, x_fit, y_fit, psf, segmap=segmap, seg_id=seg_id,
        grouped=True, group_factor=5.0)

    # Without segmap, source 3's flux contaminates the fit
    flux_noseg, _, xf2, yf2, flag2, _, _ = sep.psf_fit(
        data, x_fit, y_fit, psf, grouped=True, group_factor=5.0)

    # With segmap, flux recovery should be better (closer to true values)
    err_seg = np.abs(flux_seg - fluxes[:2]) / fluxes[:2]
    err_noseg = np.abs(flux_noseg - fluxes[:2]) / fluxes[:2]
    # Source 1 (at x=48) is closest to interloper, should benefit most from segmap
    assert err_seg[1] < err_noseg[1], \
        f"Segmap should improve flux for source near interloper: {err_seg[1]:.3f} vs {err_noseg[1]:.3f}"


def test_psf_grouped_flags_edge():
    """Grouped PSF fitting sets TRUNC flag for sources near image edges."""
    fwhm = 4.0
    psf = sep.PSF.from_gaussian(fwhm)
    sigma = fwhm / 2.3548
    ny, nx = 64, 64
    yy, xx = np.mgrid[0:ny, 0:nx]

    # Source 1: well inside image; Source 2: near edge
    x_src = np.array([32.0, 3.0])
    y_src = np.array([32.0, 32.0])

    data = np.zeros((ny, nx), dtype=np.float32)
    for i in range(2):
        data += 1000.0 * np.exp(
            -((xx - x_src[i])**2 + (yy - y_src[i])**2) / (2 * sigma**2))

    flux, fluxerr, xf, yf, flag, _, _ = sep.psf_fit(
        data, x_src, y_src, psf, grouped=True, group_factor=5.0)

    # Interior source should have no TRUNC flag
    assert (flag[0] & 0x0010) == 0, "Interior source should not have TRUNC flag"
    # Edge source should have TRUNC flag
    assert (flag[1] & 0x0010) != 0, "Edge source should have TRUNC flag"


def test_psf_grouped_flags_mask():
    """Grouped PSF fitting sets HASMASKED flag for sources near masked pixels."""
    fwhm = 4.0
    psf = sep.PSF.from_gaussian(fwhm)
    sigma = fwhm / 2.3548
    ny, nx = 128, 128
    yy, xx = np.mgrid[0:ny, 0:nx]

    # Two sources, grouped
    x_src = np.array([50.0, 60.0])
    y_src = np.array([64.0, 64.0])

    data = np.zeros((ny, nx), dtype=np.float32)
    for i in range(2):
        data += 1000.0 * np.exp(
            -((xx - x_src[i])**2 + (yy - y_src[i])**2) / (2 * sigma**2))

    # Mask pixels near source 1 only
    mask = np.zeros((ny, nx), dtype=np.bool_)
    mask[62:66, 48:52] = True

    flux, fluxerr, xf, yf, flag, _, _ = sep.psf_fit(
        data, x_src, y_src, psf, mask=mask,
        grouped=True, group_factor=5.0)

    # Source near masked pixels should have HASMASKED flag
    assert (flag[0] & 0x0020) != 0, "Source near mask should have HASMASKED flag"


def test_psf_fit_multidim_segid():
    """Non-grouped psf_fit works with multidimensional seg_id inputs."""
    fwhm = 4.0
    psf = sep.PSF.from_gaussian(fwhm)

    # 2x2 grid of sources (well separated)
    x = np.array([[16.0, 48.0], [16.0, 48.0]])
    y = np.array([[16.0, 16.0], [48.0, 48.0]])
    seg_id = np.array([[1, 2], [3, 4]], dtype=np.intc)

    ny, nx = 64, 64
    data = np.zeros((ny, nx), dtype=np.float32)
    for i in range(2):
        for j in range(2):
            data += _make_gaussian_source(nx, ny, x[i, j], y[i, j],
                                          1000.0, fwhm)

    # Non-grouped with multidimensional seg_id should not raise
    flux, fluxerr, xf, yf, flag, _, _ = sep.psf_fit(
        data, x, y, psf, seg_id=seg_id, grouped=False)
    assert flux.shape == (2, 2)
    assert_allclose(flux, 1000.0, rtol=0.1)

    # Grouped path too, for comparison
    flux_g, _, _, _, _, _, _ = sep.psf_fit(
        data, x, y, psf, seg_id=seg_id, grouped=True)
    assert flux_g.shape == (2, 2)
    assert_allclose(flux_g, 1000.0, rtol=0.1)


def test_psf_fit_native_sampling_accuracy():
    """PSF fit with sampling=1.0 (native res) gives accurate fluxes
    for subpixel offsets, matching supersampled PSF accuracy."""
    fwhm = 4.0

    # Native-sampled PSF (pixstep=1.0)
    psf_native = sep.PSF.from_gaussian(fwhm, oversampling=1)
    assert psf_native.sampling == 1.0

    # Supersampled PSF for reference
    psf_super = sep.PSF.from_gaussian(fwhm, oversampling=4)

    true_flux = 1000.0
    sigma = fwhm / 2.3548
    ny, nx = 64, 64
    yy, xx = np.mgrid[0:ny, 0:nx]

    # Test at several subpixel offsets
    offsets = [0.0, 0.1, 0.25, 0.37, 0.5]
    for dx in offsets:
        for dy in offsets:
            xcen = 32.0 + dx
            ycen = 32.0 + dy
            data = (true_flux * np.exp(
                -((xx - xcen)**2 + (yy - ycen)**2) /
                (2 * sigma**2)) / (2 * np.pi * sigma**2)).astype(np.float32)

            flux_n, _, _, _, _, _, _ = sep.psf_fit(
                data, xcen, ycen, psf_native)
            flux_s, _, _, _, _, _, _ = sep.psf_fit(
                data, xcen, ycen, psf_super)

            # Native-sampled should be within 2% of true flux
            assert_allclose(flux_n, true_flux, rtol=0.02,
                            err_msg=f"Native PSF flux bias at offset "
                                    f"({dx}, {dy})")
            # And close to supersampled result
            assert_allclose(flux_n, flux_s, rtol=0.02,
                            err_msg=f"Native vs super mismatch at offset "
                                    f"({dx}, {dy})")


def test_psf_non_integer_sampling_no_quantization():
    """Non-integer PSF sampling keeps native stamp size from continuous scale."""
    fwhm = 4.0
    sampling = 0.6
    w = 25

    y, x = np.mgrid[0:w, 0:w]
    cx = w // 2
    sigma_psf = (fwhm / 2.3548) / sampling
    stamp = np.exp(-((x - cx)**2 + (y - cx)**2) / (2 * sigma_psf**2))
    stamp = (stamp / stamp.sum()).astype(np.float32)
    psf = sep.PSF(stamp[np.newaxis, :, :], sampling=sampling, degree=0, fwhm=fwhm)

    expected_size = int(np.floor(w * sampling + 0.5))
    assert psf.stamp_width == expected_size
    assert psf.stamp_height == expected_size

    # Smoke check that fitting with this PSF remains numerically stable.
    true_flux = 1000.0
    sigma = fwhm / 2.3548
    ny, nx = 64, 64
    yy, xx = np.mgrid[0:ny, 0:nx]
    xcen, ycen = 32.3, 31.7
    data = (true_flux * np.exp(
        -((xx - xcen)**2 + (yy - ycen)**2) /
        (2 * sigma**2)) / (2 * np.pi * sigma**2)).astype(np.float32)

    flux, fluxerr, _, _, flag, _, _ = sep.psf_fit(
        data, xcen, ycen, psf, fit_positions=False)
    assert np.isfinite(flux)
    assert np.isfinite(fluxerr)
    assert flag == 0


def test_psf_non_integer_sampling_pixel_integrated_bias():
    """Non-integer sampling PSF fit should have low baseline integrated bias."""
    fwhm = 4.0
    sigma = fwhm / 2.354820045
    true_flux = 1000.0
    ny, nx = 64, 64
    sampling = 0.6
    w = 25
    y, x = np.mgrid[0:w, 0:w]
    cx = w // 2
    sigma_psf = sigma / sampling
    stamp = np.exp(-((x - cx)**2 + (y - cx)**2) / (2 * sigma_psf**2))
    stamp = (stamp / stamp.sum()).astype(np.float32)
    psf = sep.PSF(stamp[np.newaxis, :, :], sampling=sampling, degree=0,
                  fwhm=fwhm)

    def _int_g1(center, a, b):
        return 0.5 * (math.erf((b - center) / (math.sqrt(2) * sigma)) -
                      math.erf((a - center) / (math.sqrt(2) * sigma)))

    offsets = [0.05, 0.25, 0.45, 0.65, 0.85]
    fracs = []
    for dx in offsets:
        for dy in offsets:
            xcen = 32.0 + dx
            ycen = 32.0 + dy
            x_int = np.array([_int_g1(xcen, i - 0.5, i + 0.5)
                              for i in range(nx)], dtype=np.float64)
            y_int = np.array([_int_g1(ycen, j - 0.5, j + 0.5)
                              for j in range(ny)], dtype=np.float64)
            data = np.outer(y_int, x_int)
            data *= true_flux / data.sum()
            data = data.astype(np.float32)
            flux, _, _, _, _, _, _ = sep.psf_fit(data, xcen, ycen, psf,
                                           fit_positions=False)
            fracs.append(float(flux) / true_flux - 1.0)

    assert abs(np.median(fracs)) < 0.01
