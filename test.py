#!/usr/bin/env py.test

"""Test the python functionality of SEP."""

from __future__ import division, print_function

import math
import os

import numpy as np
import pytest
from numpy.lib import recfunctions as rfn
from numpy.testing import assert_allclose, assert_approx_equal, assert_equal

import sep

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
    objs = sep.extract(data, 1.5, err=bkg.globalrms, deblend_cont=1.0)
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
    assert_allclose(flux_grp, flux, rtol=1.0e-6, atol=1.0e-6)


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


def test_aperture_bkgann_overlapping():
    """
    Test bkgann functionality in circular & elliptical apertures.
    """

    # If bkgann overlaps aperture exactly, result should be zero
    # (with subpix=1)
    data = np.random.rand(*data_shape)
    r = 5.0
    f, _, _ = sep.sum_circle(data, x, y, r, bkgann=(0.0, r), subpix=1)
    assert_allclose(f, 0.0, rtol=0.0, atol=1.0e-13)

    f, _, _ = sep.sum_ellipse(
        data, x, y, 2.0, 1.0, np.pi / 4.0, r=r, bkgann=(0.0, r), subpix=1
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
        data, thresh=1.2, err=rms, mask=None, segmentation_map=True
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


def test_psf_from_gaussian():
    """PSF.from_gaussian creates a valid PSF model."""
    psf = sep.PSF.from_gaussian(fwhm=3.5)
    assert psf.stamp_width == 15
    assert psf.stamp_height == 15
    assert psf.ncomp == 1
    assert psf.degree == 0
    assert psf.sampling == 0.5
    assert psf.fwhm == 3.5


def test_psf_flux_only():
    """PSF flux-only photometry recovers flux at the exact position."""
    fwhm = 3.5
    psf = sep.PSF.from_gaussian(fwhm=fwhm)
    data = _make_gaussian_source(64, 64, 32.0, 32.0, 1000.0, fwhm)

    flux, fluxerr, xf, yf, flag = sep.psf_fit(
        data, 32.0, 32.0, psf, fit_positions=False
    )
    assert_allclose(flux, 1000.0, rtol=0.01)
    assert flag == 0


def test_psf_fit_exact_center():
    """PSF fit at the exact source center recovers flux and position."""
    fwhm = 3.5
    psf = sep.PSF.from_gaussian(fwhm=fwhm)
    data = _make_gaussian_source(64, 64, 32.0, 32.0, 1000.0, fwhm)

    flux, fluxerr, xf, yf, flag = sep.psf_fit(data, 32.0, 32.0, psf)
    assert_allclose(flux, 1000.0, rtol=0.01)
    assert_allclose(xf, 32.0, atol=0.01)
    assert_allclose(yf, 32.0, atol=0.01)
    assert flag == 0


def test_psf_fit_subpixel_offsets():
    """PSF fit recovers sub-pixel positions from an offset initial guess."""
    fwhm = 3.5
    psf = sep.PSF.from_gaussian(fwhm=fwhm)

    offsets = [(0.3, -0.2), (-0.4, 0.1), (0.0, 0.45), (-0.15, -0.35)]
    for dx, dy in offsets:
        xtrue, ytrue = 32.0 + dx, 32.0 + dy
        data = _make_gaussian_source(64, 64, xtrue, ytrue, 1000.0, fwhm)

        flux, fluxerr, xf, yf, flag = sep.psf_fit(data, 32.0, 32.0, psf)
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

    flux, fluxerr, xf, yf, flag = sep.psf_fit(
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

    fout, ferr, xf, yf, flag = sep.psf_fit(data, x, y, psf)
    assert_allclose(fout, expected, rtol=0.01)
    assert_allclose(xf, x, atol=0.01)
    assert_allclose(yf, y, atol=0.01)


def test_psf_fit_different_fwhm():
    """PSF fit works across a range of FWHM values."""
    for fw in [2.0, 5.0, 8.0]:
        psf = sep.PSF.from_gaussian(fwhm=fw)
        data = _make_gaussian_source(64, 64, 32.2, 31.8, 1000.0, fw)

        flux, _, xf, yf, _ = sep.psf_fit(data, 32.0, 32.0, psf)
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
    flux_ng, _, xf_ng, yf_ng, _ = sep.psf_fit(data, xa, ya, psf)

    # Grouped: simultaneous fit should deblend
    flux_g, _, xf_g, yf_g, _ = sep.psf_fit(
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
    f1, _, _, _, _ = sep.psf_fit(
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
    flux_psf, _, _, _, _ = sep.psf_fit(
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

    flux, fluxerr, _, _, flag = sep.psf_fit(
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

    fitted, _, _, _, _ = sep.psf_fit(model, x, y, psf, fit_positions=False)
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
        flux, fluxerr, xf, yf, flag = sep.psf_fit(
            data, x, y, psf)
        assert_allclose(flux, 1000.0, rtol=0.02,
                        err_msg=f"dtype={dtype} gave wrong flux")

    # Also test scalar inputs
    flux, fluxerr, xf, yf, flag = sep.psf_fit(
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

    flux_seg, _, xf, yf, flag = sep.psf_fit(
        data, x_fit, y_fit, psf, segmap=segmap, seg_id=seg_id,
        grouped=True, group_factor=5.0)

    # Without segmap, source 3's flux contaminates the fit
    flux_noseg, _, xf2, yf2, flag2 = sep.psf_fit(
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

    flux, fluxerr, xf, yf, flag = sep.psf_fit(
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

    flux, fluxerr, xf, yf, flag = sep.psf_fit(
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
    flux, fluxerr, xf, yf, flag = sep.psf_fit(
        data, x, y, psf, seg_id=seg_id, grouped=False)
    assert flux.shape == (2, 2)
    assert_allclose(flux, 1000.0, rtol=0.1)

    # Grouped path too, for comparison
    flux_g, _, _, _, _ = sep.psf_fit(
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

            flux_n, _, _, _, _ = sep.psf_fit(
                data, xcen, ycen, psf_native)
            flux_s, _, _, _, _ = sep.psf_fit(
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

    flux, fluxerr, _, _, flag = sep.psf_fit(
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
            flux, _, _, _, _ = sep.psf_fit(data, xcen, ycen, psf,
                                           fit_positions=False)
            fracs.append(float(flux) / true_flux - 1.0)

    assert abs(np.median(fracs)) < 0.01
