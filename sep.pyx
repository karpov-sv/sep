# This wrapper licensed under an MIT license.

"""
Source Extraction and Photometry

This module is a wrapper of the SEP C library.
"""
import numpy as np

cimport cython
cimport numpy as np
from cpython.mem cimport PyMem_Free, PyMem_Malloc
from cpython.version cimport PY_MAJOR_VERSION
from libc cimport limits
from libc.math cimport exp, isfinite, sqrt
from libc.stdlib cimport qsort

np.import_array()  # To access the numpy C-API.

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sep-x")
except PackageNotFoundError:
    # package is not installed
    __version__ = "0+unknown"

# -----------------------------------------------------------------------------
# Definitions from the SEP C library

# macro definitions from sep.h
DEF SEP_TBYTE = 11
DEF SEP_TINT = 31
DEF SEP_TFLOAT = 42
DEF SEP_TDOUBLE = 82

# input flag values (C macros)
DEF SEP_NOISE_NONE = 0
DEF SEP_NOISE_STDDEV = 1
DEF SEP_NOISE_VAR = 2

# filter types for sep_extract
DEF SEP_FILTER_CONV = 0
DEF SEP_FILTER_MATCHED = 1
# deblend methods for sep_extract
DEF SEP_DEBLEND_THRESH = 0
DEF SEP_DEBLEND_WATERSHED = 1

# Threshold types
DEF SEP_THRESH_REL = 0
DEF SEP_THRESH_ABS = 1

# input flags for aperture photometry
DEF SEP_MASK_IGNORE = 0x0004

# Output flag values accessible from python
OBJ_MERGED = np.short(0x0001)
OBJ_TRUNC = np.short(0x0002)
OBJ_DOVERFLOW = np.short(0x0004)
OBJ_SINGU = np.short(0x0008)
APER_TRUNC = np.short(0x0010)
APER_HASMASKED = np.short(0x0020)
APER_ALLMASKED = np.short(0x0040)
APER_NONPOSITIVE = np.short(0x0080)

# macro defintion from sepcore.h
# This is not part of the SEP API, but we pull it out because want to
# explicitly detect memory errors so that we can raise MemoryError().
DEF MEMORY_ALLOC_ERROR = 1

# header definitions
cdef extern from "sep.h":

    ctypedef struct sep_image:
        const void *data
        const void *noise
        const void *mask
        const void *segmap
        int dtype
        int ndtype
        int mdtype
        int sdtype
        np.int64_t *segids
        np.int64_t *idcounts
        np.int64_t numids
        np.int64_t w
        np.int64_t h
        double noiseval
        short noise_type
        double gain
        double maskthresh

    ctypedef struct sep_bkg:
        np.int64_t w
        np.int64_t h
        float globalback
        float globalrms

    ctypedef struct sep_catalog:
        np.int64_t  nobj
        float       *thresh
        np.int64_t  *npix
        np.int64_t  *tnpix
        np.int64_t  *xmin
        np.int64_t  *xmax
        np.int64_t  *ymin
        np.int64_t  *ymax
        double *x
        double *y
        double *x2
        double *y2
        double *xy
        double *errx2
        double *erry2
        double *errxy
        float  *a
        float  *b
        float  *theta
        float  *fwhm
        float  *cxx
        float  *cyy
        float  *cxy
        float  *cflux
        float  *flux
        float  *cpeak
        float  *peak
        np.int64_t  *xcpeak
        np.int64_t  *ycpeak
        np.int64_t  *xpeak
        np.int64_t  *ypeak
        short       *flag
        np.int64_t  **pix
        np.int64_t  *objectspix

    int sep_background(const sep_image *im,
                       np.int64_t bw, np.int64_t bh,
                       np.int64_t fw, np.int64_t fh,
                       double fthresh,
                       sep_bkg **bkg)

    float sep_bkg_global(const sep_bkg *bkg)
    float sep_bkg_globalrms(const sep_bkg *bkg)
    int sep_bkg_array(const sep_bkg *bkg, void *arr, int dtype)
    int sep_bkg_rmsarray(const sep_bkg *bkg, void *arr, int dtype)
    int sep_bkg_subarray(const sep_bkg *bkg, void *arr, int dtype)
    void sep_bkg_free(sep_bkg *bkg)

    int sep_extract(const sep_image *image,
                    float thresh,
                    int thresh_type,
                    int minarea,
                    float *conv,
                    np.int64_t convw, np.int64_t convh,
                    int filter_type,
                    int deblend_nthresh,
                    double deblend_cont,
                    double deblend_fwhm,
                    int deblend_method,
                    int clean_flag,
                    double clean_param,
                    sep_catalog **catalog)
    int sep_extract_with_pixels(const sep_image *image,
                                float thresh,
                                int thresh_type,
                                int minarea,
                                float *conv,
                                np.int64_t convw,
                                np.int64_t convh,
                                int filter_type,
                                int deblend_nthresh,
                                double deblend_cont,
                                double deblend_fwhm,
                                int deblend_method,
                                int clean_flag,
                                double clean_param,
                                int include_pixels,
                                sep_catalog **catalog)

    void sep_catalog_free(sep_catalog *catalog)

    int sep_sum_circle(const sep_image *image,
                       double x, double y, double r,
                       int id, int subpix, short inflags,
                       double *sum, double *sumerr, double *area, short *flag)

    int sep_sum_circle_optimal(const sep_image *image,
                               double x, double y, double r, double fwhm,
                               int id, int subpix, short inflags,
                               double *sum, double *sumerr, double *area,
                               short *flag)

    int sep_sum_circle_optimal_multi(const sep_image *image,
                                     double *x, double *y, double *r,
                                     double *fwhm, np.int64_t n,
                                     int *id, double group_factor,
                                     double halo_factor,
                                     int subpix, short inflags,
                                     double *sum, double *sumerr, double *area,
                                     short *flag)

    int sep_sum_circle_optimal_multi_bkg(const sep_image *image,
                                         double *x, double *y, double *r,
                                         double *fwhm, np.int64_t n,
                                         int *id, double group_factor,
                                         double halo_factor,
                                         int subpix, short inflags,
                                         double *bkg_mean, double *bkg_mean_err,
                                         double *bkg_weight,
                                         double *sum, double *sumerr,
                                         double *area, short *flag)

    int sep_sum_circann(const sep_image *image,
                        double x, double y, double rin, double rout,
                        int id, int subpix, short inflags,
                        double *sum, double *sumerr, double *area, short *flag)

    int sep_stats_circann(const sep_image *image,
                          double x, double y, double rin, double rout,
                          int id, int subpix, short inflags,
                          double clip_sigma, int clip_iters,
                          double *mean, double *std, double *median,
                          double *mad_std, double *mean_clip,
                          double *area, double *sumerr, short *flag)

    int sep_stats_ellipann(const sep_image *image,
                           double x, double y, double a, double b, double theta,
                           double rin, double rout, int id, int subpix,
                           short inflags, double clip_sigma, int clip_iters,
                           double *mean, double *std, double *median,
                           double *mad_std, double *mean_clip,
                           double *area, double *sumerr, short *flag)

    int sep_sum_ellipse(const sep_image *image,
                        double x, double y, double a, double b, double theta,
                        double r, int id, int subpix, short inflags,
                        double *sum, double *sumerr, double *area,
                        short *flag)

    int sep_sum_ellipann(const sep_image *image,
                         double x, double y, double a, double b,
                         double theta, double rin, double rout,
                         int id, int subpix,
                         short inflags,
                         double *sum, double *sumerr, double *area,
                         short *flag)

    int sep_flux_radius(const sep_image *image,
                        double x, double y, double rmax, int id, int subpix,
                        short inflag,
                        double *fluxtot, double *fluxfrac, int n,
                        double *r, short *flag)

    int sep_kron_radius(const sep_image *image,
                        double x, double y, double cxx, double cyy,
                        double cxy, double r, int id,
                        double *kronrad, short *flag)

    int sep_windowed(const sep_image *image,
                     double x, double y, double sig,
                     int subpix, short inflag, int id, double maxstep,
                     double maxshift,
                     double *xout, double *yout, int *niter, short *flag)
    int sep_windowed_psf(const sep_image *image, sep_psf *psf,
                         double x, double y,
                         short inflag, int id, double maxstep,
                         double maxshift,
                         double *xout, double *yout, int *niter, short *flag)
    int sep_windowed_psf_array(const sep_image *image, sep_psf *psf,
                               const double *x, const double *y, np.int64_t n,
                               const int *id, short inflag,
                               const double *maxstep,
                               const double *maxshift,
                               double *xout, double *yout,
                               int *niter, short *flag)

    int sep_ellipse_axes(double cxx, double cyy, double cxy,
                         double *a, double *b, double *theta)

    void sep_ellipse_coeffs(double a, double b, double theta,
                            double *cxx, double *cyy, double *cxy)

    void sep_set_ellipse(unsigned char *arr, np.int64_t w, np.int64_t h,
                         double x, double y,
                         double cxx, double cyy, double cxy, double r,
                         unsigned char val)

    void sep_set_extract_pixstack(size_t val)
    size_t sep_get_extract_pixstack()

    void sep_set_sub_object_limit(int val)
    int sep_get_sub_object_limit()

    ctypedef struct sep_psf:
        int w, h
        int ncomp
        int degree
        double x0, y0
        double sx, sy
        float pixstep
        double fwhm
        float *data
        float *loc
        float *resi
        int rw, rh
        double fit_radius
        double damp_snthresh

    int sep_psf_create(sep_psf **psf,
                       const float *data, int w, int h, int ncomp,
                       int degree, double x0, double y0, double sx, double sy,
                       float pixstep, double fwhm)
    void sep_psf_free(sep_psf *psf)
    int sep_psf_build(sep_psf *psf, double x, double y)
    int sep_psf_resample(sep_psf *psf, double dx, double dy)
    int sep_set_psf(void *arr, int dtype, np.int64_t w, np.int64_t h,
                    sep_psf *psf, double x, double y, double flux)
    int sep_sum_psf(const sep_image *im, sep_psf *psf,
                    double x, double y, int id, short inflag,
                    double *sum, double *sumerr, double *area, short *flag)
    int sep_psf_snr(const sep_image *im, sep_psf *psf, int local_bkg,
                    double *out)
    int sep_psf_fit(const sep_image *im, sep_psf *psf,
                    double x, double y, int id, short inflag, int maxiter,
                    double *flux, double *fluxerr,
                    double *xfit, double *yfit,
                    double *xerr, double *yerr,
                    int *niter, double *chi2, short *flag)
    int sep_psf_fit_array(const sep_image *im, sep_psf *psf,
                          const double *x, const double *y, np.int64_t n,
                          const int *id,
                          short inflag, int maxiter,
                          double *flux, double *fluxerr,
                          double *xfit, double *yfit,
                          double *xerr, double *yerr,
                          int *niter, double *chi2, short *flag)
    int sep_psf_fit_multi(const sep_image *im, sep_psf *psf,
                          const double *x, const double *y, np.int64_t n,
                          const int *id, double group_factor,
                          short inflag, int maxiter, int fit_positions,
                          double *flux, double *fluxerr,
                          double *xfit, double *yfit,
                          double *xerr, double *yerr,
                          int *niter, double *chi2, short *flag)

    void sep_get_errmsg(int status, char *errtext)
    void sep_get_errdetail(char *errtext)

cdef class PSF

# -----------------------------------------------------------------------------
# Utility functions

cdef int _get_sep_dtype(dtype) except -1:
    """Convert a numpy dtype to the corresponding SEP dtype integer code."""
    if not dtype.isnative:
        raise ValueError(
            "Input array with dtype `{0}` has non-native byte order. "
            "Only native byte order arrays are supported. "
            "To change the byte order of the array `data`, do "
            "`data = data.astype(data.dtype.newbyteorder('='))`".format(dtype))
    t = dtype.type
    if t is np.single:
        return SEP_TFLOAT
    elif t is np.bool_ or t is np.ubyte:
        return SEP_TBYTE
    elif dtype == np.double:
        return SEP_TDOUBLE
    elif dtype == np.intc:
        return SEP_TINT
    raise ValueError('input array dtype not supported: {0}'.format(dtype))


cdef int _check_array_get_dims(np.ndarray arr, np.int64_t *w, np.int64_t *h) except -1:
    """Check some things about an array and return dimensions"""

    # Raise an informative message if array is not C-contiguous
    if not arr.flags["C_CONTIGUOUS"]:
        raise ValueError("array is not C-contiguous")

    # Check that there are exactly 2 dimensions
    if arr.ndim != 2:
        raise ValueError("array must be 2-d")

    # ensure that arr dimensions are not too large for C ints.
    if arr.shape[0] <= <Py_ssize_t> limits.INT_MAX:
        h[0] = arr.shape[0]
    else:
        raise ValueError("array height  ({0:d}) greater than INT_MAX ({1:d})"
                         .format(arr.shape[0], limits.INT_MAX))
    if arr.shape[1] <= <Py_ssize_t> limits.INT_MAX:
       w[0] = arr.shape[1]
    else:
        raise ValueError("array width ({0:d}) greater than INT_MAX ({1:d})"
                         .format(arr.shape[1], limits.INT_MAX))
    return 0

cdef int _assert_ok(int status) except -1:
    """Get the SEP error message corresponding to status code"""
    cdef char *errmsg
    cdef char *errdetail

    if status == 0:
        return 0

    # First check if we have an out-of-memory error, so we don't try to
    # allocate more memory to hold the error message.
    if status == MEMORY_ALLOC_ERROR:
        raise MemoryError

    # Otherwise, get error message.
    errmsg = <char *>PyMem_Malloc(61 * sizeof(char))
    sep_get_errmsg(status, errmsg)
    pyerrmsg = <bytes> errmsg
    PyMem_Free(errmsg)

    # Get error detail.
    errdetail = <char *>PyMem_Malloc(512 * sizeof(char))
    sep_get_errdetail(errdetail)
    pyerrdetail = <bytes> errdetail
    PyMem_Free(errdetail)

    # If error detail is present, append it to the message.
    if pyerrdetail != b"":
        pyerrmsg = pyerrmsg + b": " + pyerrdetail

    # Convert string to unicode if on python 3
    if PY_MAJOR_VERSION == 3:
        msg = pyerrmsg.decode()
    else:
        msg = pyerrmsg

    raise Exception(msg)


cdef int _parse_arrays(np.ndarray data, err, var, mask, segmap,
                       sep_image *im) except -1:
    """Helper function for functions accepting data, error, mask & segmap arrays.
    Fills in an sep_image struct."""

    cdef np.int64_t ew, eh, mw, mh, sw, sh
    cdef np.uint8_t[:,:] buf, ebuf, mbuf, sbuf

    # Clear im fields we might not touch (everything besides data, dtype, w, h)
    im.noise = NULL
    im.mask = NULL
    im.segmap = NULL
    im.numids = 0
    im.ndtype = 0
    im.mdtype = 0
    im.noiseval = 0.0
    im.noise_type = SEP_NOISE_NONE
    im.gain = 0.0
    im.maskthresh = 0.0

    # Get main image info
    _check_array_get_dims(data, &(im.w), &(im.h))
    im.dtype = _get_sep_dtype(data.dtype)
    buf = data.view(dtype=np.uint8)
    im.data = <void*>&buf[0, 0]

    # Check if noise is error or variance.
    noise = None  # will point to either error or variance.
    if err is not None:
        if var is not None:
            raise ValueError("Cannot specify both err and var")
        noise = err
        im.noise_type = SEP_NOISE_STDDEV
    elif var is not None:
        noise = var
        im.noise_type = SEP_NOISE_VAR

    # parse noise
    if noise is None:
        im.noise = NULL
        im.noise_type = SEP_NOISE_NONE
        im.noiseval = 0.0
    elif isinstance(noise, np.ndarray):
        if noise.ndim == 0:
            im.noise = NULL
            im.noiseval = noise
        elif noise.ndim == 2:
            _check_array_get_dims(noise, &ew, &eh)
            if ew != im.w or eh != im.h:
                raise ValueError("size of error/variance array must match"
                                 " data")
            im.ndtype = _get_sep_dtype(noise.dtype)
            ebuf = noise.view(dtype=np.uint8)
            im.noise = <void*>&ebuf[0, 0]
        else:
            raise ValueError("error/variance array must be 0-d or 2-d")
    else:
        im.noise = NULL
        im.noiseval = noise

    # Optional input: mask
    if mask is None:
        im.mask = NULL
    else:
        _check_array_get_dims(mask, &mw, &mh)
        if mw != im.w or mh != im.h:
            raise ValueError("size of mask array must match data")
        im.mdtype = _get_sep_dtype(mask.dtype)
        mbuf = mask.view(dtype=np.uint8)
        im.mask = <void*>&mbuf[0, 0]

    # Optional input: segmap
    if segmap is None:
        im.segmap = NULL
    else:
        _check_array_get_dims(segmap, &sw, &sh)
        if sw != im.w or sh != im.h:
            raise ValueError("size of segmap array must match data")
        im.sdtype = _get_sep_dtype(segmap.dtype)
        sbuf = segmap.view(dtype=np.uint8)
        im.segmap = <void*>&sbuf[0, 0]

# -----------------------------------------------------------------------------
# Background Estimation

cdef class Background:
    """
    Background(data, mask=None, maskthresh=0.0, bw=64, bh=64,
               fw=3, fh=3, fthresh=0.0)

    Representation of spatially variable image background and noise.

    Parameters
    ----------
    data : 2-d `~numpy.ndarray`
        Data array.
    mask : 2-d `~numpy.ndarray`, optional
        Mask array, optional
    maskthresh : float, optional
        Mask threshold. This is the inclusive upper limit on the mask value
        in order for the corresponding pixel to be unmasked. For boolean
        arrays, False and True are interpreted as 0 and 1, respectively.
        Thus, given a threshold of zero, True corresponds to masked and
        False corresponds to unmasked.
    bw, bh : int, optional
        Size of background boxes in pixels. Default is 64.
    fw, fh : int, optional
        Filter width and height in boxes. Default is 3.
    fthresh : float, optional
        Filter threshold. Default is 0.0.
    """

    cdef sep_bkg *ptr      # pointer to C struct
    cdef np.dtype orig_dtype  # dtype code of original image

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self, np.ndarray data not None, np.ndarray mask=None,
                  float maskthresh=0.0, int bw=64, int bh=64,
                  int fw=3, int fh=3, float fthresh=0.0):

        cdef int status
        cdef sep_image im

        _parse_arrays(data, None, None, mask, None, &im)
        im.maskthresh = maskthresh
        status = sep_background(&im, bw, bh, fw, fh, fthresh, &self.ptr)
        _assert_ok(status)

        self.orig_dtype = data.dtype

    # Note: all initialization work is done in __cinit__. This is just here
    # for the docstring.
    def __init__(self, np.ndarray data not None, np.ndarray mask=None,
                 float maskthresh=0.0, int bw=64, int bh=64,
                 int fw=3, int fh=3, float fthresh=0.0):
        """Background(data, mask=None, maskthresh=0.0, bw=64, bh=64,
                      fw=3, fh=3, fthresh=0.0)"""
        pass

    property globalback:
        """Global background level."""
        def __get__(self):
            return sep_bkg_global(self.ptr)

    property globalrms:
        """Global background RMS."""
        def __get__(self):
            return sep_bkg_globalrms(self.ptr)

    def back(self, dtype=None, copy=None):
        """back(dtype=None)

        Create an array of the background.

        Parameters
        ----------
        dtype : `~numpy.dtype`, optional
             Data type of output array. Default is the dtype of the original
             data.

        Returns
        -------
        back : `~numpy.ndarray`
            Array with same dimensions as original data.
        """
        cdef int sep_dtype
        cdef np.uint8_t[:, :] buf

        if dtype is None:
            dtype = self.orig_dtype
        else:
            dtype = np.dtype(dtype)
        sep_dtype = _get_sep_dtype(dtype)

        result = np.empty((self.ptr.h, self.ptr.w), dtype=dtype)
        buf = result.view(dtype=np.uint8)
        status = sep_bkg_array(self.ptr, &buf[0, 0], sep_dtype)
        _assert_ok(status)

        if copy:
            return result.copy()
        else:
            return result

    def rms(self, dtype=None):
        """rms(dtype=None)

        Create an array of the background rms.

        Parameters
        ----------
        dtype : `~numpy.dtype`, optional
             Data type of output array. Default is the dtype of the original
             data.

        Returns
        -------
        rms : `~numpy.ndarray`
            Array with same dimensions as original data.
        """
        cdef int sep_dtype
        cdef np.uint8_t[:, :] buf

        if dtype is None:
            dtype = self.orig_dtype
        else:
            dtype = np.dtype(dtype)
        sep_dtype = _get_sep_dtype(dtype)

        result = np.empty((self.ptr.h, self.ptr.w), dtype=dtype)
        buf = result.view(dtype=np.uint8)
        status = sep_bkg_rmsarray(self.ptr, &buf[0, 0], sep_dtype)
        _assert_ok(status)

        return result


    def subfrom(self, np.ndarray data not None):
        """subfrom(data)

        Subtract the background from an existing array.

        Like ``data = data - bkg``, but avoids making a copy of the data.

        Parameters
        ----------
        data : `~numpy.ndarray`
            Input array, which will be updated in-place. Shape must match
            that of the original image used to measure the background.
        """

        cdef np.int64_t w, h
        cdef int status, sep_dtype
        cdef np.uint8_t[:, :] buf

        assert self.ptr is not NULL

        _check_array_get_dims(data, &w, &h)
        sep_dtype = _get_sep_dtype(data.dtype)
        buf = data.view(dtype=np.uint8)

        # ensure dimensions match original image
        if (w != self.ptr.w or h != self.ptr.h):
            raise ValueError("Data dimensions do not match background "
                             "dimensions")

        status = sep_bkg_subarray(self.ptr, &buf[0, 0], sep_dtype)
        _assert_ok(status)

    def __array__(self, dtype=None, copy=None):
        return self.back(dtype=dtype, copy=copy)

    def __rsub__(self, np.ndarray data not None):
        data = np.copy(data)
        self.subfrom(data)
        return data

    def __dealloc__(self):
        if self.ptr is not NULL:
            sep_bkg_free(self.ptr)

# -----------------------------------------------------------------------------
# Source Extraction

# This needs to match the result from extract
cdef packed struct Object:
    np.float64_t thresh
    np.int64_t npix
    np.int64_t tnpix
    np.int64_t xmin
    np.int64_t xmax
    np.int64_t ymin
    np.int64_t ymax
    np.float64_t x
    np.float64_t y
    np.float64_t x2
    np.float64_t y2
    np.float64_t xy
    np.float64_t a
    np.float64_t b
    np.float64_t theta
    np.float64_t fwhm
    np.float64_t cxx
    np.float64_t cyy
    np.float64_t cxy
    np.float64_t cflux
    np.float64_t flux
    np.float64_t cpeak
    np.float64_t peak
    np.float64_t errx2
    np.float64_t erry2
    np.float64_t errxy
    np.int64_t xcpeak
    np.int64_t ycpeak
    np.int64_t xpeak
    np.int64_t ypeak
    short flag

default_kernel = np.array([[1.0, 2.0, 1.0],
                           [2.0, 4.0, 2.0],
                           [1.0, 2.0, 1.0]], dtype=np.float32)

@cython.boundscheck(False)
@cython.wraparound(False)
def extract(np.ndarray data not None, float thresh, err=None, var=None,
            gain=None, np.ndarray mask=None, double maskthresh=0.0,
            int minarea=5,
            np.ndarray filter_kernel=default_kernel, filter_type='matched',
            int deblend_nthresh=32, double deblend_cont=0.005,
            bint clean=True, double clean_param=1.0,
            segmentation_map=None, double deblend_fwhm=0.0,
            deblend_method='threshold'):
    """extract(data, thresh, err=None, mask=None, minarea=5,
               filter_kernel=default_kernel, filter_type='matched',
               deblend_nthresh=32, deblend_cont=0.005, clean=True,
               clean_param=1.0, segmentation_map=False, deblend_fwhm=0.0,
               deblend_method='threshold')

    Extract sources from an image.

    Parameters
    ----------
    data : `~numpy.ndarray`
        Data array (2-d).
    thresh : float
        Threshold pixel value for detection. If an ``err`` or ``var`` array
        is not given, this is interpreted as an absolute threshold. If ``err``
        or ``var`` is given, this is interpreted as a relative threshold: the
        absolute threshold at pixel (j, i) will be ``thresh * err[j, i]`` or
        ``thresh * sqrt(var[j, i])``.
    err, var : float or `~numpy.ndarray`, optional
        Error *or* variance (specify at most one). This can be used to
        specify a pixel-by-pixel detection threshold; see "thresh" argument.
    gain : float, optional
        Conversion factor between data array units and poisson counts. This
        does not affect detection; it is used only in calculating Poisson
        noise contribution to uncertainty parameters such as ``errx2``. If
        not given, no Poisson noise will be added.
    mask : `~numpy.ndarray`, optional
        Mask array. ``True`` values, or numeric values greater than
        ``maskthresh``, are considered masked. Masking a pixel is equivalent
        to setting data to zero and noise (if present) to infinity, and occurs
        *before* filtering.
    maskthresh : float, optional
        Threshold for a pixel to be masked. Default is ``0.0``.
    minarea : int, optional
        Minimum number of pixels required for an object. Default is 5.
    filter_kernel : `~numpy.ndarray` or None, optional
        Filter kernel used for on-the-fly filtering (used to
        enhance detection). Default is a 3x3 array:
        [[1,2,1], [2,4,2], [1,2,1]]. Set to ``None`` to skip
        convolution.
    filter_type : {'matched', 'conv'}, optional
        Filter treatment. This affects filtering behavior when a noise
        array is supplied. ``'matched'`` (default) accounts for
        pixel-to-pixel noise in the filter kernel. ``'conv'`` is
        simple convolution of the data array, ignoring pixel-to-pixel
        noise across the kernel. Scalar noise values use the same normalized
        matched-filter statistic as constant noise arrays. ``'matched'``
        should yield better
        detection of faint sources in areas of rapidly varying noise
        (such as found in coadded images made from semi-overlapping
        exposures), and expresses the filtered threshold in S/N units.
        ``'conv'`` retains the traditional SExtractor-style threshold
        convention.
    deblend_nthresh : int, optional
        Number of thresholds used for object deblending. Default is 32.
    deblend_cont : float, optional
        Minimum contrast ratio used for object deblending. Default is 0.005.
        To entirely disable deblending, set to 1.0.
    deblend_fwhm : float, optional
        If > 0, use a fixed circular Gaussian with this FWHM (in pixels)
        when assigning ambiguous pixels during deblending. This uses a
        deterministic max-weight assignment and is less sensitive to
        moment estimates in crowded fields. Default is 0.0 (use adaptive
        shapes and stochastic assignment). When ``deblend_method='watershed'``,
        this also enforces a minimum peak separation of 0.5*FWHM.
    deblend_method : {'threshold', 'watershed'} or int, optional
        Deblending algorithm. ``'threshold'`` (default) uses the traditional
        multi-threshold method. ``'watershed'`` seeds local maxima and applies
        watershed assignment within each detection footprint using the same
        filtered detection statistic that created the footprint.
    clean : bool, optional
        Perform cleaning? Default is True.
    clean_param : float, optional
        Cleaning parameter (see SExtractor manual). Default is 1.0.
    segmentation_map : `~numpy.ndarray` or bool, optional
        If ``True``, also return a "segmentation map" giving the member
        pixels of each object. Default is False.

        *New in v1.3.0*:
        An existing segmentation map can also be supplied in
        the form of an `~numpy.ndarray`. If this is the case, then the
        object detection stage is skipped, and the objects in the
        segmentation map are analysed and extracted.

    Returns
    -------
    objects : `~numpy.ndarray`
        Extracted object parameters (structured array). Available fields are:

        * ``thresh`` (float) Threshold at object location.
        * ``npix`` (int) Number of pixels belonging to the object.
        * ``tnpix`` (int) Number of pixels above threshold (unconvolved data).
        * ``xmin``, ``xmax`` (int) Minimum, maximum x coordinates of pixels.
        * ``ymin``, ``ymax`` (int) Minimum, maximum y coordinates of pixels.
        * ``x``, ``y`` (float) object barycenter (first moments).
        * ``x2``, ``y2``, ``xy`` (float) Second moments.
        * ``errx2``, ``erry2``, ``errxy`` (float) Second moment errors.
          Note that these will be zero if error is not given.
        * ``a``, ``b``, ``theta`` (float) Ellipse parameters, scaled as
          described by Section 8.4.2 in "The Source Extractor Guide" or
          Section 10.1.5-6 of v2.13 of SExtractor's User Manual.
        * ``fwhm`` (float) Gaussian-core FWHM (pixels), computed assuming
          background-subtracted data.
        * ``cxx``, ``cyy``, ``cxy`` (float) Alternative ellipse parameters.
        * ``cflux`` (float) Sum of member pixels in convolved data.
        * ``flux`` (float) Sum of member pixels in unconvolved data.
        * ``cpeak`` (float) Peak value in convolved data.
        * ``peak`` (float) Peak value in unconvolved data.
        * ``xcpeak``, ``ycpeak`` (int) Coordinate of convolved peak pixel.
        * ``xpeak``, ``ypeak`` (int) Coordinate of unconvolved peak pixel.
        * ``flag`` (int) Extraction flags.

    segmap : `~numpy.ndarray`, optional
        Array of integers with same shape as data. Pixels not belonging to
        any object have value 0. All pixels belonging to the ``i``-th object
        (e.g., ``objects[i]``) have value ``i+1``. Only returned if
        ``segmentation_map = True | ~numpy.ndarray``.
    """

    cdef int kernelw, kernelh, status, i, j, include_pixels
    cdef int filter_typecode, thresh_type
    cdef sep_catalog *catalog = NULL
    cdef np.ndarray[Object] result
    cdef float[:, :] kernelflt
    cdef float *kernelptr
    cdef np.int32_t[:, :] segmap_buf
    cdef np.int32_t *segmap_ptr
    cdef np.int64_t *objpix
    cdef sep_image im
    cdef np.int64_t[:] idbuf, countbuf

    if not np.isfinite(thresh) or thresh <= 0.0:
        raise ValueError("thresh must be finite and greater than zero")
    if minarea < 1:
        raise ValueError("minarea must be at least 1")
    if deblend_nthresh < 1:
        raise ValueError("deblend_nthresh must be at least 1")
    if not np.isfinite(deblend_cont) or not 0.0 <= deblend_cont <= 1.0:
        raise ValueError("deblend_cont must be finite and between 0 and 1")
    if not np.isfinite(deblend_fwhm) or deblend_fwhm < 0.0:
        raise ValueError("deblend_fwhm must be finite and non-negative")
    if not np.isfinite(clean_param) or clean_param <= 0.0:
        raise ValueError("clean_param must be finite and greater than zero")

    # parse arrays
    if type(segmentation_map) is np.ndarray:
        _parse_arrays(data, err, var, mask, segmentation_map, &im)

        ids, counts = np.unique(segmentation_map, return_counts=True)

        # Remove non-object IDs:
        filter_ids = ids>0
        segids = np.ascontiguousarray(ids[filter_ids].astype(dtype=np.int64))
        idcounts = np.ascontiguousarray(counts[filter_ids].astype(dtype=np.int64))
        if np.nansum(idcounts)>get_extract_pixstack():
            raise ValueError(
                f"The number of object pixels ({np.nansum(idcounts)}) in "
                "the segmentation map exceeds the allocated pixel stack "
                f"({get_extract_pixstack()}). Use "
                "`sep.set_extract_pixstack()` to increase the size, "
                "or check that the correct segmentation map has been "
                "supplied."
            )

        idbuf = segids.view(dtype=np.int64)
        countbuf = idcounts.view(dtype=np.int64)
        im.segids = <np.int64_t*>&idbuf[0]
        im.idcounts = <np.int64_t*>&countbuf[0]
        im.numids = len(segids)
    else:
        _parse_arrays(data, err, var, mask, None, &im)
    im.maskthresh = maskthresh
    if gain is not None:
        im.gain = gain

    # Parse filter input
    if filter_kernel is None:
        kernelptr = NULL
        kernelw = 0
        kernelh = 0
    else:
        if (filter_kernel.ndim != 2 or filter_kernel.shape[0] == 0 or
                filter_kernel.shape[1] == 0):
            raise ValueError("filter_kernel must be a non-empty 2-d array")
        if not np.all(np.isfinite(filter_kernel)):
            raise ValueError("filter_kernel values must all be finite")
        if np.sum(np.abs(filter_kernel), dtype=np.float64) == 0.0:
            raise ValueError("filter_kernel must contain a nonzero value")
        kernelflt = filter_kernel.astype(np.float32)
        if not np.all(np.isfinite(kernelflt)):
            raise ValueError("filter_kernel values must be representable as float32")
        kernelptr = &kernelflt[0, 0]
        kernelw = kernelflt.shape[1]
        kernelh = kernelflt.shape[0]

    if filter_type == 'matched':
        filter_typecode = SEP_FILTER_MATCHED
    elif filter_type == 'conv':
        filter_typecode = SEP_FILTER_CONV
    else:
        raise ValueError("unknown filter_type: {!r}".format(filter_type))

    if isinstance(deblend_method, str):
        if deblend_method == 'threshold':
            deblend_methodcode = SEP_DEBLEND_THRESH
        elif deblend_method == 'watershed':
            deblend_methodcode = SEP_DEBLEND_WATERSHED
        else:
            raise ValueError("unknown deblend_method: {!r}".format(deblend_method))
    else:
        deblend_methodcode = int(deblend_method)
        if deblend_methodcode not in (SEP_DEBLEND_THRESH, SEP_DEBLEND_WATERSHED):
            raise ValueError("unknown deblend_method: {!r}".format(deblend_method))

    # If image has error info, the threshold is relative, otherwise
    # it is absolute.
    if im.noise_type == SEP_NOISE_NONE:
        thresh_type = SEP_THRESH_ABS
    else:
        thresh_type = SEP_THRESH_REL

    include_pixels = 1 if (type(segmentation_map) is np.ndarray or segmentation_map) else 0
    status = sep_extract_with_pixels(&im,
                                     thresh, thresh_type, minarea,
                                     kernelptr, kernelw, kernelh, filter_typecode,
                                     deblend_nthresh, deblend_cont, deblend_fwhm,
                                     deblend_methodcode,
                                     clean, clean_param, include_pixels,
                                     &catalog)
    _assert_ok(status)

    # Allocate result record array and fill it
    result = np.empty(catalog.nobj,
                      dtype=np.dtype([('thresh', np.float64),
                                      ('npix', np.int64),
                                      ('tnpix', np.int64),
                                      ('xmin', np.int64),
                                      ('xmax', np.int64),
                                      ('ymin', np.int64),
                                      ('ymax', np.int64),
                                      ('x', np.float64),
                                      ('y', np.float64),
                                      ('x2', np.float64),
                                      ('y2', np.float64),
                                      ('xy', np.float64),
                                      ('errx2', np.float64),
                                      ('erry2', np.float64),
                                      ('errxy', np.float64),
                                      ('a', np.float64),
                                      ('b', np.float64),
                                      ('theta', np.float64),
                                      ('fwhm', np.float64),
                                      ('cxx', np.float64),
                                      ('cyy', np.float64),
                                      ('cxy', np.float64),
                                      ('cflux', np.float64),
                                      ('flux', np.float64),
                                      ('cpeak', np.float64),
                                      ('peak', np.float64),
                                      ('xcpeak', np.int64),
                                      ('ycpeak', np.int64),
                                      ('xpeak', np.int64),
                                      ('ypeak', np.int64),
                                      ('flag', np.short)]))

    for i in range(catalog.nobj):
        result['thresh'][i] = catalog.thresh[i]
        result['npix'][i] = catalog.npix[i]
        result['tnpix'][i] = catalog.tnpix[i]
        result['xmin'][i] = catalog.xmin[i]
        result['xmax'][i] = catalog.xmax[i]
        result['ymin'][i] = catalog.ymin[i]
        result['ymax'][i] = catalog.ymax[i]
        result['x'][i] = catalog.x[i]
        result['y'][i] = catalog.y[i]
        result['x2'][i] = catalog.x2[i]
        result['y2'][i] = catalog.y2[i]
        result['xy'][i] = catalog.xy[i]
        result['errx2'][i] = catalog.errx2[i]
        result['erry2'][i] = catalog.erry2[i]
        result['errxy'][i] = catalog.errxy[i]
        result['a'][i] = catalog.a[i]
        result['b'][i] = catalog.b[i]
        result['theta'][i] = catalog.theta[i]
        result['fwhm'][i] = catalog.fwhm[i]
        result['cxx'][i] = catalog.cxx[i]
        result['cyy'][i] = catalog.cyy[i]
        result['cxy'][i] = catalog.cxy[i]
        result['cflux'][i] = catalog.cflux[i]
        result['flux'][i] = catalog.flux[i]
        result['cpeak'][i] = catalog.cpeak[i]
        result['peak'][i] = catalog.peak[i]
        result['xcpeak'][i] = catalog.xcpeak[i]
        result['ycpeak'][i] = catalog.ycpeak[i]
        result['xpeak'][i] = catalog.xpeak[i]
        result['ypeak'][i] = catalog.ypeak[i]
        result['flag'][i] = catalog.flag[i]

    # construct a segmentation map, if it was requested.
    if type(segmentation_map) is np.ndarray or segmentation_map:
        # Note: We have to write out `(data.shape[0], data.shape[1])` because
        # because Cython turns `data.shape` later into an int pointer when
        # the function argument is typed as np.ndarray.
        segmap = np.zeros((data.shape[0], data.shape[1]), dtype=np.int32)
        segmap_buf = segmap
        segmap_ptr = &segmap_buf[0, 0]
        for i in range(catalog.nobj):
            objpix = catalog.pix[i]
            for j in range(catalog.npix[i]):
                segmap_ptr[objpix[j]] = i + 1

    # Free the C catalog
    sep_catalog_free(catalog)

    if type(segmentation_map) is np.ndarray or segmentation_map:
        return result, segmap
    else:
        return result

# -----------------------------------------------------------------------------
# Aperture Photometry

@cython.boundscheck(False)
@cython.wraparound(False)
def sum_circle(np.ndarray data not None, x, y, r,
               var=None, err=None, gain=None, np.ndarray mask=None,
               double maskthresh=0.0,
               seg_id=None, np.ndarray segmap=None,
               bkgann=None, int subpix=5,
               double clip_sigma=3.0, int clip_iters=5):
    """sum_circle(data, x, y, r, err=None, var=None, mask=None, maskthresh=0.0,
                  segmap=None, seg_id=None,
                  bkgann=None, gain=None, subpix=5,
                  clip_sigma=3.0, clip_iters=5)

    Sum data in circular aperture(s).

    Parameters
    ----------
    data : `~numpy.ndarray`
        2-d array to be summed.

    x, y, r : array_like
        Center coordinates and radius (radii) of aperture(s).
        ``x`` corresponds to the second ("fast") axis of the input array
        and ``y`` corresponds to the first ("slow") axis.
        ``x, y = (0.0, 0.0)`` corresponds to the center of the first
        element of the array. These inputs obey numpy broadcasting rules.

    err, var : float or `~numpy.ndarray`
        Error *or* variance (specify at most one).

    mask : `~numpy.ndarray`, optional
        Mask array. If supplied, a given pixel is masked if its value
        is greater than ``maskthresh``.

    maskthresh : float, optional
        Threshold for a pixel to be masked. Default is ``0.0``.

    segmap : `~numpy.ndarray`, optional
        Segmentation image with dimensions of ``data`` and dtype ``np.int32``.
        This is an optional input and corresponds to the segmentation map
        output by `~sep.extract`.

    seg_id : array_like, optional
        Array of segmentation ids used to mask additional pixels in the image.
        Dimensions correspond to the dimensions of ``x`` and ``y``. The
        behavior differs depending on whether ``seg_id`` is negative or
        positive. If ``seg_id`` is positive, all pixels belonging to other
        objects are masked. (Pixel ``j, i`` is masked if ``seg[j, i] != seg_id
        and seg[j, i] != 0``). If ``seg_id`` is negative, all pixels other
        than those belonging to the object of interest are masked. (Pixel ``j,
        i`` is masked if ``seg[j, i] != -seg_id``).  NB: must be included if
        ``segmap`` is provided.

    bkgann : tuple, optional
        Length 2 tuple giving the inner and outer radius of a
        "background annulus". If supplied, the background is estimated
        by averaging unmasked pixels in this annulus. If ``clip_iters=0``,
        this reduces to the unclipped annulus mean (legacy behavior). If
        supplied, the inner
        and outer radii obey numpy broadcasting rules along with ``x``,
        ``y`` and ``r``.

    gain : float, optional
        Conversion factor between data array units and poisson counts,
        used in calculating poisson noise in aperture sum. If ``None``
        (default), do not add poisson noise.

    subpix : int, optional
        Subpixel sampling factor. If 0, exact overlap is calculated.
        Default is 5.

    clip_sigma : float, optional
        Sigma value for clipping when ``bkgann`` is provided. Default is 3.0.

    clip_iters : int, optional
        Maximum number of clipping iterations when ``bkgann`` is provided.
        Default is 5. Set to 0 to disable clipping.

    Returns
    -------
    sum : `~numpy.ndarray`
        The sum of the data array within the aperture.

    sumerr : `~numpy.ndarray`
        Error on the sum.

    flags : `~numpy.ndarray`
        Integer giving flags. (0 if no flags set.)

    """

    cdef double flux1, fluxerr1, area1
    cdef double bkgflux, bkgfluxerr, bkgarea
    cdef double mean, std, med, mad_std, mean_clip
    cdef short flag1, bkgflag
    cdef int status
    cdef np.broadcast it
    cdef sep_image im

    # Test for map without seg_id.  Nothing happens if seg_id supplied but
    # without segmap.
    if (segmap is not None) and (seg_id is None):
        raise ValueError('`segmap` supplied but not `seg_id`.')

    _parse_arrays(data, err, var, mask, segmap, &im)
    im.maskthresh = maskthresh
    if gain is not None:
        im.gain = gain

    # Require that inputs are float64 arrays. This has to be done because we
    # are using a broadcasting iterator below, where we need to know the type
    # in advance. There are other ways to do this, e.g., using NpyIter_Multi
    # in the numpy C-API. However, the best way to use this from cython
    # is not clear to me at this time.
    #
    # docs.scipy.org/doc/numpy/reference/c-api.iterator.html#NpyIter_MultiNew
    dt = np.dtype(np.double)
    dint = np.dtype(np.int32)

    x = np.require(x, dtype=dt)
    y = np.require(y, dtype=dt)
    r = np.require(r, dtype=dt)

    # Segmentation image and ids with same dimensions as x, y, etc.
    if seg_id is not None:
        seg_id = np.require(seg_id, dtype=dint)
        if seg_id.shape != x.shape:
            raise ValueError('Shapes of `x` and `seg_id` do not match')
    else:
        seg_id = np.zeros(len(x), dtype=dint)

    if bkgann is None:

        # allocate ouput arrays
        shape = np.broadcast(x, y, r).shape
        sum = np.empty(shape, dt)
        sumerr = np.empty(shape, dt)
        flag = np.empty(shape, np.short)

        it = np.broadcast(x, y, r, seg_id, sum, sumerr, flag)

        while np.PyArray_MultiIter_NOTDONE(it):

            status = sep_sum_circle(
                &im,
                (<double*>np.PyArray_MultiIter_DATA(it, 0))[0],
                (<double*>np.PyArray_MultiIter_DATA(it, 1))[0],
                (<double*>np.PyArray_MultiIter_DATA(it, 2))[0],
                (<int*>np.PyArray_MultiIter_DATA(it, 3))[0],
                subpix, 0,
                <double*>np.PyArray_MultiIter_DATA(it, 4),
                <double*>np.PyArray_MultiIter_DATA(it, 5),
                &area1,
                <short*>np.PyArray_MultiIter_DATA(it, 6))
            _assert_ok(status)

            # Advance the iterator
            np.PyArray_MultiIter_NEXT(it)

        return sum, sumerr, flag

    else:
        rin, rout = bkgann

        # Require float arrays (see note above)
        rin = np.require(rin, dtype=dt)
        rout = np.require(rout, dtype=dt)

        # allocate ouput arrays
        shape = np.broadcast(x, y, r, rin, rout).shape
        sum = np.empty(shape, dt)
        sumerr = np.empty(shape, dt)
        flag = np.empty(shape, np.short)

        it = np.broadcast(x, y, r, rin, rout, seg_id, sum, sumerr, flag)
        while np.PyArray_MultiIter_NOTDONE(it):
            status = sep_sum_circle(
                &im,
                (<double*>np.PyArray_MultiIter_DATA(it, 0))[0],
                (<double*>np.PyArray_MultiIter_DATA(it, 1))[0],
                (<double*>np.PyArray_MultiIter_DATA(it, 2))[0],
                (<int*>np.PyArray_MultiIter_DATA(it, 5))[0],
                subpix, 0, &flux1, &fluxerr1, &area1, &flag1)
            _assert_ok(status)

            # background subtraction
            # Note that background output flags are not used.
            if clip_iters == 0:
                status = sep_sum_circann(
                    &im,
                    (<double*>np.PyArray_MultiIter_DATA(it, 0))[0],
                    (<double*>np.PyArray_MultiIter_DATA(it, 1))[0],
                    (<double*>np.PyArray_MultiIter_DATA(it, 3))[0],
                    (<double*>np.PyArray_MultiIter_DATA(it, 4))[0],
                    (<int*>np.PyArray_MultiIter_DATA(it, 5))[0],
                    1, SEP_MASK_IGNORE, &bkgflux, &bkgfluxerr, &bkgarea, &bkgflag)
                _assert_ok(status)
                if not bkgarea > 0:
                    raise ValueError(
                        "The background annulus does not contain any valid pixels, "
                        "for the object at index "
                        f"{np.PyArray_MultiIter_INDEX(it)}."
                    )
                mean_clip = bkgflux / bkgarea
            else:
                status = sep_stats_circann(
                    &im,
                    (<double*>np.PyArray_MultiIter_DATA(it, 0))[0],
                    (<double*>np.PyArray_MultiIter_DATA(it, 1))[0],
                    (<double*>np.PyArray_MultiIter_DATA(it, 3))[0],
                    (<double*>np.PyArray_MultiIter_DATA(it, 4))[0],
                    (<int*>np.PyArray_MultiIter_DATA(it, 5))[0],
                    1, SEP_MASK_IGNORE,
                    clip_sigma, clip_iters,
                    &mean, &std, &med, &mad_std, &mean_clip,
                    &bkgarea, &bkgfluxerr, &bkgflag)
                _assert_ok(status)
                if not bkgarea > 0 or mean_clip != mean_clip:
                    raise ValueError(
                        "The background annulus does not contain any valid pixels, "
                        "for the object at index "
                        f"{np.PyArray_MultiIter_INDEX(it)}."
                    )

            if area1 > 0:
                flux1 -= mean_clip * area1
                bkgfluxerr = bkgfluxerr / bkgarea * area1
                fluxerr1 = sqrt(fluxerr1*fluxerr1 + bkgfluxerr*bkgfluxerr)
            (<double*>np.PyArray_MultiIter_DATA(it, 6))[0] = flux1
            (<double*>np.PyArray_MultiIter_DATA(it, 7))[0] = fluxerr1
            (<short*>np.PyArray_MultiIter_DATA(it, 8))[0] = flag1

            np.PyArray_MultiIter_NEXT(it)

        return sum, sumerr, flag


@cython.boundscheck(False)
@cython.wraparound(False)
def sum_circle_optimal(np.ndarray data not None, x, y, r, fwhm,
                       var=None, err=None, gain=None, np.ndarray mask=None,
                       double maskthresh=0.0,
                       seg_id=None, np.ndarray segmap=None,
                       bkgann=None, bint grouped=False, int subpix=5,
                       double clip_sigma=3.0, int clip_iters=5,
                       double group_radius_factor=1.0,
                       group_halo_factor=None):
    """sum_circle_optimal(data, x, y, r, fwhm, err=None, var=None,
                           mask=None, maskthresh=0.0,
                           segmap=None, seg_id=None,
                           bkgann=None, gain=None,
                           grouped=False, subpix=5,
                           clip_sigma=3.0, clip_iters=5,
                           group_radius_factor=1.0,
                           group_halo_factor=None)

    Optimal extraction in circular aperture(s) using a Gaussian PSF.

    Parameters are identical to `~sep.sum_circle`, with the addition of
    ``fwhm`` which sets the Gaussian PSF width used for weighting.
    ``bkgann`` may be supplied to subtract a local background annulus using
    a sigma-clipped mean. Set ``clip_iters=0`` to disable clipping.
    ``clip_sigma`` and ``clip_iters`` control the sigma-clipping parameters
    used for the annulus statistics.
    Set ``grouped=True`` to auto-group overlapping
    apertures and solve all fluxes in each group simultaneously; in this
    case the background is estimated per group from the members' annuli.
    ``group_radius_factor`` scales the grouping radius (1.0 matches the
    aperture overlap criterion). ``group_halo_factor`` scales the local
    context halo used within large grouped solves; by default it matches
    ``group_radius_factor``.
    """

    cdef double flux1, fluxerr1, area1
    cdef double bkgflux, bkgfluxerr, bkgarea
    cdef double mean, std, med, mad_std, mean_clip
    cdef short flag1, bkgflag
    cdef int status
    cdef np.broadcast it
    cdef sep_image im
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] x1, y1, r1, fwhm1
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] rin1, rout1
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] sum1, sumerr1, area_arr
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] bkg_mean_arr
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] bkg_mean_err_arr
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] bkg_weight_arr
    cdef np.ndarray[np.int32_t, ndim=1, mode="c"] seg_id1
    cdef np.ndarray[np.int16_t, ndim=1, mode="c"] flag_arr
    cdef Py_ssize_t n
    cdef Py_ssize_t i
    cdef double group_halo_factor_val

    if (segmap is not None) and (seg_id is None):
        raise ValueError('`segmap` supplied but not `seg_id`.')
    if group_radius_factor <= 0.0:
        raise ValueError('`group_radius_factor` must be positive.')
    if group_halo_factor is None:
        group_halo_factor_val = group_radius_factor
    else:
        group_halo_factor_val = float(group_halo_factor)
        if group_halo_factor_val <= 0.0:
            raise ValueError('`group_halo_factor` must be positive.')

    _parse_arrays(data, err, var, mask, segmap, &im)
    im.maskthresh = maskthresh
    if gain is not None:
        im.gain = gain

    dt = np.dtype(np.double)
    dint = np.dtype(np.int32)

    x = np.require(x, dtype=dt)
    y = np.require(y, dtype=dt)
    r = np.require(r, dtype=dt)
    fwhm = np.require(fwhm, dtype=dt)

    if seg_id is not None:
        seg_id = np.require(seg_id, dtype=dint)
        if seg_id.shape != x.shape:
            raise ValueError('Shapes of `x` and `seg_id` do not match')
    else:
        seg_id = np.zeros(len(x), dtype=dint)

    if grouped:
        if bkgann is None:
            shape = np.broadcast(x, y, r, fwhm).shape
            x1 = np.ascontiguousarray(np.broadcast_to(x, shape).ravel(),
                                      dtype=np.float64)
            y1 = np.ascontiguousarray(np.broadcast_to(y, shape).ravel(),
                                      dtype=np.float64)
            r1 = np.ascontiguousarray(np.broadcast_to(r, shape).ravel(),
                                      dtype=np.float64)
            fwhm1 = np.ascontiguousarray(np.broadcast_to(fwhm, shape).ravel(),
                                         dtype=np.float64)

            if seg_id is not None:
                if seg_id.shape != shape:
                    seg_id = np.broadcast_to(seg_id, shape)
                seg_id1 = np.ascontiguousarray(seg_id.ravel(), dtype=np.int32)
            else:
                seg_id1 = np.zeros(x1.shape[0], dtype=np.int32)

            n = x1.shape[0]
            sum1 = np.empty(n, dtype=np.float64)
            sumerr1 = np.empty(n, dtype=np.float64)
            area_arr = np.empty(n, dtype=np.float64)
            flag_arr = np.empty(n, dtype=np.int16)

            status = sep_sum_circle_optimal_multi(
                &im,
                <double*>x1.data,
                <double*>y1.data,
                <double*>r1.data,
                <double*>fwhm1.data,
                n,
                <int*>seg_id1.data,
                group_radius_factor,
                group_halo_factor_val,
                subpix,
                0,
                <double*>sum1.data,
                <double*>sumerr1.data,
                <double*>area_arr.data,
                <short*>flag_arr.data
            )
            _assert_ok(status)

            return (sum1.reshape(shape),
                    sumerr1.reshape(shape),
                    flag_arr.reshape(shape))

        rin, rout = bkgann
        rin = np.require(rin, dtype=dt)
        rout = np.require(rout, dtype=dt)

        shape = np.broadcast(x, y, r, fwhm, rin, rout).shape
        x1 = np.ascontiguousarray(np.broadcast_to(x, shape).ravel(),
                                  dtype=np.float64)
        y1 = np.ascontiguousarray(np.broadcast_to(y, shape).ravel(),
                                  dtype=np.float64)
        r1 = np.ascontiguousarray(np.broadcast_to(r, shape).ravel(),
                                  dtype=np.float64)
        fwhm1 = np.ascontiguousarray(np.broadcast_to(fwhm, shape).ravel(),
                                     dtype=np.float64)
        rin1 = np.ascontiguousarray(np.broadcast_to(rin, shape).ravel(),
                                    dtype=np.float64)
        rout1 = np.ascontiguousarray(np.broadcast_to(rout, shape).ravel(),
                                     dtype=np.float64)

        if seg_id is not None:
            if seg_id.shape != shape:
                seg_id = np.broadcast_to(seg_id, shape)
            seg_id1 = np.ascontiguousarray(seg_id.ravel(), dtype=np.int32)
        else:
            seg_id1 = np.zeros(x1.shape[0], dtype=np.int32)

        n = x1.shape[0]
        sum1 = np.empty(n, dtype=np.float64)
        sumerr1 = np.empty(n, dtype=np.float64)
        area_arr = np.empty(n, dtype=np.float64)
        flag_arr = np.empty(n, dtype=np.int16)
        bkg_mean_arr = np.empty(n, dtype=np.float64)
        bkg_mean_err_arr = np.empty(n, dtype=np.float64)
        bkg_weight_arr = np.empty(n, dtype=np.float64)

        for i in range(n):
            if clip_iters == 0:
                status = sep_sum_circann(
                    &im,
                    (<double*>x1.data)[i],
                    (<double*>y1.data)[i],
                    (<double*>rin1.data)[i],
                    (<double*>rout1.data)[i],
                    (<int*>seg_id1.data)[i],
                    1,
                    SEP_MASK_IGNORE,
                    &bkgflux,
                    &bkgfluxerr,
                    &bkgarea,
                    &bkgflag
                )
                _assert_ok(status)
                if not bkgarea > 0:
                    raise ValueError(
                        "The background annulus does not contain any valid pixels, "
                        "for the object at index "
                        f"{i}."
                    )
                mean_clip = bkgflux / bkgarea
            else:
                status = sep_stats_circann(
                    &im,
                    (<double*>x1.data)[i],
                    (<double*>y1.data)[i],
                    (<double*>rin1.data)[i],
                    (<double*>rout1.data)[i],
                    (<int*>seg_id1.data)[i],
                    1,
                    SEP_MASK_IGNORE,
                    clip_sigma,
                    clip_iters,
                    &mean,
                    &std,
                    &med,
                    &mad_std,
                    &mean_clip,
                    &bkgarea,
                    &bkgfluxerr,
                    &bkgflag
                )
                _assert_ok(status)

                if not bkgarea > 0 or mean_clip != mean_clip:
                    raise ValueError(
                        "The background annulus does not contain any valid pixels, "
                        "for the object at index "
                        f"{i}."
                    )

            bkg_mean_arr[i] = mean_clip
            bkg_weight_arr[i] = bkgarea
            bkg_mean_err_arr[i] = bkgfluxerr / bkgarea

        status = sep_sum_circle_optimal_multi_bkg(
            &im,
            <double*>x1.data,
            <double*>y1.data,
            <double*>r1.data,
            <double*>fwhm1.data,
            n,
            <int*>seg_id1.data,
            group_radius_factor,
            group_halo_factor_val,
            subpix,
            0,
            <double*>bkg_mean_arr.data,
            <double*>bkg_mean_err_arr.data,
            <double*>bkg_weight_arr.data,
            <double*>sum1.data,
            <double*>sumerr1.data,
            <double*>area_arr.data,
            <short*>flag_arr.data
        )
        _assert_ok(status)

        return (sum1.reshape(shape),
                sumerr1.reshape(shape),
                flag_arr.reshape(shape))

    if bkgann is None:
        shape = np.broadcast(x, y, r, fwhm).shape
        sum = np.empty(shape, dt)
        sumerr = np.empty(shape, dt)
        flag = np.empty(shape, np.short)

        it = np.broadcast(x, y, r, fwhm, seg_id, sum, sumerr, flag)

        while np.PyArray_MultiIter_NOTDONE(it):
            status = sep_sum_circle_optimal(
                &im,
                (<double*>np.PyArray_MultiIter_DATA(it, 0))[0],
                (<double*>np.PyArray_MultiIter_DATA(it, 1))[0],
                (<double*>np.PyArray_MultiIter_DATA(it, 2))[0],
                (<double*>np.PyArray_MultiIter_DATA(it, 3))[0],
                (<int*>np.PyArray_MultiIter_DATA(it, 4))[0],
                subpix, 0,
                <double*>np.PyArray_MultiIter_DATA(it, 5),
                <double*>np.PyArray_MultiIter_DATA(it, 6),
                &area1,
                <short*>np.PyArray_MultiIter_DATA(it, 7))
            _assert_ok(status)

            np.PyArray_MultiIter_NEXT(it)

        return sum, sumerr, flag

    else:
        rin, rout = bkgann

        rin = np.require(rin, dtype=dt)
        rout = np.require(rout, dtype=dt)

        shape = np.broadcast(x, y, r, fwhm, rin, rout).shape
        sum = np.empty(shape, dt)
        sumerr = np.empty(shape, dt)
        flag = np.empty(shape, np.short)

        it = np.broadcast(x, y, r, fwhm, rin, rout, seg_id, sum, sumerr, flag)
        while np.PyArray_MultiIter_NOTDONE(it):
            status = sep_sum_circle_optimal(
                &im,
                (<double*>np.PyArray_MultiIter_DATA(it, 0))[0],
                (<double*>np.PyArray_MultiIter_DATA(it, 1))[0],
                (<double*>np.PyArray_MultiIter_DATA(it, 2))[0],
                (<double*>np.PyArray_MultiIter_DATA(it, 3))[0],
                (<int*>np.PyArray_MultiIter_DATA(it, 6))[0],
                subpix, 0, &flux1, &fluxerr1, &area1, &flag1)
            _assert_ok(status)

            if clip_iters == 0:
                status = sep_sum_circann(
                    &im,
                    (<double*>np.PyArray_MultiIter_DATA(it, 0))[0],
                    (<double*>np.PyArray_MultiIter_DATA(it, 1))[0],
                    (<double*>np.PyArray_MultiIter_DATA(it, 4))[0],
                    (<double*>np.PyArray_MultiIter_DATA(it, 5))[0],
                    (<int*>np.PyArray_MultiIter_DATA(it, 6))[0],
                    1,
                    SEP_MASK_IGNORE,
                    &bkgflux,
                    &bkgfluxerr,
                    &bkgarea,
                    &bkgflag
                )
                _assert_ok(status)
                if not bkgarea > 0:
                    raise ValueError(
                        "The background annulus does not contain any valid pixels, "
                        "for the object at index "
                        f"{np.PyArray_MultiIter_INDEX(it)}."
                    )
                mean_clip = bkgflux / bkgarea
            else:
                status = sep_stats_circann(
                    &im,
                    (<double*>np.PyArray_MultiIter_DATA(it, 0))[0],
                    (<double*>np.PyArray_MultiIter_DATA(it, 1))[0],
                    (<double*>np.PyArray_MultiIter_DATA(it, 4))[0],
                    (<double*>np.PyArray_MultiIter_DATA(it, 5))[0],
                    (<int*>np.PyArray_MultiIter_DATA(it, 6))[0],
                    1,
                    SEP_MASK_IGNORE,
                    clip_sigma,
                    clip_iters,
                    &mean,
                    &std,
                    &med,
                    &mad_std,
                    &mean_clip,
                    &bkgarea,
                    &bkgfluxerr,
                    &bkgflag
                )
                _assert_ok(status)

                if not bkgarea > 0 or mean_clip != mean_clip:
                    raise ValueError(
                        "The background annulus does not contain any valid pixels, "
                        "for the object at index "
                        f"{np.PyArray_MultiIter_INDEX(it)}."
                    )

            if area1 > 0:
                flux1 -= mean_clip * area1
                bkgfluxerr = bkgfluxerr / bkgarea * area1
                fluxerr1 = sqrt(fluxerr1*fluxerr1 + bkgfluxerr*bkgfluxerr)
            (<double*>np.PyArray_MultiIter_DATA(it, 7))[0] = flux1
            (<double*>np.PyArray_MultiIter_DATA(it, 8))[0] = fluxerr1
            (<short*>np.PyArray_MultiIter_DATA(it, 9))[0] = flag1

            np.PyArray_MultiIter_NEXT(it)

        return sum, sumerr, flag



@cython.boundscheck(False)
@cython.wraparound(False)
def sum_circann(np.ndarray data not None, x, y, rin, rout,
                var=None, err=None, gain=None, np.ndarray mask=None,
                double maskthresh=0.0, seg_id=None, np.ndarray segmap=None,
                int subpix=5):
    """sum_circann(data, x, y, rin, rout, var=None, err=None, mask=None,
                   maskthresh=0.0, seg_id=None, segmap=None, gain=None,
                   subpix=5)

    Sum data in circular annular aperture(s).

    Parameters
    ----------
    data : `~numpy.ndarray`
        2-d array to be summed.

    x, y, rin, rout : array_like
        Center coordinates and inner and outer radii of aperture(s).
        ``x`` corresponds to the second ("fast") axis of the input array
        and ``y`` corresponds to the first ("slow") axis.
        ``x, y = (0.0, 0.0)`` corresponds to the center of the first
        element of the array. These inputs obey numpy broadcasting rules.
        It is required that ``rout >= rin >= 0.0``.

    var, err : float or ndarray
        Variance *or* error (specify at most one).

    mask : `~numpy.ndarray`, optional
        Mask array. If supplied, a given pixel is masked if its value
        is greater than ``maskthresh``.

    maskthresh : float, optional
        Threshold for a pixel to be masked. Default is ``0.0``.

    segmap : `~numpy.ndarray`, optional
        Segmentation image with dimensions of ``data`` and dtype ``np.int32``.
        This is an optional input and corresponds to the segmentation map
        output by `~sep.extract`.

    seg_id : array_like, optional
        Array of segmentation ids used to mask additional pixels in the image.
        Dimensions correspond to the dimensions of ``x`` and ``y``. The
        behavior differs depending on whether ``seg_id`` is negative or
        positive. If ``seg_id`` is positive, all pixels belonging to other
        objects are masked. (Pixel ``j, i`` is masked if ``seg[j, i] != seg_id
        and seg[j, i] != 0``). If ``seg_id`` is negative, all pixels other
        than those belonging to the object of interest are masked. (Pixel ``j,
        i`` is masked if ``seg[j, i] != -seg_id``).  NB: must be included if
        ``segmap`` is provided.

    gain : float, optional
        Conversion factor between data array units and poisson counts,
        used in calculating poisson noise in aperture sum. If ``None``
        (default), do not add poisson noise.

    subpix : int, optional
        Subpixel sampling factor. Default is 5.

    Returns
    -------
    sum : `~numpy.ndarray`
        The sum of the data array within the aperture.

    sumerr : `~numpy.ndarray`
        Error on the sum.

    flags : `~numpy.ndarray`
        Integer giving flags. (0 if no flags set.)
    """

    cdef double area1
    cdef int status
    cdef np.broadcast it
    cdef sep_image im

    # Test for segmap without seg_id.  Nothing happens if seg_id supplied but
    # without segmap.
    if (segmap is not None) and (seg_id is None):
        raise ValueError('`segmap` supplied but not `seg_id`.')

    _parse_arrays(data, err, var, mask, segmap, &im)
    im.maskthresh = maskthresh
    if gain is not None:
        im.gain = gain

    # convert inputs to double arrays
    dt = np.dtype(np.double)
    dint = np.dtype(np.int32)
    x = np.require(x, dtype=dt)
    y = np.require(y, dtype=dt)
    rin = np.require(rin, dtype=dt)
    rout = np.require(rout, dtype=dt)

    # Segmentation image and ids with same dimensions as x, y, etc.
    if seg_id is not None:
        seg_id = np.require(seg_id, dtype=dint)
        if seg_id.shape != x.shape:
            raise ValueError('Shapes of `x` and `seg_id` do not match')
    else:
        seg_id = np.zeros(len(x), dtype=dint)

    # allocate ouput arrays
    shape = np.broadcast(x, y, rin, rout).shape
    sum = np.empty(shape, dt)
    sumerr = np.empty(shape, dt)
    flag = np.empty(shape, np.short)

    # it = np.broadcast(x, y, rin, rout, sum, sumerr, flag)
    it = np.broadcast(x, y, rin, rout, seg_id, sum, sumerr, flag)

    while np.PyArray_MultiIter_NOTDONE(it):
        status = sep_sum_circann(
            &im,
            (<double*>np.PyArray_MultiIter_DATA(it, 0))[0],
            (<double*>np.PyArray_MultiIter_DATA(it, 1))[0],
            (<double*>np.PyArray_MultiIter_DATA(it, 2))[0],
            (<double*>np.PyArray_MultiIter_DATA(it, 3))[0],
            (<int*>np.PyArray_MultiIter_DATA(it, 4))[0],
            subpix, 0,
            <double*>np.PyArray_MultiIter_DATA(it, 5),
            <double*>np.PyArray_MultiIter_DATA(it, 6),
            &area1,
            <short*>np.PyArray_MultiIter_DATA(it, 7))

        _assert_ok(status)

        np.PyArray_MultiIter_NEXT(it)

    return sum, sumerr, flag


@cython.boundscheck(False)
@cython.wraparound(False)
def stats_circann(np.ndarray data not None, x, y, rin, rout,
                  var=None, err=None, gain=None, np.ndarray mask=None,
                  double maskthresh=0.0, seg_id=None, np.ndarray segmap=None,
                  int subpix=5, double clip_sigma=3.0, int clip_iters=5):
    """stats_circann(data, x, y, rin, rout, err=None, var=None, mask=None,
                     maskthresh=0.0, seg_id=None, segmap=None, gain=None,
                     subpix=5)

    Compute statistics in circular annular aperture(s).

    Parameters
    ----------
    data : `~numpy.ndarray`
        2-d array to be sampled.

    x, y, rin, rout : array_like
        Center coordinates and inner and outer radii of aperture(s).
        ``x`` corresponds to the second ("fast") axis of the input array
        and ``y`` corresponds to the first ("slow") axis.
        ``x, y = (0.0, 0.0)`` corresponds to the center of the first
        element of the array. These inputs obey numpy broadcasting rules.
        It is required that ``rout >= rin >= 0.0``.

    err, var : float or ndarray
        Error *or* variance (specify at most one). Accepted for API
        compatibility; not used in the statistics.

    mask : `~numpy.ndarray`, optional
        Mask array. If supplied, a given pixel is masked if its value
        is greater than ``maskthresh``. Masked pixels are ignored (no
        correction applied).

    maskthresh : float, optional
        Threshold for a pixel to be masked. Default is ``0.0``.

    segmap : `~numpy.ndarray`, optional
        Segmentation image with dimensions of ``data`` and dtype ``np.int32``.
        This is an optional input and corresponds to the segmentation map
        output by `~sep.extract`.

    seg_id : array_like, optional
        Array of segmentation ids used to mask additional pixels in the image.
        Dimensions correspond to the dimensions of ``x`` and ``y``. The
        behavior differs depending on whether ``seg_id`` is negative or
        positive. If ``seg_id`` is positive, all pixels belonging to other
        objects are masked. (Pixel ``j, i`` is masked if ``seg[j, i] != seg_id
        and seg[j, i] != 0``). If ``seg_id`` is negative, all pixels other
        than those belonging to the object of interest are masked. (Pixel ``j,
        i`` is masked if ``seg[j, i] != -seg_id``).  NB: must be included if
        ``segmap`` is provided.

    gain : float, optional
        Accepted for API compatibility; not used in the statistics.

    subpix : int, optional
        Subpixel sampling factor. Default is 5. If 0, exact overlap is used.

    clip_sigma : float, optional
        Sigma value for clipping. Default is 3.0.

    clip_iters : int, optional
        Maximum number of clipping iterations. Default is 5.

    Returns
    -------
    mean : `~numpy.ndarray`
        Weighted mean of the data within the aperture.

    std : `~numpy.ndarray`
        Weighted standard deviation of the data within the aperture.

    median : `~numpy.ndarray`
        Weighted median of the data within the aperture.

    mad_std : `~numpy.ndarray`
        Weighted median absolute deviation scaled by 1.4826, providing a
        robust estimate of the standard deviation for Gaussian data.

    mean_clip : `~numpy.ndarray`
        Sigma-clipped mean using median/MAD.

    flags : `~numpy.ndarray`
        Integer giving flags. (0 if no flags set.)

    Notes
    -----
    If no unmasked pixels fall within the annulus, outputs are ``NaN`` and
    the ``APER_ALLMASKED`` flag is set.
    """

    cdef int status
    cdef double area1, sumerr1
    cdef np.broadcast it
    cdef sep_image im

    # Test for segmap without seg_id.  Nothing happens if seg_id supplied but
    # without segmap.
    if (segmap is not None) and (seg_id is None):
        raise ValueError('`segmap` supplied but not `seg_id`.')

    _parse_arrays(data, err, var, mask, segmap, &im)
    im.maskthresh = maskthresh
    if gain is not None:
        im.gain = gain

    # convert inputs to double arrays
    dt = np.dtype(np.double)
    dint = np.dtype(np.int32)
    x = np.require(x, dtype=dt)
    y = np.require(y, dtype=dt)
    rin = np.require(rin, dtype=dt)
    rout = np.require(rout, dtype=dt)

    # Segmentation image and ids with same dimensions as x, y, etc.
    if seg_id is not None:
        seg_id = np.require(seg_id, dtype=dint)
        if seg_id.shape != x.shape:
            raise ValueError('Shapes of `x` and `seg_id` do not match')
    else:
        seg_id = np.zeros(len(x), dtype=dint)

    # allocate output arrays
    shape = np.broadcast(x, y, rin, rout).shape
    mean = np.empty(shape, dt)
    std = np.empty(shape, dt)
    median = np.empty(shape, dt)
    mad_std = np.empty(shape, dt)
    mean_clip = np.empty(shape, dt)
    flag = np.empty(shape, np.short)

    it = np.broadcast(x, y, rin, rout, seg_id, mean, std, median, mad_std, mean_clip, flag)

    while np.PyArray_MultiIter_NOTDONE(it):
        status = sep_stats_circann(
            &im,
            (<double*>np.PyArray_MultiIter_DATA(it, 0))[0],
            (<double*>np.PyArray_MultiIter_DATA(it, 1))[0],
            (<double*>np.PyArray_MultiIter_DATA(it, 2))[0],
            (<double*>np.PyArray_MultiIter_DATA(it, 3))[0],
            (<int*>np.PyArray_MultiIter_DATA(it, 4))[0],
            subpix, SEP_MASK_IGNORE,
            clip_sigma, clip_iters,
            <double*>np.PyArray_MultiIter_DATA(it, 5),
            <double*>np.PyArray_MultiIter_DATA(it, 6),
            <double*>np.PyArray_MultiIter_DATA(it, 7),
            <double*>np.PyArray_MultiIter_DATA(it, 8),
            <double*>np.PyArray_MultiIter_DATA(it, 9),
            &area1,
            &sumerr1,
            <short*>np.PyArray_MultiIter_DATA(it, 10))

        _assert_ok(status)

        np.PyArray_MultiIter_NEXT(it)

    return mean, std, median, mad_std, mean_clip, flag


def sum_ellipse(np.ndarray data not None, x, y, a, b, theta, r=1.0,
                var=None, err=None, gain=None, np.ndarray mask=None,
                double maskthresh=0.0,
                seg_id=None, np.ndarray segmap=None,
                bkgann=None, int subpix=5,
                double clip_sigma=3.0, int clip_iters=5):
    """sum_ellipse(data, x, y, a, b, theta, r, err=None, var=None, mask=None,
                   maskthresh=0.0, seg_id=None, segmap=None, bkgann=None,
                   gain=None, subpix=5, clip_sigma=3.0, clip_iters=5)

    Sum data in elliptical aperture(s).

    Parameters
    ----------
    data : `~numpy.ndarray`
        2-d array to be summed.

    x, y : array_like
        Center coordinates and radius (radii) of aperture(s).
        ``x`` corresponds to the second ("fast") axis of the input array
        and ``y`` corresponds to the first ("slow") axis.
        ``x, y = (0.0, 0.0)`` corresponds to the center of the first
        element of the array. These inputs obey numpy broadcasting rules.

    a, b, theta : array_like
        Ellipse parameters. These inputs, along with ``x``, ``y``, and ``r``,
        obey numpy broadcasting rules. ``a`` is the semi-major axis,
        ``b`` is the semi-minor axis and ``theta`` is angle in radians between
        the positive x axis and the major axis. It must be in the range
        ``[-pi/2, pi/2]``. It is also required that ``a >= b >= 0.0``.

    r : array_like, optional
        Scaling factor for the semi-minor and semi-major axes. The
        actual ellipse used will have semi-major axis ``a * r`` and
        semi-minor axis ``b * r``. Setting this parameter to a value
        other than 1.0 is exactly equivalent to scaling both ``a`` and
        ``b`` by the same value. Default is 1.0.

    err, var : float or `~numpy.ndarray`
        Error *or* variance (specify at most one).

    mask : `~numpy.ndarray`, optional
        Mask array. If supplied, a given pixel is masked if its value
        is greater than ``maskthresh``.

    maskthresh : float, optional
        Threshold for a pixel to be masked. Default is ``0.0``.

    segmap : `~numpy.ndarray`, optional
        Segmentation image with dimensions of ``data`` and dtype ``np.int32``.
        This is an optional input and corresponds to the segmentation map
        output by `~sep.extract`.

    seg_id : array_like, optional
        Array of segmentation ids used to mask additional pixels in the image.
        Dimensions correspond to the dimensions of ``x`` and ``y``. The
        behavior differs depending on whether ``seg_id`` is negative or
        positive. If ``seg_id`` is positive, all pixels belonging to other
        objects are masked. (Pixel ``j, i`` is masked if ``seg[j, i] != seg_id
        and seg[j, i] != 0``). If ``seg_id`` is negative, all pixels other
        than those belonging to the object of interest are masked. (Pixel ``j,
        i`` is masked if ``seg[j, i] != -seg_id``).  NB: must be included if
        ``segmap`` is provided.

    bkgann : tuple, optional
        Length 2 tuple giving the inner and outer radius of a
        "background annulus". If supplied, the background is estimated
        by averaging unmasked pixels in this annulus. If ``clip_iters=0``,
        this reduces to the unclipped annulus mean (legacy behavior). If supplied, the inner
        and outer radii obey numpy broadcasting rules, along with ``x``,
        ``y``, and ellipse parameters.

    gain : float, optional
        Conversion factor between data array units and poisson counts,
        used in calculating poisson noise in aperture sum. If ``None``
        (default), do not add poisson noise.

    subpix : int, optional
        Subpixel sampling factor. Default is 5.

    clip_sigma : float, optional
        Sigma value for clipping when ``bkgann`` is provided. Default is 3.0.

    clip_iters : int, optional
        Maximum number of clipping iterations when ``bkgann`` is provided.
        Default is 5. Set to 0 to disable clipping.

    Returns
    -------
    sum : `~numpy.ndarray`
        The sum of the data array within the aperture.

    sumerr : `~numpy.ndarray`
        Error on the sum.

    flags : `~numpy.ndarray`
        Integer giving flags. (0 if no flags set.)

    """

    cdef double flux1, fluxerr1, x1, y1, r1, area1, rin1, rout1
    cdef double bkgflux, bkgfluxerr, bkgarea
    cdef double mean, std, med, mad_std, mean_clip
    cdef short flag1, bkgflag
    cdef size_t i
    cdef int status
    cdef np.broadcast it
    cdef sep_image im

    # Test for seg without seg_id.  Nothing happens if seg_id supplied but
    # without segmap.
    if (segmap is not None) and (seg_id is None):
        raise ValueError('`segmap` supplied but not `seg_id`.')

    _parse_arrays(data, err, var, mask, segmap, &im)
    im.maskthresh = maskthresh
    if gain is not None:
        im.gain = gain

    # Require that inputs are float64 arrays. See note in circular aperture.
    dt = np.dtype(np.double)
    dint = np.dtype(np.int32)
    x = np.require(x, dtype=dt)
    y = np.require(y, dtype=dt)
    a = np.require(a, dtype=dt)
    b = np.require(b, dtype=dt)
    theta = np.require(theta, dtype=dt)
    r = np.require(r, dtype=dt)

    # Segmentation image and ids with same dimensions as x, y, etc.
    if seg_id is not None:
        seg_id = np.require(seg_id, dtype=dint)
        if seg_id.shape != x.shape:
            raise ValueError('Shapes of `x` and `seg_id` do not match')
    else:
        seg_id = np.zeros(len(x), dtype=dint)

    if bkgann is None:

        # allocate ouput arrays
        shape = np.broadcast(x, y, a, b, theta, r).shape
        sum = np.empty(shape, dt)
        sumerr = np.empty(shape, dt)
        flag = np.empty(shape, np.short)

        it = np.broadcast(x, y, a, b, theta, r, seg_id, sum, sumerr, flag)
        while np.PyArray_MultiIter_NOTDONE(it):
            status = sep_sum_ellipse(
                &im,
                (<double*>np.PyArray_MultiIter_DATA(it, 0))[0],
                (<double*>np.PyArray_MultiIter_DATA(it, 1))[0],
                (<double*>np.PyArray_MultiIter_DATA(it, 2))[0],
                (<double*>np.PyArray_MultiIter_DATA(it, 3))[0],
                (<double*>np.PyArray_MultiIter_DATA(it, 4))[0],
                (<double*>np.PyArray_MultiIter_DATA(it, 5))[0],
                (<int*>np.PyArray_MultiIter_DATA(it, 6))[0],
                subpix, 0,
                <double*>np.PyArray_MultiIter_DATA(it, 7),
                <double*>np.PyArray_MultiIter_DATA(it, 8),
                &area1,
                <short*>np.PyArray_MultiIter_DATA(it, 9))
            _assert_ok(status)

            np.PyArray_MultiIter_NEXT(it)

        return sum, sumerr, flag

    else:
        rin, rout = bkgann

        # Require float arrays (see note above)
        rin = np.require(rin, dtype=dt)
        rout = np.require(rout, dtype=dt)

        # allocate ouput arrays
        shape = np.broadcast(x, y, a, b, theta, r, rin, rout).shape
        sum = np.empty(shape, dt)
        sumerr = np.empty(shape, dt)
        flag = np.empty(shape, np.short)

        it = np.broadcast(x, y, a, b, theta, r, rin, rout, seg_id, sum, sumerr, flag)
        while np.PyArray_MultiIter_NOTDONE(it):
            status = sep_sum_ellipse(
                &im,
                (<double*>np.PyArray_MultiIter_DATA(it, 0))[0],
                (<double*>np.PyArray_MultiIter_DATA(it, 1))[0],
                (<double*>np.PyArray_MultiIter_DATA(it, 2))[0],
                (<double*>np.PyArray_MultiIter_DATA(it, 3))[0],
                (<double*>np.PyArray_MultiIter_DATA(it, 4))[0],
                (<double*>np.PyArray_MultiIter_DATA(it, 5))[0],
                (<int*>np.PyArray_MultiIter_DATA(it, 8))[0],
                subpix, 0, &flux1, &fluxerr1, &area1, &flag1)
            _assert_ok(status)

            if clip_iters == 0:
                status = sep_sum_ellipann(
                    &im,
                    (<double*>np.PyArray_MultiIter_DATA(it, 0))[0],
                    (<double*>np.PyArray_MultiIter_DATA(it, 1))[0],
                    (<double*>np.PyArray_MultiIter_DATA(it, 2))[0],
                    (<double*>np.PyArray_MultiIter_DATA(it, 3))[0],
                    (<double*>np.PyArray_MultiIter_DATA(it, 4))[0],
                    (<double*>np.PyArray_MultiIter_DATA(it, 6))[0],
                    (<double*>np.PyArray_MultiIter_DATA(it, 7))[0],
                    (<int*>np.PyArray_MultiIter_DATA(it, 8))[0],
                    subpix, 0, &bkgflux, &bkgfluxerr, &bkgarea, &bkgflag)
                _assert_ok(status)

                if not bkgarea > 0:
                    raise ValueError(
                        "The background annulus does not contain any valid pixels, "
                        "for the object at index "
                        f"{np.PyArray_MultiIter_INDEX(it)}."
                    )
                mean_clip = bkgflux / bkgarea
            else:
                status = sep_stats_ellipann(
                    &im,
                    (<double*>np.PyArray_MultiIter_DATA(it, 0))[0],
                    (<double*>np.PyArray_MultiIter_DATA(it, 1))[0],
                    (<double*>np.PyArray_MultiIter_DATA(it, 2))[0],
                    (<double*>np.PyArray_MultiIter_DATA(it, 3))[0],
                    (<double*>np.PyArray_MultiIter_DATA(it, 4))[0],
                    (<double*>np.PyArray_MultiIter_DATA(it, 6))[0],
                    (<double*>np.PyArray_MultiIter_DATA(it, 7))[0],
                    (<int*>np.PyArray_MultiIter_DATA(it, 8))[0],
                    subpix, SEP_MASK_IGNORE,
                    clip_sigma, clip_iters,
                    &mean, &std, &med, &mad_std, &mean_clip,
                    &bkgarea, &bkgfluxerr, &bkgflag)
                _assert_ok(status)

                if not bkgarea > 0 or mean_clip != mean_clip:
                    raise ValueError(
                        "The background annulus does not contain any valid pixels, "
                        "for the object at index "
                        f"{np.PyArray_MultiIter_INDEX(it)}."
                    )

            if (area1 > 0):
              flux1 -= mean_clip * area1
              bkgfluxerr = bkgfluxerr / bkgarea * area1
              fluxerr1 = sqrt(fluxerr1*fluxerr1 + bkgfluxerr*bkgfluxerr)

            (<double*>np.PyArray_MultiIter_DATA(it, 9))[0] = flux1
            (<double*>np.PyArray_MultiIter_DATA(it, 10))[0] = fluxerr1
            (<short*>np.PyArray_MultiIter_DATA(it, 11))[0] = flag1

            #PyArray_MultiIter_NEXT is used to advance the iterator
            np.PyArray_MultiIter_NEXT(it)

        return sum, sumerr, flag


@cython.boundscheck(False)
@cython.wraparound(False)
def sum_ellipann(np.ndarray data not None, x, y, a, b, theta, rin, rout,
                 var=None, err=None, gain=None, np.ndarray mask=None,
                 double maskthresh=0.0,
                 seg_id=None, np.ndarray segmap=None,
                 int subpix=5):
    """sum_ellipann(data, x, y, a, b, theta, rin, rout, err=None, var=None,
                    mask=None, maskthresh=0.0, gain=None, subpix=5)

    Sum data in elliptical annular aperture(s).

    Parameters
    ----------
    data : `~numpy.ndarray`
        2-d array to be summed.

    x, y : array_like
        Center coordinates and radius (radii) of aperture(s).
        ``x`` corresponds to the second ("fast") axis of the input array
        and ``y`` corresponds to the first ("slow") axis.
        ``x, y = (0.0, 0.0)`` corresponds to the center of the first
        element of the array. These inputs obey numpy broadcasting rules.

    a, b, theta, rin, rout : array_like
        Elliptical annulus parameters. These inputs, along with ``x`` and ``y``,
        obey numpy broadcasting rules. ``a`` is the semi-major axis,
        ``b`` is the semi-minor axis and ``theta`` is angle in radians between
        the positive x axis and the major axis. It must be in the range
        ``[-pi/2, pi/2]``. It is also required that ``a >= b >= 0.0`` and
        ``rout >= rin >= 0.0``

    err, var : float or `~numpy.ndarray`
        Error *or* variance (specify at most one).

    mask : `~numpy.ndarray`, optional
        Mask array. If supplied, a given pixel is masked if its value
        is greater than ``maskthresh``.

    maskthresh : float, optional
        Threshold for a pixel to be masked. Default is ``0.0``.

    gain : float, optional
        Conversion factor between data array units and poisson counts,
        used in calculating poisson noise in aperture sum. If ``None``
        (default), do not add poisson noise.

    segmap : `~numpy.ndarray`, optional
        Segmentation image with dimensions of ``data`` and dtype ``np.int32``.
        This is an optional input and corresponds to the segmentation map
        output by `~sep.extract`.

    seg_id : array_like, optional
        Array of segmentation ids used to mask additional pixels in the image.
        Dimensions correspond to the dimensions of ``x`` and ``y``. The
        behavior differs depending on whether ``seg_id`` is negative or
        positive. If ``seg_id`` is positive, all pixels belonging to other
        objects are masked. (Pixel ``j, i`` is masked if ``seg[j, i] != seg_id
        and seg[j, i] != 0``). If ``seg_id`` is negative, all pixels other
        than those belonging to the object of interest are masked. (Pixel ``j,
        i`` is masked if ``seg[j, i] != -seg_id``).  NB: must be included if
        ``segmap`` is provided.

    subpix : int, optional
        Subpixel sampling factor. Default is 5.

    Returns
    -------
    sum : `~numpy.ndarray`
        The sum of the data array within the aperture(s).

    sumerr : `~numpy.ndarray`
        Error on the sum.

    flags : `~numpy.ndarray`
        Integer giving flags. (0 if no flags set.)
    """

    cdef double flux1, fluxerr1, x1, y1, r1, area1, rin1, rout1
    cdef double bkgflux, bkgfluxerr, bkgarea
    cdef short flag1, bkgflag
    cdef size_t i
    cdef int status
    cdef np.broadcast it
    cdef sep_image im

    # Test for segmap without seg_id.  Nothing happens if seg_id supplied but
    # without segmap.
    if (segmap is not None) and (seg_id is None):
        raise ValueError('`segmap` supplied but not `seg_id`.')

    _parse_arrays(data, err, var, mask, segmap, &im)
    im.maskthresh = maskthresh
    if gain is not None:
        im.gain = gain

    # Require that inputs are float64 arrays. See note in circular aperture.
    dt = np.dtype(np.double)
    dint = np.dtype(np.int32)

    x = np.require(x, dtype=dt)
    y = np.require(y, dtype=dt)
    a = np.require(a, dtype=dt)
    b = np.require(b, dtype=dt)
    theta = np.require(theta, dtype=dt)
    rin = np.require(rin, dtype=dt)
    rout = np.require(rout, dtype=dt)

    # Segmentation image and ids with same dimensions as x, y, etc.
    if seg_id is not None:
        seg_id = np.require(seg_id, dtype=dint)
        if seg_id.shape != x.shape:
            raise ValueError('Shapes of `x` and `seg_id` do not match')
    else:
        seg_id = np.zeros(len(x), dtype=dint)

    # allocate ouput arrays
    shape = np.broadcast(x, y, a, b, theta, rin, rout).shape
    sum = np.empty(shape, dt)
    sumerr = np.empty(shape, dt)
    flag = np.empty(shape, np.short)

    it = np.broadcast(x, y, a, b, theta, rin, rout, seg_id, sum, sumerr, flag)
    while np.PyArray_MultiIter_NOTDONE(it):
        status = sep_sum_ellipann(
            &im,
            (<double*>np.PyArray_MultiIter_DATA(it, 0))[0],
            (<double*>np.PyArray_MultiIter_DATA(it, 1))[0],
            (<double*>np.PyArray_MultiIter_DATA(it, 2))[0],
            (<double*>np.PyArray_MultiIter_DATA(it, 3))[0],
            (<double*>np.PyArray_MultiIter_DATA(it, 4))[0],
            (<double*>np.PyArray_MultiIter_DATA(it, 5))[0],
            (<double*>np.PyArray_MultiIter_DATA(it, 6))[0],
            (<int*>np.PyArray_MultiIter_DATA(it, 7))[0],
            subpix, 0,
            <double*>np.PyArray_MultiIter_DATA(it, 8),
            <double*>np.PyArray_MultiIter_DATA(it, 9),
            &area1,
            <short*>np.PyArray_MultiIter_DATA(it, 10))
        _assert_ok(status)
        np.PyArray_MultiIter_NEXT(it)

    return sum, sumerr, flag

@cython.boundscheck(False)
@cython.wraparound(False)
def flux_radius(np.ndarray data not None, x, y, rmax, frac, normflux=None,
                np.ndarray mask=None, double maskthresh=0.0,
                seg_id=None, np.ndarray segmap=None,
                int subpix=5):
    """flux_radius(data, x, y, rmax, frac, normflux=None, mask=None,
                   maskthresh=0.0, subpix=5)

    Return radius of a circle enclosing requested fraction of total flux.

    Parameters
    ----------
    data : `~numpy.ndarray`
        2-d array to be summed.

    x, y : array_like
        Center coordinates and radius (radii) of aperture(s).
        ``x`` corresponds to the second ("fast") axis of the input array
        and ``y`` corresponds to the first ("slow") axis.
        ``x, y = (0.0, 0.0)`` corresponds to the center of the first
        element of the array. Shapes must match.

    rmax : array_like
        Maximum radius to analyze. Used as normalizing flux if ``normflux``
        is None. Shape must match x and y.

    frac : array_like
        Requested fraction of light (in range 0 to 1). Can be scalar or array.

    normflux : array_like, optional
        Normalizing flux for each position. If not given, the sum
        within ``rmax`` is used as the normalizing flux. If given,
        shape must match x, y and rmax.

    mask : `~numpy.ndarray`, optional
        Mask array. If supplied, a given pixel is masked if its value
        is greater than ``maskthresh``.

    maskthresh : float, optional
        Threshold for a pixel to be masked. Default is ``0.0``.

    segmap : `~numpy.ndarray`, optional
        Segmentation image with dimensions of ``data`` and dtype ``np.int32``.
        This is an optional input and corresponds to the segmentation map
        output by `~sep.extract`.

    seg_id : array_like, optional
        Array of segmentation ids used to mask additional pixels in the image.
        Dimensions correspond to the dimensions of ``x`` and ``y``. The
        behavior differs depending on whether ``seg_id`` is negative or
        positive. If ``seg_id`` is positive, all pixels belonging to other
        objects are masked. (Pixel ``j, i`` is masked if ``seg[j, i] != seg_id
        and seg[j, i] != 0``). If ``seg_id`` is negative, all pixels other
        than those belonging to the object of interest are masked. (Pixel ``j,
        i`` is masked if ``seg[j, i] != -seg_id``).  NB: must be included if
        ``segmap`` is provided.

    subpix : int, optional
        Subpixel sampling factor. Default is 5.

    Returns
    -------
    radius : `~numpy.ndarray`
        The sum of the data array within the aperture(s). Shape is
        same as ``x``, except if ``frac`` is an array; then the
        dimension of ``frac`` will be appended. For example, if ``x``
        and ``frac`` are both 1-d arrays, the result will be a 2-d
        array with the trailing dimension corresponding to ``frac``.

    flags : `~numpy.ndarray`
        Integer giving flags. Same shape as ``x``. (0 if no flags set.)

    """

    cdef double flux1, fluxerr1, x1, y1, r1, area1, rin1, rout1
    cdef double bkgflux, bkgfluxerr, bkgarea
    cdef short flag1, bkgflag
    cdef int i
    cdef int status, fracn
    cdef short[:] flag
    cdef double[:, :] radius
    cdef double[:] fractmp
    cdef double[:] xtmp
    cdef double[:] ytmp
    cdef double[:] rtmp
    cdef double[:] normfluxbuf
    cdef double *normfluxptr
    cdef sep_image im

    # Test for segmap without seg_id.  Nothing happens if seg_id supplied but
    # without segmap.
    if (segmap is not None) and (seg_id is None):
        raise ValueError('`segmap` supplied but not `seg_id`.')

    _parse_arrays(data, None, None, mask, segmap, &im)
    im.maskthresh = maskthresh

    # Require that inputs are float64 arrays with same shape. See note in
    # circular aperture.
    # Also require that frac is a contiguous array.
    dt = np.dtype(np.double)
    dint = np.dtype(np.int32)

    x = np.require(x, dtype=dt)
    y = np.require(y, dtype=dt)
    rmax = np.require(rmax, dtype=dt)
    frac = np.require(frac, dtype=dt)
    inshape = x.shape
    infracshape = frac.shape
    if (y.shape != inshape or rmax.shape != inshape):
        raise ValueError("shape of x, y, and r must match")

    if seg_id is not None:
        seg_id = np.require(seg_id, dtype=dint)
        if seg_id.shape != x.shape:
            raise ValueError('Shapes of `x` and `seg_id` do not match')
    else:
        seg_id = np.zeros(len(x), dtype=dint)

    # Convert input arrays to 1-d for correct looping and indexing.
    xtmp = np.ravel(x)
    ytmp = np.ravel(y)
    rtmp = np.ravel(rmax)
    itmp = np.ravel(seg_id)
    fractmp = np.ravel(np.ascontiguousarray(frac))
    fracn = len(fractmp)

    # optional `normflux` input.
    normfluxptr = NULL
    if normflux is not None:
        normflux = np.require(normflux, dtype=dt)
        if normflux.shape != inshape:
            raise ValueError("shape of normflux must match shape of "
                             "x, y and r")
        normfluxbuf = np.ravel(normflux)
        normfluxptr = &normfluxbuf[0]

    # Allocate ouput arrays. (We'll reshape these later to match the
    # input shapes.)
    flag = np.empty(len(xtmp), np.short)
    radius = np.empty((len(xtmp), len(fractmp)), dt)

    for i in range(len(xtmp)):
        if normfluxptr != NULL:
            normfluxptr = &normfluxbuf[i]
        status = sep_flux_radius(&im,
                                 xtmp[i], ytmp[i], rtmp[i], itmp[i],
                                 subpix, 0,
                                 normfluxptr, &fractmp[0], fracn,
                                 &radius[i, 0], &flag[i])
        _assert_ok(status)

    return (np.asarray(radius).reshape(inshape + infracshape),
            np.asarray(flag).reshape(inshape))


@cython.boundscheck(False)
@cython.wraparound(False)
def mask_ellipse(np.ndarray arr not None, x, y, a=None, b=None, theta=None,
                 r=1.0, cxx=None, cyy=None, cxy=None):
    """mask_ellipse(arr, x, y, a, b, theta, r=1.0)

    Mask ellipse(s) in an array.

    Set array elements to True (or 1) if they fall within the given
    ellipse.  The ``r`` keyword can be used to scale the ellipse.
    Equivalently, after converting ``a``, ``b``, ``theta`` to a
    coefficient ellipse representation (``cxx``, ``cyy``, ``cxy``),
    pixels that fulfill the condition

    .. math::

       cxx(x_i - x)^2 + cyy(y_i - y)^2 + cxx(x_i - x)(y_i - y) < r^2

    will be masked.

    Parameters
    ----------
    arr : `~numpy.ndarray`
        Input array to be masked. Array is updated in-place.
    x, y : array_like
        Center of ellipse(s).
    a, b, theta : array_like, optional
        Parameters defining the extent of the ellipe(s).
    cxx, cyy, cxy : array_like, optional
        Alternative ellipse representation. Can be used as
        ``mask_ellipse(arr, x, y, cxx=..., cyy=..., cxy=...)``.
    r : array_like, optional
        Scale factor of ellipse(s). Default is 1.
    """

    cdef np.int64_t w, h
    cdef np.uint8_t[:,:] buf
    cdef double cxx_, cyy_, cxy_

    dt = np.dtype(np.double)

    # only boolean arrays supported
    if not (arr.dtype.type is np.bool_ or arr.dtype.type is np.ubyte):
        raise ValueError("Array data type not supported: {0:s}"
                         .format(arr.dtype))
    _check_array_get_dims(arr, &w, &h)
    buf = arr.view(dtype=np.uint8)

    x = np.require(x, dtype=dt)
    y = np.require(y, dtype=dt)
    r = np.require(r, dtype=dt)

    # a, b, theta representation
    if (a is not None and b is not None and theta is not None):
        a = np.require(a, dtype=dt)
        b = np.require(b, dtype=dt)
        theta = np.require(theta, dtype=dt)

        it = np.broadcast(x, y, a, b, theta, r)
        while np.PyArray_MultiIter_NOTDONE(it):
            sep_ellipse_coeffs((<double*>np.PyArray_MultiIter_DATA(it, 2))[0],
                               (<double*>np.PyArray_MultiIter_DATA(it, 3))[0],
                               (<double*>np.PyArray_MultiIter_DATA(it, 4))[0],
                               &cxx_, &cyy_, &cxy_)
            sep_set_ellipse(<unsigned char *>&buf[0, 0], w, h,
                            (<double*>np.PyArray_MultiIter_DATA(it, 0))[0],
                            (<double*>np.PyArray_MultiIter_DATA(it, 1))[0],
                            cxx_, cyy_, cxy_,
                            (<double*>np.PyArray_MultiIter_DATA(it, 5))[0],
                            1)
            np.PyArray_MultiIter_NEXT(it)

    # cxx, cyy, cxy representation
    elif (cxx is not None and cyy is not None and cxy is not None):
        cxx = np.require(cxx, dtype=dt)
        cyy = np.require(cyy, dtype=dt)
        cxy = np.require(cxy, dtype=dt)

        it = np.broadcast(x, y, cxx, cyy, cxy, r)
        while np.PyArray_MultiIter_NOTDONE(it):
            sep_set_ellipse(<unsigned char *>&buf[0, 0], w, h,
                            (<double*>np.PyArray_MultiIter_DATA(it, 0))[0],
                            (<double*>np.PyArray_MultiIter_DATA(it, 1))[0],
                            (<double*>np.PyArray_MultiIter_DATA(it, 2))[0],
                            (<double*>np.PyArray_MultiIter_DATA(it, 3))[0],
                            (<double*>np.PyArray_MultiIter_DATA(it, 4))[0],
                            (<double*>np.PyArray_MultiIter_DATA(it, 5))[0],
                            1)
            np.PyArray_MultiIter_NEXT(it)
    else:
        raise ValueError("Must specify either a, b and theta or "
                         "cxx, cyy and cxy.")


def kron_radius(np.ndarray data not None, x, y, a, b, theta, r,
                np.ndarray mask=None, double maskthresh=0.0,
                seg_id=None, np.ndarray segmap=None):
    """kron_radius(data, x, y, a, b, theta, r, mask=None, maskthresh=0.0, seg_id=None, segmap=None)

    Calculate Kron "radius" within an ellipse.

    The Kron radius is given by

    .. math::

       \sum_i r_i I(r_i) / \sum_i I(r_i)

    where the sum is over all pixels in the aperture and the radius is given
    in units of ``a`` and ``b``: ``r_i`` is the distance to the pixel relative
    to the distance to the ellipse specified by ``a``, ``b``, ``theta``.
    Equivalently, after converting the ellipse parameters to their coefficient
    representation, ``r_i`` is given by

    .. math::

       r_i^2 = cxx(x_i - x)^2 + cyy(y_i - y)^2 + cxx(x_i - x)(y_i - y)

    Parameters
    ----------
    data : `~numpy.ndarray`
        Data array.

    x, y : array_like
        Ellipse center(s).

    a, b, theta : array_like
        Ellipse parameters.

    r : array_like
        "Radius" of ellipse over which to integrate. If the ellipse
        extent correponds to second moments of an object, this is the
        number of "isophotal radii" in Source Extractor parlance. A
        Fixed value of 6 is used in Source Extractor.

    mask : `numpy.ndarray`, optional
        An optional mask.

    maskthresh : float, optional
        Pixels with mask > maskthresh will be ignored.

    segmap : `~numpy.ndarray`, optional
        Segmentation image with dimensions of ``data`` and dtype ``np.int32``.
        This is an optional input and corresponds to the segmentation map
        output by `~sep.extract`.

    seg_id : array_like, optional
        Array of segmentation ids used to mask additional pixels in the image.
        Dimensions correspond to the dimensions of ``x`` and ``y``. The
        behavior differs depending on whether ``seg_id`` is negative or
        positive. If ``seg_id`` is positive, all pixels belonging to other
        objects are masked. (Pixel ``j, i`` is masked if ``seg[j, i] != seg_id
        and seg[j, i] != 0``). If ``seg_id`` is negative, all pixels other
        than those belonging to the object of interest are masked. (Pixel ``j,
        i`` is masked if ``seg[j, i] != -seg_id``).  NB: must be included if
        ``segmap`` is provided.

    Returns
    -------
    kronrad : array_like
        The Kron radius.

    flag : array_like
        Integer value indicating conditions about the aperture or how
        many masked pixels it contains.

    """

    cdef double cxx, cyy, cxy
    cdef sep_image im

    # Test for segmap without seg_id.  Nothing happens if seg_id supplied but
    # without segmap.
    if (segmap is not None) and (seg_id is None):
        raise ValueError('`segmap` supplied but not `seg_id`.')

    _parse_arrays(data, None, None, mask, segmap, &im)
    im.maskthresh = maskthresh

    # See note in apercirc on requiring specific array type
    dt = np.dtype(np.double)
    dint = np.dtype(np.int32)

    x = np.require(x, dtype=dt)
    y = np.require(y, dtype=dt)
    a = np.require(a, dtype=dt)
    b = np.require(b, dtype=dt)
    theta = np.require(theta, dtype=dt)
    r = np.require(r, dtype=dt)

    # Segmentation image and ids with same dimensions as x, y, etc.
    if seg_id is not None:
        seg_id = np.require(seg_id, dtype=dint)
        if seg_id.shape != x.shape:
            raise ValueError('Shapes of `x` and `seg_id` do not match')
    else:
        seg_id = np.zeros(len(x), dtype=dint)

    # allocate output arrays
    shape = np.broadcast(x, y, a, b, theta, r).shape
    kr = np.empty(shape, np.float64)
    flag = np.empty(shape, np.short)

    it = np.broadcast(x, y, a, b, theta, r, seg_id, kr, flag)
    while np.PyArray_MultiIter_NOTDONE(it):
        sep_ellipse_coeffs((<double*>np.PyArray_MultiIter_DATA(it, 2))[0],
                           (<double*>np.PyArray_MultiIter_DATA(it, 3))[0],
                           (<double*>np.PyArray_MultiIter_DATA(it, 4))[0],
                           &cxx, &cyy, &cxy)
        status = sep_kron_radius(&im,
                                 (<double*>np.PyArray_MultiIter_DATA(it, 0))[0],
                                 (<double*>np.PyArray_MultiIter_DATA(it, 1))[0],
                                 cxx, cyy, cxy,
                                 (<double*>np.PyArray_MultiIter_DATA(it, 5))[0],
                                 (<int*>np.PyArray_MultiIter_DATA(it, 6))[0],
                                 <double*>np.PyArray_MultiIter_DATA(it, 7),
                                 <short*>np.PyArray_MultiIter_DATA(it, 8))
        _assert_ok(status)
        np.PyArray_MultiIter_NEXT(it)

    return kr, flag

def winpos(np.ndarray data not None, xinit, yinit, sig=None,
           np.ndarray mask=None, double maskthresh=0.0, int subpix=11,
           double minsig=2.0/2.35*0.5, seg_id=None, np.ndarray segmap=None,
           maxstep=None, maxshift=None, ignore_masked=None, PSF psf=None):
    """winpos(data, xinit, yinit, sig=None, mask=None, maskthresh=0.0,
              subpix=11, minsig=2.0/2.35*0.5, seg_id=None, segmap=None,
              maxstep=None, maxshift=None, ignore_masked=None, psf=None)

    Calculate more accurate object centroids using 'windowed' algorithm.

    Starting from the supplied initial center position, an iterative
    algorithm is used to determine a better object centroid. On each
    iteration, the centroid is calculated from pixels around the current
    position with either Gaussian weighting or, if ``psf`` is supplied,
    the resampled PSF model itself. Iteration stops when the change in
    position falls under some threshold or a maximum number of iterations
    is reached. The Gaussian mode is equivalent to ``XWIN_IMAGE`` and
    ``YWIN_IMAGE`` parameters in Source Extractor (for the correct choice
    of sigma for each object).

    **Note:** One should be cautious about using windowed positions in
    crowded fields or for sources with nearby neighbors, as the iterative
    algorithm can fail catastrophically.

    Parameters
    ----------
    data : `~numpy.ndarray`
        Data array.

    xinit, yinit : array_like
        Initial center(s).

    sig : array_like, optional
        Gaussian sigma used for weighting pixels. Pixels within a circular
        aperture of radius 4*sig are included. Required unless ``psf`` is
        supplied. Ignored when ``psf`` is supplied.

    mask : `numpy.ndarray`, optional
        An optional mask.

    maskthresh : float, optional
        Pixels with mask > maskthresh will be ignored.

    subpix : int, optional
        Subpixel sampling used to determine pixel overlap with
        aperture.  11 is used in Source Extractor. For exact overlap
        calculation, use 0. Ignored when ``psf`` is supplied.

    minsig : float, optional
        Minimum bound on ``sig`` parameter. ``sig`` values smaller than this
        are increased to ``minsig`` to replicate Source Extractor behavior.
        Source Extractor uses a minimum half-light radius of 0.5 pixels,
        equivalent to a sigma of 0.5 * 2.0 / 2.35. Ignored when ``psf`` is
        supplied.

    segmap : `~numpy.ndarray`, optional
        Segmentation image with dimensions of ``data`` and dtype ``np.int32``.
        This is an optional input and corresponds to the segmentation map
        output by `~sep.extract`.

    seg_id : array_like, optional
        Array of segmentation ids used to mask additional pixels in the image.
        Dimensions correspond to the dimensions of ``xinit`` and ``yinit``.
        The behavior differs depending on whether ``seg_id`` is negative or
        positive. If ``seg_id`` is positive, all pixels belonging to other
        objects are masked. (Pixel ``j, i`` is masked if ``seg[j, i] != seg_id
        and seg[j, i] != 0``). If ``seg_id`` is negative, all pixels other
        than those belonging to the object of interest are masked. (Pixel ``j,
        i`` is masked if ``seg[j, i] != -seg_id``).  NB: must be included if
        ``segmap`` is provided.

    maxstep : float or array_like, optional
        Maximum step size per iteration in pixels. If ``None`` or <= 0,
        no step limiting is applied.

    maxshift : float or array_like, optional
        Maximum total shift from the initial position in pixels. If ``None``
        or <= 0, no total shift limiting is applied. If the limit is reached,
        the output position is projected to the limit and ``APER_TRUNC`` is
        set.

    ignore_masked : bool, optional
        If ``True``, masked pixels and segmentation-masked pixels are ignored
        when updating the centroid. If ``False``, they are replaced by the
        mean unmasked pixel value, matching the historical aperture masking
        correction. The default is ``True`` when ``segmap`` is supplied and
        ``False`` otherwise.

    psf : `PSF`, optional
        If supplied, use the evaluated and resampled PSF model as the
        centroid weighting function instead of a Gaussian window. This is
        useful when centroiding should follow the same PSF model used for
        optimal extraction or PSF fitting.

    Returns
    -------
    x, y : np.ndarray
        New x and y position(s).

    flag : np.ndarray
        Flags.

    """

    cdef int status
    cdef short inflag
    cdef double sigval
    cdef double maxstepval
    cdef double maxshiftval
    cdef int niter = 0  # not currently returned
    cdef sep_image im
    cdef object shape
    cdef np.ndarray xarr, yarr, maxsteparr, maxshiftarr, segidarr
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] xbuf, ybuf, maxstepbuf
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] maxshiftbuf
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] xoutbuf, youtbuf
    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] segidbuf, niterarr
    cdef np.ndarray[np.int16_t, ndim=1, mode='c'] flagbuf

    # Test for segmap without seg_id.  Nothing happens if seg_id supplied but
    # without segmap.
    if (segmap is not None) and (seg_id is None):
        raise ValueError('`segmap` supplied but not `seg_id`.')

    _parse_arrays(data, None, None, mask, segmap, &im)
    im.maskthresh = maskthresh

    # See note in apercirc on requiring specific array type
    dt = np.dtype(np.double)
    xinit = np.require(xinit, dtype=dt)
    yinit = np.require(yinit, dtype=dt)
    if psf is None:
        if sig is None:
            raise ValueError('`sig` is required unless `psf` is supplied.')
        sig = np.require(sig, dtype=dt)
    if maxstep is None:
        maxstep = 0.0
    maxstep = np.require(maxstep, dtype=dt)
    if maxshift is None:
        maxshift = 0.0
    maxshift = np.require(maxshift, dtype=dt)
    if ignore_masked is None:
        ignore_masked = segmap is not None
    inflag = SEP_MASK_IGNORE if ignore_masked else 0

    if psf is None:
        shape = np.broadcast(xinit, yinit, sig, maxstep, maxshift).shape
    else:
        shape = np.broadcast(xinit, yinit, maxstep, maxshift).shape

    # Segmentation image and ids with same dimensions as xinit, yinit, etc.
    if seg_id is not None:
        seg_id = np.require(seg_id, dtype=np.int32)
        if seg_id.shape != shape:
            raise ValueError('Shapes of `xinit` and `seg_id` do not match')
    else:
        seg_id = np.zeros(shape, dtype=np.int32)

    # allocate output arrays
    x = np.empty(shape, np.float64)
    y = np.empty(shape, np.float64)
    flag = np.empty(shape, np.short)

    if psf is None:
        it = np.broadcast(
            xinit, yinit, sig, maxstep, maxshift, seg_id, x, y, flag)
        while np.PyArray_MultiIter_NOTDONE(it):
            sigval = (<double*>np.PyArray_MultiIter_DATA(it, 2))[0]
            maxstepval = (<double*>np.PyArray_MultiIter_DATA(it, 3))[0]
            maxshiftval = (<double*>np.PyArray_MultiIter_DATA(it, 4))[0]
            if sigval < minsig:
                sigval = minsig
            status = sep_windowed(&im,
                                  (<double*>np.PyArray_MultiIter_DATA(it, 0))[0],
                                  (<double*>np.PyArray_MultiIter_DATA(it, 1))[0],
                                  sigval,
                                  subpix, inflag,
                                  (<int*>np.PyArray_MultiIter_DATA(it, 5))[0],
                                  maxstepval,
                                  maxshiftval,
                                  <double*>np.PyArray_MultiIter_DATA(it, 6),
                                  <double*>np.PyArray_MultiIter_DATA(it, 7),
                                  &niter,
                                  <short*>np.PyArray_MultiIter_DATA(it, 8))
            _assert_ok(status)
            np.PyArray_MultiIter_NEXT(it)
    else:
        xarr, yarr, maxsteparr, maxshiftarr = np.broadcast_arrays(
            xinit, yinit, maxstep, maxshift)
        xbuf = np.ascontiguousarray(xarr, dtype=dt).reshape(-1)
        ybuf = np.ascontiguousarray(yarr, dtype=dt).reshape(-1)
        maxstepbuf = np.ascontiguousarray(maxsteparr, dtype=dt).reshape(-1)
        maxshiftbuf = np.ascontiguousarray(maxshiftarr, dtype=dt).reshape(-1)
        segidbuf = np.ascontiguousarray(seg_id, dtype=np.intc).reshape(-1)
        xoutbuf = np.empty(xbuf.size, dtype=np.float64)
        youtbuf = np.empty(ybuf.size, dtype=np.float64)
        flagbuf = np.empty(xbuf.size, dtype=np.int16)
        niterarr = np.empty(xbuf.size, dtype=np.intc)

        status = sep_windowed_psf_array(&im,
                                        psf.ptr,
                                        <double*>xbuf.data,
                                        <double*>ybuf.data,
                                        xbuf.size,
                                        <int*>segidbuf.data,
                                        inflag,
                                        <double*>maxstepbuf.data,
                                        <double*>maxshiftbuf.data,
                                        <double*>xoutbuf.data,
                                        <double*>youtbuf.data,
                                        <int*>niterarr.data,
                                        <short*>flagbuf.data)
        _assert_ok(status)
        x = xoutbuf.reshape(shape)
        y = youtbuf.reshape(shape)
        flag = flagbuf.reshape(shape)

    return x, y, flag



def ellipse_coeffs(a, b, theta):
    """ellipse_coeffs(a, b, theta)

    Convert from ellipse axes and angle to coefficient representation.

    Parameters
    ----------
    a, b, theta : array_like
        Ellipse(s) semi-major, semi-minor axes and position angle
        respectively.  Position angle is radians counter clockwise
        from positive x axis to major axis, and lies in range
        ``[-pi/2, pi/2]``

    Returns
    -------
    cxx, cyy, cxy : `~numpy.ndarray`
        Describes the ellipse(s) ``cxx * x^2 + cyy * y^2 + cxy * xy = 1``
    """

    dt = np.dtype(np.double)
    a = np.require(a, dtype=dt)
    b = np.require(b, dtype=dt)
    theta = np.require(theta, dtype=dt)

    shape = np.broadcast(a, b, theta).shape
    cxx = np.empty(shape, dt)
    cyy = np.empty(shape, dt)
    cxy = np.empty(shape, dt)

    it = np.broadcast(a, b, theta, cxx, cyy, cxy)
    while np.PyArray_MultiIter_NOTDONE(it):
        sep_ellipse_coeffs((<double*>np.PyArray_MultiIter_DATA(it, 0))[0],
                           (<double*>np.PyArray_MultiIter_DATA(it, 1))[0],
                           (<double*>np.PyArray_MultiIter_DATA(it, 2))[0],
                           <double*>np.PyArray_MultiIter_DATA(it, 3),
                           <double*>np.PyArray_MultiIter_DATA(it, 4),
                           <double*>np.PyArray_MultiIter_DATA(it, 5))
        np.PyArray_MultiIter_NEXT(it)

    return cxx, cyy, cxy

def ellipse_axes(cxx, cyy, cxy):
    """ellipse_axes(cxx, cyy, cxy)

    Convert from coefficient ellipse representation to ellipse axes and angle.

    Parameters
    ----------
    cxx, cyy, cxy : array_like
        Describes the ellipse(s) ``cxx * x**2 + cyy * y**2 + cxy * x * y = 1``

    Returns
    -------
    a, b, theta : `~numpy.ndarray`
        Ellipse(s) semi-major, semi-minor axes and position angle
        respectively.  Position angle is radians counter clockwise
        from positive x axis to major axis, and lies in range
        ``(-pi/2, pi/2)``

    Raises
    ------
    ValueError
        If input parameters do not describe an ellipse.

    """

    cdef int status

    dt = np.dtype(np.double)
    cxx = np.require(cxx, dtype=dt)
    cyy = np.require(cyy, dtype=dt)
    cxy = np.require(cxy, dtype=dt)

    shape = np.broadcast(cxx, cyy, cxy).shape
    a = np.empty(shape, dt)
    b = np.empty(shape, dt)
    theta = np.empty(shape, dt)

    status = 0
    it = np.broadcast(cxx, cyy, cxy, a, b, theta)
    while np.PyArray_MultiIter_NOTDONE(it):
        status = sep_ellipse_axes(
            (<double*>np.PyArray_MultiIter_DATA(it, 0))[0],
            (<double*>np.PyArray_MultiIter_DATA(it, 1))[0],
            (<double*>np.PyArray_MultiIter_DATA(it, 2))[0],
            <double*>np.PyArray_MultiIter_DATA(it, 3),
            <double*>np.PyArray_MultiIter_DATA(it, 4),
            <double*>np.PyArray_MultiIter_DATA(it, 5))
        if status:
            break

        np.PyArray_MultiIter_NEXT(it)

    if status:
        raise ValueError(
            "parameters do not describe ellipse: "
            "cxx={0:f}, cyy={1:f}, cxy={2:f}".format(
                (<double*>np.PyArray_MultiIter_DATA(it, 0))[0],
                (<double*>np.PyArray_MultiIter_DATA(it, 1))[0],
                (<double*>np.PyArray_MultiIter_DATA(it, 2))[0]))

    return a, b, theta

# -----------------------------------------------------------------------------
# Utility functions

def set_extract_pixstack(size_t size):
    """set_extract_pixstack(size)

    Set the size in pixels of the internal pixel buffer used in extract().

    The current value can be retrieved with get_extract_pixstack. The
    initial default is 300000.
    """
    sep_set_extract_pixstack(size)

def get_extract_pixstack():
    """get_extract_pixstack()

    Get the size in pixels of the internal pixel buffer used in extract().
    """
    return sep_get_extract_pixstack()


def set_sub_object_limit(int limit):
    """set_sub_object_limit(limit)

    Set the limit on the number of sub-objects when deblending in extract().

    The current value can be retrieved with get_sub_object_limit. The
    initial default is 1024.
    """
    sep_set_sub_object_limit(limit)

def get_sub_object_limit():
    """get_sub_object_limit()

    Get the limit on the number of sub-objects when deblending in extract().
    """
    return sep_get_sub_object_limit()

# -----------------------------------------------------------------------------
# PSF photometry

cdef class PSF:
    """PSF(data, sampling=1.0, degree=0, x0=0.0, y0=0.0, sx=1.0, sy=1.0, fwhm=0.0)

    Represents a spatially varying PSF model as a polynomial expansion
    over supersampled component images (e.g., from PSFEx).

    Parameters
    ----------
    data : `~numpy.ndarray`
        Polynomial basis component images. Shape ``(ncomp, h, w)`` or
        ``(h, w)`` for a single constant component.
    sampling : float, optional
        PSF pixel size in image pixels. Values < 1 mean the PSF is
        supersampled (default 1.0).
    degree : int, optional
        Polynomial degree for spatial variation (default 0 = constant).
    x0, y0 : float, optional
        Context normalization offsets (image coordinates).
    sx, sy : float, optional
        Context normalization scales.
    fwhm : float, optional
        Typical PSF FWHM in image pixels.
    """

    cdef sep_psf *ptr

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self, np.ndarray data not None,
                  double sampling=1.0, int degree=0,
                  double x0=0.0, double y0=0.0,
                  double sx=1.0, double sy=1.0,
                  double fwhm=0.0):
        cdef int status
        cdef np.ndarray[float, ndim=3, mode='c'] darr

        if data.ndim == 2:
            data = data[np.newaxis, :, :]
        if data.ndim != 3:
            raise ValueError("data must be 2-d or 3-d array")

        darr = np.ascontiguousarray(data, dtype=np.float32)

        status = sep_psf_create(&self.ptr,
                                <float*>darr.data,
                                darr.shape[2], darr.shape[1], darr.shape[0],
                                degree, x0, y0, sx, sy,
                                <float>sampling, fwhm)
        _assert_ok(status)

    def __init__(self, np.ndarray data not None,
                 double sampling=1.0, int degree=0,
                 double x0=0.0, double y0=0.0,
                 double sx=1.0, double sy=1.0,
                 double fwhm=0.0):
        pass

    def __dealloc__(self):
        if self.ptr is not NULL:
            sep_psf_free(self.ptr)

    property width:
        """Supersampled PSF stamp width."""
        def __get__(self):
            return self.ptr.w

    property height:
        """Supersampled PSF stamp height."""
        def __get__(self):
            return self.ptr.h

    property ncomp:
        """Number of polynomial components."""
        def __get__(self):
            return self.ptr.ncomp

    property degree:
        """Polynomial degree for spatial variation."""
        def __get__(self):
            return self.ptr.degree

    property sampling:
        """PSF sampling step (image pixels per PSF pixel)."""
        def __get__(self):
            return self.ptr.pixstep

    property fwhm:
        """Typical PSF FWHM in image pixels."""
        def __get__(self):
            return self.ptr.fwhm

    property stamp_width:
        """Native-resolution stamp width."""
        def __get__(self):
            return self.ptr.rw

    property stamp_height:
        """Native-resolution stamp height."""
        def __get__(self):
            return self.ptr.rh

    property fit_radius:
        """Fitting radius in image pixels. If > 0, only pixels within this
        radius of the source center participate in PSF photometry. 0 means
        use the full stamp (default)."""
        def __get__(self):
            return self.ptr.fit_radius
        def __set__(self, double value):
            self.ptr.fit_radius = value

    @classmethod
    def from_gaussian(cls, double fwhm, int size=0, int oversampling=2):
        """Create a PSF model from a circular Gaussian profile.

        Parameters
        ----------
        fwhm : float
            Full width at half maximum in image pixels.
        size : int, optional
            Stamp size in native image pixels. If 0 (default), set to
            ``int(ceil(4 * fwhm)) | 1`` (nearest odd number).
        oversampling : int, optional
            Oversampling factor (default 2).

        Returns
        -------
        PSF
        """
        if size <= 0:
            size = int(np.ceil(4.0 * fwhm))
            if size % 2 == 0:
                size += 1

        ossize = size * oversampling
        sigma = fwhm / 2.354820045
        sigma_os = sigma * oversampling

        y, x = np.mgrid[0:ossize, 0:ossize]
        cx = ossize // 2
        cy = ossize // 2
        stamp = np.exp(-((x - cx)**2 + (y - cy)**2) / (2.0 * sigma_os**2))
        stamp = stamp / stamp.sum()
        data = stamp[np.newaxis, :, :].astype(np.float32)

        return cls(data, sampling=1.0 / oversampling, degree=0, fwhm=fwhm)

    @classmethod
    def from_psfex(cls, filename):
        """Load a PSF model from a PSFEx ``.psf`` file.

        Parameters
        ----------
        filename : str
            Path to PSFEx output file.

        Returns
        -------
        PSF
        """
        from astropy.io import fits

        hdu = fits.open(filename)
        header = hdu[1].header
        data = hdu[1].data[0][0]
        hdu.close()

        w = header.get('PSFAXIS1')
        h = header.get('PSFAXIS2')
        degree = header.get('POLDEG1', 0)
        x0 = header.get('POLZERO1', 0.0)
        sx = header.get('POLSCAL1', 1.0)
        y0 = header.get('POLZERO2', 0.0)
        sy = header.get('POLSCAL2', 1.0)
        sampling = header.get('PSF_SAMP', 1.0)
        fwhm = header.get('PSF_FWHM', 0.0)

        return cls(np.ascontiguousarray(data, dtype=np.float32),
                   sampling=sampling, degree=degree,
                   x0=x0, y0=y0, sx=sx, sy=sy, fwhm=fwhm)


@cython.boundscheck(False)
@cython.wraparound(False)
def model_psf(np.ndarray arr not None, x, y, flux, PSF psf not None):
    """model_psf(arr, x, y, flux, psf)

    Add PSF model image(s) to an array in-place.

    This evaluates and resamples the supplied PSF at each source position,
    normalizes the stamp to unit sum, scales by ``flux``, and adds the result
    into ``arr``.

    Parameters
    ----------
    arr : `~numpy.ndarray`
        Output image array to update in-place. Must be 2-d, C-contiguous,
        and have dtype ``float32`` or ``float64``.
    x, y : array_like
        Source center(s).
    flux : array_like
        Source flux(es).
    psf : `PSF`
        PSF model.
    """

    cdef int status
    cdef int dtype
    cdef np.int64_t w, h
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] xbuf, ybuf, fbuf
    cdef object shape
    cdef np.int64_t i

    _check_array_get_dims(arr, &w, &h)
    dtype = _get_sep_dtype(arr.dtype)
    if dtype != SEP_TFLOAT and dtype != SEP_TDOUBLE:
        raise ValueError("arr must have dtype float32 or float64")

    shape = np.broadcast(x, y, flux).shape
    xbuf = np.ascontiguousarray(
        np.broadcast_to(np.asarray(x, dtype=np.float64), shape).ravel()
    )
    ybuf = np.ascontiguousarray(
        np.broadcast_to(np.asarray(y, dtype=np.float64), shape).ravel()
    )
    fbuf = np.ascontiguousarray(
        np.broadcast_to(np.asarray(flux, dtype=np.float64), shape).ravel()
    )

    for i in range(xbuf.shape[0]):
        status = sep_set_psf(<void*>arr.data, dtype, w, h, psf.ptr,
                             xbuf[i], ybuf[i], fbuf[i])
        _assert_ok(status)


@cython.boundscheck(False)
@cython.wraparound(False)
def psf_snr(np.ndarray data not None, PSF psf not None,
            var=None, err=None, gain=None, np.ndarray mask=None,
            double maskthresh=0.0, bint local_bkg=False):
    """psf_snr(data, psf, var=None, err=None, gain=None, mask=None, maskthresh=0.0, local_bkg=False)

    Compute a PSF-matched significance image.

    At each image pixel, this evaluates the supplied PSF centered on that
    pixel and returns ``sum(P * data / var) / sqrt(sum(P**2 / var))``.
    Masked pixels and pixels with non-positive variance are ignored in each
    local sum.

    Parameters
    ----------
    data : `~numpy.ndarray`
        2-d image array.
    psf : `PSF`
        PSF model.
    var : float or `~numpy.ndarray`, optional
        Variance (scalar or 2-d array). Mutually exclusive with ``err``.
    err : float or `~numpy.ndarray`, optional
        Standard deviation (scalar or 2-d array). Mutually exclusive
        with ``var``.
    gain : float, optional
        Effective gain in electrons per data unit. If supplied, positive
        pixel values add Poisson variance as in PSF photometry.
    mask : `~numpy.ndarray`, optional
        Mask array.
    maskthresh : float, optional
        Mask threshold.
    local_bkg : bool, optional
        If True, fit and remove a constant local background term within each
        PSF footprint before computing the source significance.

    Returns
    -------
    snr : `~numpy.ndarray`
        PSF-matched significance image with dtype ``float64``.
    """

    cdef int status
    cdef sep_image im
    cdef np.ndarray[np.double_t, ndim=2, mode="c"] out

    _parse_arrays(data, err, var, mask, None, &im)
    im.maskthresh = maskthresh
    if gain is not None:
        im.gain = gain

    out = np.empty((data.shape[0], data.shape[1]), dtype=np.float64)
    status = sep_psf_snr(&im, psf.ptr, 1 if local_bkg else 0,
                         <double*>out.data)
    _assert_ok(status)
    return out


def _normalize_psf_snr(snr, mask, maskthresh, snr_bw, snr_bh, snr_fw, snr_fh,
                       snr_fthresh):
    snr_bkg = Background(snr, mask=mask, maskthresh=maskthresh,
                         bw=snr_bw, bh=snr_bh, fw=snr_fw, fh=snr_fh,
                         fthresh=snr_fthresh)
    return (snr - snr_bkg.back(dtype=np.float64)) / snr_bkg.rms(dtype=np.float64)


def _suppress_close_peaks(x, y, values, double min_distance):
    if min_distance <= 0.0 or len(x) <= 1:
        return np.arange(len(x), dtype=np.int64)

    order = np.argsort(values)[::-1]
    keep = []
    grid = {}
    cell = min_distance
    min_distance2 = min_distance * min_distance

    for idx in order:
        xi = float(x[idx])
        yi = float(y[idx])
        gx = int(np.floor(xi / cell))
        gy = int(np.floor(yi / cell))
        accept = True

        for ngx in range(gx - 1, gx + 2):
            for ngy in range(gy - 1, gy + 2):
                for kept_idx in grid.get((ngx, ngy), ()):
                    dx = xi - float(x[kept_idx])
                    dy = yi - float(y[kept_idx])
                    if dx * dx + dy * dy <= min_distance2:
                        accept = False
                        break
                if not accept:
                    break
            if not accept:
                break

        if accept:
            keep.append(idx)
            grid.setdefault((gx, gy), []).append(idx)

    keep = np.asarray(keep, dtype=np.int64)
    return keep[np.argsort(y[keep] * np.max(x + 1) + x[keep])]


@cython.boundscheck(False)
@cython.wraparound(False)
def _psf_peak_table_from_snr(np.ndarray[np.float64_t, ndim=2] snr not None,
                             double thresh, int maxfilter_size,
                             double min_distance):
    cdef Py_ssize_t h = snr.shape[0]
    cdef Py_ssize_t w = snr.shape[1]
    cdef Py_ssize_t radius = maxfilter_size // 2
    cdef Py_ssize_t y, x, yy, xx, y0, y1, x0, x1
    cdef Py_ssize_t npeak = 0
    cdef Py_ssize_t i, idx, kept_idx, cell_idx
    cdef Py_ssize_t gx, gy, ngx, ngy, grid_nx, grid_ny
    cdef double value, other, dx, dy, min_distance2, cell
    cdef bint is_peak
    cdef np.ndarray[np.int64_t, ndim=1] xpeak
    cdef np.ndarray[np.int64_t, ndim=1] ypeak
    cdef np.ndarray[np.float64_t, ndim=1] values
    cdef np.ndarray[np.int64_t, ndim=1] order
    cdef np.ndarray[np.int64_t, ndim=1] head
    cdef np.ndarray[np.int64_t, ndim=1] next_idx
    cdef np.ndarray[np.uint8_t, ndim=1] keep
    cdef object result

    if maxfilter_size < 1 or maxfilter_size % 2 != 1:
        raise ValueError("maxfilter_size must be a positive odd integer")

    for y in range(h):
        y0 = max(0, y - radius)
        y1 = min(h, y + radius + 1)
        for x in range(w):
            value = snr[y, x]
            if not isfinite(value) or value <= thresh:
                continue
            is_peak = True
            x0 = max(0, x - radius)
            x1 = min(w, x + radius + 1)
            for yy in range(y0, y1):
                for xx in range(x0, x1):
                    if yy == y and xx == x:
                        continue
                    other = snr[yy, xx]
                    if not isfinite(other) or value < other:
                        is_peak = False
                        break
                if not is_peak:
                    break
            if is_peak:
                npeak += 1

    xpeak = np.empty(npeak, dtype=np.int64)
    ypeak = np.empty(npeak, dtype=np.int64)
    values = np.empty(npeak, dtype=np.float64)
    npeak = 0
    for y in range(h):
        y0 = max(0, y - radius)
        y1 = min(h, y + radius + 1)
        for x in range(w):
            value = snr[y, x]
            if not isfinite(value) or value <= thresh:
                continue
            is_peak = True
            x0 = max(0, x - radius)
            x1 = min(w, x + radius + 1)
            for yy in range(y0, y1):
                for xx in range(x0, x1):
                    if yy == y and xx == x:
                        continue
                    other = snr[yy, xx]
                    if not isfinite(other) or value < other:
                        is_peak = False
                        break
                if not is_peak:
                    break
            if is_peak:
                xpeak[npeak] = x
                ypeak[npeak] = y
                values[npeak] = value
                npeak += 1

    keep = np.ones(npeak, dtype=np.uint8)
    if min_distance > 0.0 and npeak > 1:
        keep[:] = 0
        order = np.asarray(np.argsort(values)[::-1], dtype=np.int64)
        cell = min_distance
        min_distance2 = min_distance * min_distance
        grid_nx = <Py_ssize_t>(w / cell) + 1
        grid_ny = <Py_ssize_t>(h / cell) + 1
        head = np.empty(grid_nx * grid_ny, dtype=np.int64)
        head.fill(-1)
        next_idx = np.empty(npeak, dtype=np.int64)
        next_idx.fill(-1)

        for i in range(npeak):
            idx = order[i]
            gx = <Py_ssize_t>(xpeak[idx] / cell)
            gy = <Py_ssize_t>(ypeak[idx] / cell)
            is_peak = True
            for ngy in range(max(0, gy - 1), min(grid_ny, gy + 2)):
                for ngx in range(max(0, gx - 1), min(grid_nx, gx + 2)):
                    kept_idx = head[ngy * grid_nx + ngx]
                    while kept_idx >= 0:
                        dx = <double>xpeak[idx] - <double>xpeak[kept_idx]
                        dy = <double>ypeak[idx] - <double>ypeak[kept_idx]
                        if dx * dx + dy * dy <= min_distance2:
                            is_peak = False
                            break
                        kept_idx = next_idx[kept_idx]
                    if not is_peak:
                        break
                if not is_peak:
                    break
            if is_peak:
                keep[idx] = 1
                cell_idx = gy * grid_nx + gx
                next_idx[idx] = head[cell_idx]
                head[cell_idx] = idx

    npeak = 0
    for i in range(keep.shape[0]):
        if keep[i]:
            npeak += 1

    result = np.empty(npeak,
                      dtype=np.dtype([('x', np.float64),
                                      ('y', np.float64),
                                      ('snr', np.float64),
                                      ('xpeak', np.int64),
                                      ('ypeak', np.int64)]))
    npeak = 0
    for i in range(keep.shape[0]):
        if keep[i]:
            result['x'][npeak] = <double>xpeak[i]
            result['y'][npeak] = <double>ypeak[i]
            result['snr'][npeak] = values[i]
            result['xpeak'][npeak] = xpeak[i]
            result['ypeak'][npeak] = ypeak[i]
            npeak += 1
    return result


def psf_peaks(np.ndarray data not None, float thresh, PSF psf not None,
              var=None, err=None, gain=None, np.ndarray mask=None,
              double maskthresh=0.0, int maxfilter_size=3,
              double min_distance=1.5, bint return_snr=False,
              bint local_bkg=False, bint normalize_snr=False,
              int snr_bw=64, int snr_bh=64, int snr_fw=3, int snr_fh=3,
              float snr_fthresh=0.0):
    """psf_peaks(data, thresh, psf, var=None, err=None, mask=None, ...)

    Find local maxima in a PSF-matched significance image.

    This is a peak-candidate detector for crowded fields. It computes
    `psf_snr`, optionally normalizes that S/N image with `Background`, then
    returns local maxima above ``thresh``. Unlike `psf_extract`, it does not
    create connected-object footprints or deblend segmentation islands.

    Parameters
    ----------
    data : `~numpy.ndarray`
        2-d image array.
    thresh : float
        Peak threshold in PSF-matched S/N units.
    psf : `PSF`
        PSF model.
    var, err, gain, mask, maskthresh, local_bkg
        Passed through to `psf_snr`.
    maxfilter_size : int, optional
        Odd-sized square neighborhood used for local-maximum testing.
        Default is 3.
    min_distance : float, optional
        Minimum distance in pixels between retained peaks. Peaks closer than
        this are greedily suppressed in descending S/N order. Default is 1.5.
    return_snr : bool, optional
        If True, return the detection S/N image in addition to the peak table.
    normalize_snr : bool, optional
        If True, estimate and remove a spatial background from the
        PSF-matched significance image, then divide by its background RMS
        before finding peaks.
    snr_bw, snr_bh, snr_fw, snr_fh, snr_fthresh : optional
        Background mesh and filter parameters used for S/N-image
        normalization when ``normalize_snr=True``.

    Returns
    -------
    peaks : `~numpy.ndarray`
        Record array with fields ``x``, ``y``, ``snr``, ``xpeak``, and
        ``ypeak``.
    snr : `~numpy.ndarray`, optional
        Returned when ``return_snr=True``.
    """

    if maxfilter_size < 1 or maxfilter_size % 2 != 1:
        raise ValueError("maxfilter_size must be a positive odd integer")

    snr = psf_snr(data, psf, var=var, err=err, gain=gain,
                  mask=mask, maskthresh=maskthresh, local_bkg=local_bkg)
    if normalize_snr:
        snr = _normalize_psf_snr(snr, mask, maskthresh, snr_bw, snr_bh,
                                 snr_fw, snr_fh, snr_fthresh)

    result = _psf_peak_table_from_snr(
        np.ascontiguousarray(snr, dtype=np.float64),
        thresh, maxfilter_size, min_distance)

    if return_snr:
        return result, snr
    return result


def _empty_psf_peak_fit_catalog():
    return np.empty(0, dtype=np.dtype([('x', np.float64),
                                       ('y', np.float64),
                                       ('flux', np.float64),
                                       ('fluxerr', np.float64),
                                       ('fit_snr', np.float64),
                                       ('peak_snr', np.float64),
                                       ('qf', np.float64),
                                       ('rchi2', np.float64),
                                       ('fracflux', np.float64),
                                       ('xpeak', np.int64),
                                       ('ypeak', np.int64),
                                       ('chi2', np.float64),
                                       ('niter', np.int64),
                                       ('flag', np.short)]))


def _psf_peak_quality(np.ndarray data not None, PSF psf not None,
                      xfit, yfit, flux, keep,
                      var=None, err=None, gain=None,
                      np.ndarray mask=None, double maskthresh=0.0):
    cdef np.ndarray[np.float64_t, ndim=2] data_arr
    cdef np.ndarray[np.float64_t, ndim=2] model
    cdef np.ndarray[np.float64_t, ndim=2] var_arr
    cdef np.ndarray[np.float64_t, ndim=2] err_arr
    cdef np.ndarray[np.float64_t, ndim=2] mask_arr
    cdef np.ndarray[np.float64_t, ndim=1] x_arr
    cdef np.ndarray[np.float64_t, ndim=1] y_arr
    cdef np.ndarray[np.float64_t, ndim=1] flux_arr
    cdef np.ndarray[np.uint8_t, ndim=1] keep_arr
    cdef int rw = psf.ptr.rw
    cdef int rh = psf.ptr.rh
    cdef int halfw = rw // 2
    cdef int halfh = rh // 2
    cdef Py_ssize_t n = len(xfit)
    cdef Py_ssize_t nkeep = int(np.sum(keep))
    cdef Py_ssize_t outidx = 0
    cdef Py_ssize_t i
    cdef int ix, iy, x0, x1, y0, y1, sx0, sx1, sy0, sy1, sx, sy
    cdef int px, py, status
    cdef double dx, dy, psfsum, qf_val, numer, rawsum, invvar
    cdef double neighbor_sum, pix, model_pix, psfw, src_pix, varpix
    cdef double scalar_var = 1.0
    cdef double gain_value = 0.0
    cdef bint has_var_array = False
    cdef bint has_err_array = False
    cdef bint has_mask = mask is not None
    cdef bint has_gain = gain is not None and gain > 0.0
    cdef np.ndarray[np.float64_t, ndim=1] qf = np.zeros(nkeep, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] rchi2 = np.full(nkeep, np.nan,
                                                          dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] fracflux = np.full(nkeep, np.nan,
                                                             dtype=np.float64)

    data_arr = np.ascontiguousarray(data, dtype=np.float64)
    x_arr = np.ascontiguousarray(xfit, dtype=np.float64)
    y_arr = np.ascontiguousarray(yfit, dtype=np.float64)
    flux_arr = np.ascontiguousarray(flux, dtype=np.float64)
    keep_arr = np.ascontiguousarray(keep, dtype=np.uint8)
    model = np.zeros((data_arr.shape[0], data_arr.shape[1]), dtype=np.float64)
    if has_gain:
        gain_value = float(gain)

    all_good = np.isfinite(flux_arr) & np.isfinite(x_arr) & np.isfinite(y_arr)
    if np.any(all_good):
        model_psf(model, x_arr[all_good], y_arr[all_good],
                  flux_arr[all_good], psf)

    if var is not None:
        if np.ndim(var) == 0:
            scalar_var = float(var)
        else:
            var_arr = np.ascontiguousarray(var, dtype=np.float64)
            if (var_arr.shape[0] != data_arr.shape[0] or
                    var_arr.shape[1] != data_arr.shape[1]):
                raise ValueError("var has wrong shape")
            has_var_array = True
    elif err is not None:
        if np.ndim(err) == 0:
            scalar_var = float(err) * float(err)
        else:
            err_arr = np.ascontiguousarray(err, dtype=np.float64)
            if (err_arr.shape[0] != data_arr.shape[0] or
                    err_arr.shape[1] != data_arr.shape[1]):
                raise ValueError("err has wrong shape")
            has_err_array = True
    if has_mask:
        mask_arr = np.ascontiguousarray(mask, dtype=np.float64)
        if (mask_arr.shape[0] != data_arr.shape[0] or
                mask_arr.shape[1] != data_arr.shape[1]):
            raise ValueError("mask has wrong shape")

    for i in range(n):
        if not keep_arr[i]:
            continue
        if not isfinite(x_arr[i]) or not isfinite(y_arr[i]):
            outidx += 1
            continue

        ix = int(x_arr[i] + 0.5)
        iy = int(y_arr[i] + 0.5)
        dx = x_arr[i] - ix
        dy = y_arr[i] - iy

        status = sep_psf_build(psf.ptr, x_arr[i], y_arr[i])
        _assert_ok(status)
        status = sep_psf_resample(psf.ptr, dx, dy)
        _assert_ok(status)

        x0 = ix - halfw
        y0 = iy - halfh
        x1 = x0 + rw
        y1 = y0 + rh
        sx0 = sy0 = 0
        sx1 = rw
        sy1 = rh
        if x0 < 0:
            sx0 = -x0
            x0 = 0
        if y0 < 0:
            sy0 = -y0
            y0 = 0
        if x1 > data_arr.shape[1]:
            sx1 -= x1 - data_arr.shape[1]
            x1 = data_arr.shape[1]
        if y1 > data_arr.shape[0]:
            sy1 -= y1 - data_arr.shape[0]
            y1 = data_arr.shape[0]

        if x0 >= x1 or y0 >= y1:
            outidx += 1
            continue

        psfsum = 0.0
        for sy in range(rh):
            for sx in range(rw):
                psfsum += psf.ptr.resi[sy * rw + sx]
        if psfsum <= 0.0:
            outidx += 1
            continue

        qf_val = 0.0
        numer = 0.0
        rawsum = 0.0
        neighbor_sum = 0.0
        for py in range(y0, y1):
            sy = sy0 + py - y0
            for px in range(x0, x1):
                sx = sx0 + px - x0
                pix = data_arr[py, px]
                if not isfinite(pix):
                    continue
                if has_mask and mask_arr[py, px] > maskthresh:
                    continue
                if has_var_array:
                    varpix = var_arr[py, px]
                elif has_err_array:
                    varpix = err_arr[py, px]
                    varpix *= varpix
                else:
                    varpix = scalar_var
                if not isfinite(varpix) or varpix <= 0.0:
                    continue
                if has_gain and pix > 0.0:
                    varpix += pix / gain_value

                psfw = psf.ptr.resi[sy * rw + sx] / psfsum
                src_pix = flux_arr[i] * psfw
                model_pix = model[py, px]
                qf_val += psfw
                rawsum += pix * psfw
                neighbor_sum += (pix - model_pix + src_pix) * psfw
                invvar = 1.0 / varpix
                numer += (pix - model_pix) * (pix - model_pix) * invvar * psfw

        qf[outidx] = qf_val
        if qf_val > 0.0:
            rchi2[outidx] = numer / qf_val
            if rawsum != 0.0:
                fracflux[outidx] = neighbor_sum / rawsum

        outidx += 1

    return qf, rchi2, fracflux


def _fit_psf_peaks(np.ndarray data not None, peaks, PSF psf not None,
                   var=None, err=None, gain=None, np.ndarray mask=None,
                   double maskthresh=0.0, fit_snr=5.0,
                   bint fit_positions=True, bint keep_flagged=False,
                   int maxiter=20, double group_factor=2.0,
                   min_qf=None, max_rchi2=None, min_fracflux=None,
                   bint compute_quality=True):
    if len(peaks) == 0:
        return _empty_psf_peak_fit_catalog()

    flux, fluxerr, xfit, yfit, flag, chi2, niter = psf_fit(
        data, peaks['x'], peaks['y'], psf, var=var, err=err, gain=gain,
        mask=mask, maskthresh=maskthresh, maxiter=maxiter,
        fit_positions=fit_positions, grouped=True,
        group_factor=group_factor,
    )
    fit_snr_values = flux / np.maximum(fluxerr, 1.0e-30)
    keep = np.isfinite(fit_snr_values)
    if fit_snr is not None:
        keep &= fit_snr_values > fit_snr
    if not keep_flagged:
        keep &= flag == 0
    if compute_quality:
        qf, rchi2, fracflux = _psf_peak_quality(
            data, psf, xfit, yfit, flux, keep, var=var, err=err, gain=gain,
            mask=mask, maskthresh=maskthresh)
        quality_keep = np.ones(len(qf), dtype=bool)
        if min_qf is not None:
            quality_keep &= qf >= min_qf
        if max_rchi2 is not None:
            quality_keep &= rchi2 <= max_rchi2
        if min_fracflux is not None:
            quality_keep &= fracflux >= min_fracflux
        if not np.all(quality_keep):
            kept_idx = np.flatnonzero(keep)
            keep = np.zeros_like(keep, dtype=bool)
            keep[kept_idx[quality_keep]] = True
            qf = qf[quality_keep]
            rchi2 = rchi2[quality_keep]
            fracflux = fracflux[quality_keep]
    else:
        qf = np.full(np.sum(keep), np.nan, dtype=np.float64)
        rchi2 = np.full(np.sum(keep), np.nan, dtype=np.float64)
        fracflux = np.full(np.sum(keep), np.nan, dtype=np.float64)

    result = np.empty(np.sum(keep),
                      dtype=np.dtype([('x', np.float64),
                                      ('y', np.float64),
                                      ('flux', np.float64),
                                      ('fluxerr', np.float64),
                                      ('fit_snr', np.float64),
                                      ('peak_snr', np.float64),
                                      ('qf', np.float64),
                                      ('rchi2', np.float64),
                                      ('fracflux', np.float64),
                                      ('xpeak', np.int64),
                                      ('ypeak', np.int64),
                                      ('chi2', np.float64),
                                      ('niter', np.int64),
                                      ('flag', np.short)]))
    result['x'] = xfit[keep]
    result['y'] = yfit[keep]
    result['flux'] = flux[keep]
    result['fluxerr'] = fluxerr[keep]
    result['fit_snr'] = fit_snr_values[keep]
    result['peak_snr'] = peaks['snr'][keep]
    result['qf'] = qf
    result['rchi2'] = rchi2
    result['fracflux'] = fracflux
    result['xpeak'] = peaks['xpeak'][keep]
    result['ypeak'] = peaks['ypeak'][keep]
    result['chi2'] = chi2[keep]
    result['niter'] = niter[keep]
    result['flag'] = flag[keep]
    return result


cdef int _compare_double(const void *a, const void *b) noexcept nogil:
    cdef double da = (<double *>a)[0]
    cdef double db = (<double *>b)[0]
    if da < db:
        return -1
    if da > db:
        return 1
    return 0


cdef double _median_double_buffer(double *buf, Py_ssize_t n) noexcept nogil:
    if n <= 0:
        return 0.0
    qsort(buf, <size_t>n, sizeof(double), _compare_double)
    if n % 2:
        return buf[n // 2]
    return 0.5 * (buf[n // 2 - 1] + buf[n // 2])


@cython.boundscheck(False)
@cython.wraparound(False)
def _psf_peak_local_sky(np.ndarray data not None, var=None,
                        np.ndarray mask=None, double maskthresh=0.0,
                        int box_size=20):
    cdef np.ndarray[np.float64_t, ndim=2] image
    cdef np.ndarray[np.float64_t, ndim=2] var_arr
    cdef np.ndarray[np.float64_t, ndim=2] mask_arr
    cdef np.ndarray[np.float64_t, ndim=2] val
    cdef np.ndarray[np.float64_t, ndim=2] used
    cdef np.ndarray[np.float64_t, ndim=2] sky
    cdef np.ndarray[np.float64_t, ndim=1] yp
    cdef np.ndarray[np.float64_t, ndim=1] xp
    cdef Py_ssize_t h, w, nbin_y, nbin_x, max_box
    cdef Py_ssize_t iy, ix, y, x, yy, xx, y0, y1, x0, x1
    cdef Py_ssize_t ylo, yhi, xlo, xhi, count, nvalid
    cdef Py_ssize_t k, nfilled, iteration, radius
    cdef bint has_var = False
    cdef bint has_mask = mask is not None
    cdef double *buf = NULL
    cdef double pix, varpix, weight, weighted_sum, weight_sum
    cdef double sigma = 0.4
    cdef double sigma2 = 2.0 * sigma * sigma
    cdef double pos, frac

    if box_size <= 0:
        raise ValueError("peak_local_sky_box must be positive")

    image = np.ascontiguousarray(data, dtype=np.float64)
    h = image.shape[0]
    w = image.shape[1]
    nbin_y = (h + box_size - 1) // box_size
    nbin_x = (w + box_size - 1) // box_size
    val = np.zeros((nbin_y, nbin_x), dtype=np.float64)
    used = np.zeros((nbin_y, nbin_x), dtype=np.float64)
    sky = np.empty((h, w), dtype=np.float64)

    if var is not None and np.ndim(var) != 0:
        var_arr = np.ascontiguousarray(var, dtype=np.float64)
        if var_arr.shape[0] != h or var_arr.shape[1] != w:
            raise ValueError("var has wrong shape")
        has_var = True
    if has_mask:
        mask_arr = np.ascontiguousarray(mask, dtype=np.float64)
        if mask_arr.shape[0] != h or mask_arr.shape[1] != w:
            raise ValueError("mask has wrong shape")

    max_box = ((h + nbin_y - 1) // nbin_y + 1) * ((w + nbin_x - 1) // nbin_x + 1)
    buf = <double *>PyMem_Malloc(max_box * sizeof(double))
    if buf == NULL:
        raise MemoryError()

    try:
        for iy in range(nbin_y):
            y0 = (iy * h) // nbin_y
            y1 = ((iy + 1) * h) // nbin_y
            for ix in range(nbin_x):
                x0 = (ix * w) // nbin_x
                x1 = ((ix + 1) * w) // nbin_x
                nvalid = 0
                for y in range(y0, y1):
                    for x in range(x0, x1):
                        pix = image[y, x]
                        if not isfinite(pix):
                            continue
                        if has_var:
                            varpix = var_arr[y, x]
                            if not isfinite(varpix) or varpix <= 0.0:
                                continue
                        if has_mask and mask_arr[y, x] > maskthresh:
                            continue
                        buf[nvalid] = pix
                        nvalid += 1
                used[iy, ix] = <double>nvalid
                if nvalid > 0:
                    val[iy, ix] = _median_double_buffer(buf, nvalid)

        for iy in range(nbin_y):
            for ix in range(nbin_x):
                if used[iy, ix] < 20.0:
                    val[iy, ix] = 0.0
                    used[iy, ix] = 0.0
    finally:
        PyMem_Free(buf)

    radius = 2
    for iteration in range(100):
        nfilled = 0
        for iy in range(nbin_y):
            for ix in range(nbin_x):
                if used[iy, ix] != 0.0:
                    continue
                weighted_sum = 0.0
                weight_sum = 0.0
                ylo = max(0, iy - radius)
                yhi = min(nbin_y, iy + radius + 1)
                xlo = max(0, ix - radius)
                xhi = min(nbin_x, ix + radius + 1)
                for yy in range(ylo, yhi):
                    for xx in range(xlo, xhi):
                        if used[yy, xx] == 0.0:
                            continue
                        weight = exp(-(
                            (yy - iy) * (yy - iy) +
                            (xx - ix) * (xx - ix)) / sigma2)
                        weighted_sum += weight * val[yy, xx]
                        weight_sum += weight
                if weight_sum > 1.0e-10:
                    val[iy, ix] = weighted_sum / weight_sum
                    used[iy, ix] = 1.0
                    nfilled += 1
        if nfilled == 0:
            break

    weighted_sum = 0.0
    weight_sum = 0.0
    for iy in range(nbin_y):
        for ix in range(nbin_x):
            if used[iy, ix] != 0.0:
                weighted_sum += val[iy, ix]
                weight_sum += 1.0
    if weight_sum > 0.0:
        weighted_sum /= weight_sum
    for iy in range(nbin_y):
        for ix in range(nbin_x):
            if used[iy, ix] == 0.0:
                val[iy, ix] = weighted_sum
                used[iy, ix] = 1.0

    yp = np.empty(h, dtype=np.float64)
    xp = np.empty(w, dtype=np.float64)
    if nbin_y == 1:
        for y in range(h):
            yp[y] = 0.0
    else:
        for y in range(h):
            pos = ((<double>y + 0.5 * h / nbin_y) * nbin_y / h) - 0.5
            if pos < 0.0:
                pos = 0.0
            elif pos > nbin_y - 1:
                pos = nbin_y - 1
            yp[y] = pos
    if nbin_x == 1:
        for x in range(w):
            xp[x] = 0.0
    else:
        for x in range(w):
            pos = ((<double>x + 0.5 * w / nbin_x) * nbin_x / w) - 0.5
            if pos < 0.0:
                pos = 0.0
            elif pos > nbin_x - 1:
                pos = nbin_x - 1
            xp[x] = pos

    for y in range(h):
        iy = <Py_ssize_t>yp[y]
        if iy >= nbin_y - 1:
            iy = nbin_y - 1
            frac = 0.0
        else:
            frac = yp[y] - iy
        for x in range(w):
            ix = <Py_ssize_t>xp[x]
            if ix >= nbin_x - 1:
                ix = nbin_x - 1
                pos = 0.0
            else:
                pos = xp[x] - ix
            if iy == nbin_y - 1 and ix == nbin_x - 1:
                sky[y, x] = val[iy, ix]
            elif iy == nbin_y - 1:
                sky[y, x] = (1.0 - pos) * val[iy, ix] + pos * val[iy, ix + 1]
            elif ix == nbin_x - 1:
                sky[y, x] = (1.0 - frac) * val[iy, ix] + frac * val[iy + 1, ix]
            else:
                sky[y, x] = (
                    (1.0 - frac) * (
                        (1.0 - pos) * val[iy, ix] + pos * val[iy, ix + 1]) +
                    frac * (
                        (1.0 - pos) * val[iy + 1, ix] +
                        pos * val[iy + 1, ix + 1])
                )

    return sky


@cython.boundscheck(False)
@cython.wraparound(False)
def _peaks_far_from_sources(peaks, sources, min_distance):
    cdef Py_ssize_t npeak = len(peaks)
    cdef Py_ssize_t nsrc = len(sources)
    cdef Py_ssize_t i, j, gx, gy, ngx, ngy, grid_nx, grid_ny, cell_idx
    cdef double cell, radius2, dx, dy, minx, miny, maxx, maxy, px, py
    cdef np.ndarray[np.float64_t, ndim=1] peak_x
    cdef np.ndarray[np.float64_t, ndim=1] peak_y
    cdef np.ndarray[np.float64_t, ndim=1] source_x
    cdef np.ndarray[np.float64_t, ndim=1] source_y
    cdef np.ndarray[np.int64_t, ndim=1] head
    cdef np.ndarray[np.int64_t, ndim=1] next_idx
    cdef np.ndarray[np.uint8_t, ndim=1] keep

    if npeak == 0 or nsrc == 0 or min_distance is None:
        return np.ones(len(peaks), dtype=bool)
    if min_distance <= 0.0:
        return np.ones(len(peaks), dtype=bool)

    peak_x = np.ascontiguousarray(peaks['x'], dtype=np.float64)
    peak_y = np.ascontiguousarray(peaks['y'], dtype=np.float64)
    source_x = np.ascontiguousarray(sources['x'], dtype=np.float64)
    source_y = np.ascontiguousarray(sources['y'], dtype=np.float64)

    minx = source_x[0]
    maxx = source_x[0]
    miny = source_y[0]
    maxy = source_y[0]
    for i in range(nsrc):
        if source_x[i] < minx:
            minx = source_x[i]
        if source_x[i] > maxx:
            maxx = source_x[i]
        if source_y[i] < miny:
            miny = source_y[i]
        if source_y[i] > maxy:
            maxy = source_y[i]
    for i in range(npeak):
        if peak_x[i] < minx:
            minx = peak_x[i]
        if peak_x[i] > maxx:
            maxx = peak_x[i]
        if peak_y[i] < miny:
            miny = peak_y[i]
        if peak_y[i] > maxy:
            maxy = peak_y[i]

    cell = max(float(min_distance), 1.0)
    radius2 = float(min_distance) * float(min_distance)
    grid_nx = <Py_ssize_t>((maxx - minx) / cell) + 2
    grid_ny = <Py_ssize_t>((maxy - miny) / cell) + 2
    head = np.empty(grid_nx * grid_ny, dtype=np.int64)
    head.fill(-1)
    next_idx = np.empty(nsrc, dtype=np.int64)
    next_idx.fill(-1)

    for i in range(nsrc):
        gx = <Py_ssize_t>((source_x[i] - minx) / cell)
        gy = <Py_ssize_t>((source_y[i] - miny) / cell)
        cell_idx = gy * grid_nx + gx
        next_idx[i] = head[cell_idx]
        head[cell_idx] = i

    keep = np.ones(npeak, dtype=np.uint8)
    for i in range(npeak):
        gx = <Py_ssize_t>((peak_x[i] - minx) / cell)
        gy = <Py_ssize_t>((peak_y[i] - miny) / cell)
        for ngy in range(max(0, gy - 1), min(grid_ny, gy + 2)):
            for ngx in range(max(0, gx - 1), min(grid_nx, gx + 2)):
                j = head[ngy * grid_nx + ngx]
                while j >= 0:
                    dx = peak_x[i] - source_x[j]
                    dy = peak_y[i] - source_y[j]
                    if dx * dx + dy * dy < radius2:
                        keep[i] = 0
                        break
                    j = next_idx[j]
                if not keep[i]:
                    break
            if not keep[i]:
                break

    return keep.astype(bool)


def _subtract_psf_peak_catalog(residual, catalog, PSF psf not None):
    if len(catalog) == 0:
        return
    model = np.zeros(residual.shape, dtype=np.float64)
    model_psf(model, catalog['x'], catalog['y'], catalog['flux'], psf)
    residual -= model


def _psf_peak_model(shape, catalog, PSF psf not None):
    model = np.zeros(shape, dtype=np.float64)
    if len(catalog) > 0:
        model_psf(model, catalog['x'], catalog['y'], catalog['flux'], psf)
    return model


def _psf_peak_candidates_from_fit_catalog(catalog):
    peaks = np.empty(len(catalog),
                     dtype=np.dtype([('x', np.float64),
                                     ('y', np.float64),
                                     ('snr', np.float64),
                                     ('xpeak', np.int64),
                                     ('ypeak', np.int64)]))
    peaks['x'] = catalog['x']
    peaks['y'] = catalog['y']
    peaks['snr'] = catalog['peak_snr']
    peaks['xpeak'] = catalog['xpeak']
    peaks['ypeak'] = catalog['ypeak']
    return peaks


def _rebuild_psf_peak_residual(np.ndarray data not None, catalog,
                               PSF psf not None):
    residual = np.ascontiguousarray(data, dtype=np.float64).copy()
    _subtract_psf_peak_catalog(residual, catalog, psf)
    return residual


def _refit_psf_peaks_with_model_sky(np.ndarray data not None, catalog,
                                    PSF psf not None, var=None, err=None,
                                    gain=None, np.ndarray mask=None,
                                    double maskthresh=0.0, fit_snr=5.0,
                                    bint fit_positions=True,
                                    bint keep_flagged=False,
                                    int maxiter=20,
                                    double group_factor=2.0,
                                    min_qf=None, max_rchi2=None,
                                    min_fracflux=None,
                                    int peak_local_sky_box=20):
    if len(catalog) == 0:
        return catalog

    data_arr = np.ascontiguousarray(data, dtype=np.float64)
    model = _psf_peak_model(data_arr.shape, catalog, psf)
    sky = _psf_peak_local_sky(
        data_arr - model, var=var, mask=mask, maskthresh=maskthresh,
        box_size=peak_local_sky_box)
    peaks = _psf_peak_candidates_from_fit_catalog(catalog)
    return _fit_psf_peaks(
        data_arr - sky, peaks, psf, var=var, err=err, gain=gain, mask=mask,
        maskthresh=maskthresh, fit_snr=fit_snr,
        fit_positions=fit_positions, keep_flagged=keep_flagged,
        maxiter=maxiter, group_factor=group_factor,
        min_qf=min_qf, max_rchi2=max_rchi2,
        min_fracflux=min_fracflux,
    )


def _psf_extract_peaks_iterative(np.ndarray data not None, float thresh,
                                 PSF psf not None, var=None, err=None,
                                 gain=None, np.ndarray mask=None,
                                 double maskthresh=0.0,
                                 bint return_snr=False,
                                 bint local_bkg=False,
                                 bint normalize_snr=False,
                                 int snr_bw=64, int snr_bh=64,
                                 int snr_fw=3, int snr_fh=3,
                                 float snr_fthresh=0.0,
                                 double peak_min_distance=1.5,
                                 int maxfilter_size=3,
                                 fit_snr=5.0,
                                 bint fit_positions=True,
                                 bint keep_flagged=False,
                                 int fit_maxiter=20,
                                 double group_factor=2.0,
                                 min_qf=None, max_rchi2=None,
                                 min_fracflux=None,
                                 int peak_iterations=1,
                                 peak_duplicate_distance=None,
                                 bint peak_local_sky=False,
                                 int peak_local_sky_box=20):
    data_arr = np.ascontiguousarray(data, dtype=np.float64)
    residual = data_arr.copy()
    fit_data = data_arr
    catalog = _empty_psf_peak_fit_catalog()
    last_snr = None
    duplicate_distance = peak_duplicate_distance
    if duplicate_distance is None:
        duplicate_distance = peak_min_distance

    for _ in range(peak_iterations):
        if peak_local_sky:
            sky = _psf_peak_local_sky(
                residual, var=var, mask=mask, maskthresh=maskthresh,
                box_size=peak_local_sky_box)
            detect_residual = residual - sky
            fit_data = data_arr - sky
        else:
            detect_residual = residual
            fit_data = data_arr

        peak_result = psf_peaks(
            detect_residual, thresh, psf, var=var, err=err, gain=gain,
            mask=mask,
            maskthresh=maskthresh, maxfilter_size=maxfilter_size,
            min_distance=peak_min_distance, return_snr=return_snr,
            local_bkg=local_bkg, normalize_snr=normalize_snr,
            snr_bw=snr_bw, snr_bh=snr_bh, snr_fw=snr_fw, snr_fh=snr_fh,
            snr_fthresh=snr_fthresh,
        )
        if return_snr:
            peaks, last_snr = peak_result
        else:
            peaks = peak_result

        if len(catalog) > 0 and len(peaks) > 0:
            peaks = peaks[_peaks_far_from_sources(
                peaks, catalog, duplicate_distance)]
        if len(peaks) == 0:
            break

        if len(catalog) == 0:
            refit_peaks = peaks
        else:
            refit_peaks = np.concatenate([
                _psf_peak_candidates_from_fit_catalog(catalog),
                peaks,
            ])

        fitted = _fit_psf_peaks(
            fit_data, refit_peaks, psf, var=var, err=err, gain=gain, mask=mask,
            maskthresh=maskthresh, fit_snr=fit_snr,
            fit_positions=fit_positions, keep_flagged=keep_flagged,
            maxiter=fit_maxiter, group_factor=group_factor,
            min_qf=min_qf, max_rchi2=max_rchi2,
            min_fracflux=min_fracflux,
            compute_quality=not peak_local_sky,
        )
        if len(fitted) == 0:
            break

        if peak_local_sky:
            fitted = _refit_psf_peaks_with_model_sky(
                data_arr, fitted, psf, var=var, err=err, gain=gain,
                mask=mask, maskthresh=maskthresh, fit_snr=fit_snr,
                fit_positions=fit_positions, keep_flagged=keep_flagged,
                maxiter=fit_maxiter, group_factor=group_factor,
                min_qf=min_qf, max_rchi2=max_rchi2,
                min_fracflux=min_fracflux,
                peak_local_sky_box=peak_local_sky_box,
            )
            if len(fitted) == 0:
                break

        if len(catalog) > 0:
            new_sources = _peaks_far_from_sources(
                fitted, catalog, duplicate_distance)
            if not np.any(new_sources):
                break

        catalog = fitted
        residual = _rebuild_psf_peak_residual(data_arr, catalog, psf)

    if return_snr:
        return catalog, last_snr
    return catalog


def psf_extract(np.ndarray data not None, float thresh, PSF psf not None,
                var=None, err=None, gain=None, np.ndarray mask=None,
                double maskthresh=0.0, int minarea=1,
                int deblend_nthresh=32, double deblend_cont=0.005,
                bint clean=True, double clean_param=1.0,
                segmentation_map=False, bint return_snr=False,
                bint local_bkg=False, bint normalize_snr=False,
                int snr_bw=64, int snr_bh=64, int snr_fw=3, int snr_fh=3,
                float snr_fthresh=0.0, mode="segments",
                double peak_min_distance=1.5, int maxfilter_size=3,
                fit_snr=5.0, bint fit_positions=True,
                bint keep_flagged=False, int fit_maxiter=20,
                double group_factor=2.0,
                min_qf=None, max_rchi2=None, min_fracflux=None,
                int peak_iterations=1, peak_duplicate_distance=None,
                bint peak_local_sky=False, int peak_local_sky_box=20):
    """psf_extract(data, thresh, psf, var=None, err=None, mask=None, ...)

    Extract sources from a PSF-matched significance image.

    In ``mode='segments'`` (default), this is a convenience wrapper around
    `psf_snr` and `extract`: it computes the PSF-matched significance image,
    then runs connected-component source extraction on that image with no
    additional filtering.

    In ``mode='peaks'``, this finds local maxima in the PSF-matched
    significance image with `psf_peaks`, then fits the PSF at those peak
    positions and prunes by fitted S/N. Set ``peak_iterations`` greater than
    one to repeat this process on residual images after subtracting accepted
    fitted sources.

    Parameters
    ----------
    data : `~numpy.ndarray`
        2-d image array.
    thresh : float
        Detection threshold in PSF-matched S/N units.
    psf : `PSF`
        PSF model.
    var : float or `~numpy.ndarray`, optional
        Variance (scalar or 2-d array). Mutually exclusive with ``err``.
    err : float or `~numpy.ndarray`, optional
        Standard deviation (scalar or 2-d array). Mutually exclusive
        with ``var``.
    gain : float, optional
        Effective gain in electrons per data unit.
    mask : `~numpy.ndarray`, optional
        Mask array. The mask is used both while computing the significance
        image and while extracting detections from it.
    maskthresh : float, optional
        Mask threshold.
    minarea : int, optional
        Minimum number of connected pixels above threshold in the significance
        image. Default is 1.
    deblend_nthresh, deblend_cont, clean, clean_param, segmentation_map
        Passed through to `extract`.
    return_snr : bool, optional
        If True, return the PSF-matched significance image in addition to the
        extraction result.
    local_bkg : bool, optional
        If True, compute the detection image with `psf_snr(...,
        local_bkg=True)`.
    normalize_snr : bool, optional
        If True, estimate and remove a spatial background from the
        PSF-matched significance image, then divide by its background RMS
        before extracting detections.
    snr_bw, snr_bh, snr_fw, snr_fh, snr_fthresh : optional
        Background mesh and filter parameters used for S/N-image
        normalization when ``normalize_snr=True``.
    mode : {'segments', 'peaks'}, optional
        Detection mode. ``'segments'`` uses connected-component extraction.
        ``'peaks'`` uses local maxima followed by PSF-fit pruning.
    peak_min_distance : float, optional
        Minimum distance between local maxima in ``mode='peaks'``.
    maxfilter_size : int, optional
        Odd-sized local-maximum neighborhood in ``mode='peaks'``.
    fit_snr : float or None, optional
        Minimum fitted ``flux / fluxerr`` retained in ``mode='peaks'``.
        Set to None to return unfit peak candidates. Default is 5.0.
    fit_positions, keep_flagged, fit_maxiter : optional
        PSF fitting controls used in ``mode='peaks'``.
    group_factor : float, optional
        Grouping radius factor passed to the peak-mode PSF fits.
    min_qf, max_rchi2, min_fracflux : float or None, optional
        Optional quality cuts applied in ``mode='peaks'`` after PSF fitting
        and diagnostic computation. By default, no quality cuts are applied.
    peak_iterations : int, optional
        Number of residual-detection iterations in ``mode='peaks'``. Values
        greater than one require fitted peak catalogs, so ``fit_snr`` may not
        be None. Default is 1.
    peak_duplicate_distance : float or None, optional
        Minimum distance from already accepted peak-mode sources for accepting
        peaks found in later residual iterations. By default, uses
        ``peak_min_distance``.
    peak_local_sky : bool, optional
        If True in ``mode='peaks'``, estimate and subtract a local sky image
        before peak fitting, then refit after re-estimating the sky from the
        image with the fitted source model subtracted. For residual
        iterations, the local sky is re-estimated from the current residual
        image. Default is False.
    peak_local_sky_box : int, optional
        Mesh size in pixels for peak-mode local sky estimation.

    Returns
    -------
    objects : `~numpy.ndarray`
        Extracted object catalog from the PSF-matched significance image in
        ``mode='segments'``. In ``mode='peaks'``, the returned table contains
        fitted positions, fluxes, fitted S/N, peak S/N, peak pixel positions,
        PSF-weighted fit diagnostics, fit chi-square, iteration counts, and
        fit flags. If ``fit_snr=None``, it contains the raw peak table
        returned by `psf_peaks`.
    segmap : `~numpy.ndarray`, optional
        Returned when ``segmentation_map=True``.
    snr : `~numpy.ndarray`, optional
        Returned when ``return_snr=True``.
    """

    if mode == "peaks":
        if type(segmentation_map) is np.ndarray or segmentation_map:
            raise ValueError("segmentation_map is not supported with mode='peaks'")
        if peak_iterations < 1:
            raise ValueError("peak_iterations must be at least 1")
        if peak_iterations > 1 and fit_snr is None:
            raise ValueError("peak_iterations > 1 requires fit_snr not None")
        if peak_iterations > 1:
            return _psf_extract_peaks_iterative(
                data, thresh, psf, var=var, err=err, gain=gain, mask=mask,
                maskthresh=maskthresh, return_snr=return_snr,
                local_bkg=local_bkg, normalize_snr=normalize_snr,
                snr_bw=snr_bw, snr_bh=snr_bh, snr_fw=snr_fw,
                snr_fh=snr_fh, snr_fthresh=snr_fthresh,
                peak_min_distance=peak_min_distance,
                maxfilter_size=maxfilter_size, fit_snr=fit_snr,
                fit_positions=fit_positions, keep_flagged=keep_flagged,
                fit_maxiter=fit_maxiter, group_factor=group_factor,
                min_qf=min_qf,
                max_rchi2=max_rchi2, min_fracflux=min_fracflux,
                peak_iterations=peak_iterations,
                peak_duplicate_distance=peak_duplicate_distance,
                peak_local_sky=peak_local_sky,
                peak_local_sky_box=peak_local_sky_box,
            )
        peak_data = data
        if peak_local_sky:
            peak_sky = _psf_peak_local_sky(
                data, var=var, mask=mask, maskthresh=maskthresh,
                box_size=peak_local_sky_box)
            peak_data = np.ascontiguousarray(data, dtype=np.float64) - peak_sky
        peak_result = psf_peaks(
            peak_data, thresh, psf, var=var, err=err, gain=gain, mask=mask,
            maskthresh=maskthresh, maxfilter_size=maxfilter_size,
            min_distance=peak_min_distance, return_snr=return_snr,
            local_bkg=local_bkg, normalize_snr=normalize_snr, snr_bw=snr_bw,
            snr_bh=snr_bh, snr_fw=snr_fw, snr_fh=snr_fh,
            snr_fthresh=snr_fthresh,
        )
        if return_snr:
            peaks, snr = peak_result
        else:
            peaks = peak_result
        if fit_snr is None:
            result = peaks
        else:
            result = _fit_psf_peaks(
                peak_data, peaks, psf, var=var, err=err, gain=gain, mask=mask,
                maskthresh=maskthresh, fit_snr=fit_snr,
                fit_positions=fit_positions, keep_flagged=keep_flagged,
                maxiter=fit_maxiter, group_factor=group_factor, min_qf=min_qf,
                max_rchi2=max_rchi2, min_fracflux=min_fracflux,
                compute_quality=not peak_local_sky,
            )
            if peak_local_sky:
                result = _refit_psf_peaks_with_model_sky(
                    data, result, psf, var=var, err=err, gain=gain,
                    mask=mask, maskthresh=maskthresh, fit_snr=fit_snr,
                    fit_positions=fit_positions, keep_flagged=keep_flagged,
                    maxiter=fit_maxiter, group_factor=group_factor,
                    min_qf=min_qf, max_rchi2=max_rchi2,
                    min_fracflux=min_fracflux,
                    peak_local_sky_box=peak_local_sky_box,
                )
        if return_snr:
            return result, snr
        return result

    if mode != "segments":
        raise ValueError("mode must be 'segments' or 'peaks'")

    snr = psf_snr(data, psf, var=var, err=err, gain=gain,
                  mask=mask, maskthresh=maskthresh, local_bkg=local_bkg)
    if normalize_snr:
        snr = _normalize_psf_snr(snr, mask, maskthresh, snr_bw, snr_bh,
                                 snr_fw, snr_fh, snr_fthresh)
    result = extract(
        snr,
        thresh,
        err=None,
        var=None,
        gain=None,
        mask=mask,
        maskthresh=maskthresh,
        minarea=minarea,
        filter_kernel=None,
        deblend_nthresh=deblend_nthresh,
        deblend_cont=deblend_cont,
        clean=clean,
        clean_param=clean_param,
        segmentation_map=segmentation_map,
    )
    if return_snr:
        if type(segmentation_map) is np.ndarray or segmentation_map:
            objects, segmap = result
            return objects, segmap, snr
        return result, snr
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def psf_fit(np.ndarray data not None, x, y, PSF psf not None,
            var=None, err=None, gain=None, np.ndarray mask=None,
            double maskthresh=0.0,
            seg_id=None, np.ndarray segmap=None,
            bint grouped=False, double group_factor=2.0,
            int maxiter=20, bint fit_positions=True,
            double fit_radius=0.0, double damp_snthresh=0.0):
    """psf_fit(data, x, y, psf, ...)

    Fit a PSF model to sources in image data.

    Parameters
    ----------
    data : `~numpy.ndarray`
        2-d array to fit.
    x, y : array_like
        Initial source positions.
    psf : `PSF`
        PSF model.
    var : float or `~numpy.ndarray`, optional
        Variance (scalar or 2-d array). Mutually exclusive with ``err``.
    err : float or `~numpy.ndarray`, optional
        Standard deviation (scalar or 2-d array). Mutually exclusive
        with ``var``.
    gain : float, optional
        Effective gain in electrons per data unit.
    mask : `~numpy.ndarray`, optional
        Mask array.
    maskthresh : float, optional
        Mask threshold.
    seg_id : array_like, optional
        Segmentation IDs.
    segmap : `~numpy.ndarray`, optional
        Segmentation map.
    grouped : bool, optional
        If True, fit overlapping sources simultaneously (default False).
        When combined with ``fit_positions=False``, uses grouped NNLS
        flux solver at fixed positions for deblending in crowded fields.
    group_factor : float, optional
        Local fitting halo factor for grouped fits (default 2.0). Source
        connectivity is limited to direct fit-support overlap; increasing
        this value expands the local context used within a grouped fit
        without merging non-overlapping sources into the same connected
        component. Values below 1 behave like 1.0. When ``fit_radius=0``,
        the support used for grouping is derived from the PSF itself rather
        than from the full stamp extent.
    maxiter : int, optional
        Maximum fitting iterations (default 20).
    fit_positions : bool, optional
        If True, fit positions as well as fluxes (default True).
    fit_radius : float, optional
        If > 0, only pixels within this radius (in image pixels) of the
        source center participate in the fit. Reduces neighbor contamination
        by limiting the effective stamp size. A value of 0 means use the
        full PSF stamp for measurement, while grouped fitting still uses a
        PSF-derived effective influence radius for connectivity (default 0.0).
        Typical values: 2-3 * FWHM.
    damp_snthresh : float, optional
        S/N threshold for position damping. Sources with S/N well above
        this threshold fit positions freely; sources below are pulled
        toward their initial positions. Internally converted to Tikhonov
        regularization strength via ``damp_pos = (damp_snthresh / sigma_psf)^2``
        where ``sigma_psf = fwhm / 2.3548``. A value of 0 disables
        damping (default 0.0). Typical values: 20-50.

    Returns
    -------
    flux : `~numpy.ndarray`
        Fitted flux for each source.
    fluxerr : `~numpy.ndarray`
        Flux error.
    xfit : `~numpy.ndarray`
        Fitted x position (same as input x if ``fit_positions=False``).
    yfit : `~numpy.ndarray`
        Fitted y position.
    flag : `~numpy.ndarray`
        Flags.
    chi2 : `~numpy.ndarray`
        Reduced chi-squared for each fit. In flux-only mode
        (``fit_positions=False``), returned as ``nan``.
    niter : `~numpy.ndarray`
        Number of fitting iterations. In flux-only mode, returned as 0.
    """

    cdef int status
    cdef sep_image im
    cdef double area1
    cdef double xerr1, yerr1, chi2_1
    cdef int niter1
    cdef np.int64_t npts
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] gx, gy
    cdef np.ndarray[np.int32_t, ndim=1, mode="c"] gid
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] gflux, gfluxerr
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] gxfit, gyfit
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] gxerr, gyerr, gchi2
    cdef np.ndarray[np.int32_t, ndim=1, mode="c"] gniter
    cdef np.ndarray[np.int16_t, ndim=1, mode="c"] gflag

    cdef double old_fit_radius
    cdef double old_damp_snthresh

    _parse_arrays(data, err, var, mask, segmap, &im)
    im.maskthresh = maskthresh

    if gain is not None:
        im.gain = gain

    # Temporarily override fit_radius and damp_snthresh on PSF.
    # The parameters always take precedence.
    old_fit_radius = psf.ptr.fit_radius
    psf.ptr.fit_radius = fit_radius
    old_damp_snthresh = psf.ptr.damp_snthresh
    psf.ptr.damp_snthresh = damp_snthresh

    # Coerce coordinate inputs to correct dtypes for safe pointer casts
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if seg_id is None:
        seg_id = np.zeros(1, dtype=np.intc)
    else:
        seg_id = np.asarray(seg_id, dtype=np.intc)

    dt = np.float64
    shape = np.broadcast(x, y).shape

    try:
        if not fit_positions and not grouped:
            # Flux-only ungrouped mode: use sep_sum_psf per source
            oflux = np.empty(shape, dt)
            ofluxerr = np.empty(shape, dt)
            oflag = np.empty(shape, np.short)
            oxfit = np.broadcast_to(x, shape).copy().astype(dt)
            oyfit = np.broadcast_to(y, shape).copy().astype(dt)
            ochi2 = np.full(shape, np.nan, dtype=dt)
            oniter = np.zeros(shape, dtype=np.intc)

            it = np.broadcast(x, y, seg_id, oflux, ofluxerr, oflag)
            while np.PyArray_MultiIter_NOTDONE(it):
                status = sep_sum_psf(
                    &im, psf.ptr,
                    (<double*>np.PyArray_MultiIter_DATA(it, 0))[0],
                    (<double*>np.PyArray_MultiIter_DATA(it, 1))[0],
                    (<int*>np.PyArray_MultiIter_DATA(it, 2))[0],
                    0,
                    <double*>np.PyArray_MultiIter_DATA(it, 3),
                    <double*>np.PyArray_MultiIter_DATA(it, 4),
                    &area1,
                    <short*>np.PyArray_MultiIter_DATA(it, 5))
                _assert_ok(status)
                np.PyArray_MultiIter_NEXT(it)

            return oflux, ofluxerr, oxfit, oyfit, oflag, ochi2, oniter

        if grouped:
            # Grouped path: flatten arrays, call multi
            gx = np.ascontiguousarray(
                np.broadcast_to(x, shape).ravel(), dtype=np.float64)
            gy = np.ascontiguousarray(
                np.broadcast_to(y, shape).ravel(), dtype=np.float64)
            gid = np.ascontiguousarray(
                np.broadcast_to(seg_id, shape).ravel(), dtype=np.intc)

            npts = gx.shape[0]
            gflux = np.empty(npts, dtype=np.float64)
            gfluxerr = np.empty(npts, dtype=np.float64)
            gxfit = np.empty(npts, dtype=np.float64)
            gyfit = np.empty(npts, dtype=np.float64)
            gxerr = np.empty(npts, dtype=np.float64)
            gyerr = np.empty(npts, dtype=np.float64)
            gniter = np.empty(npts, dtype=np.intc)
            gchi2 = np.empty(npts, dtype=np.float64)
            gflag = np.empty(npts, dtype=np.int16)

            status = sep_psf_fit_multi(
                &im, psf.ptr,
                <double*>gx.data,
                <double*>gy.data,
                npts,
                <int*>gid.data,
                group_factor,
                0, maxiter,
                1 if fit_positions else 0,
                <double*>gflux.data,
                <double*>gfluxerr.data,
                <double*>gxfit.data,
                <double*>gyfit.data,
                <double*>gxerr.data,
                <double*>gyerr.data,
                <int*>gniter.data,
                <double*>gchi2.data,
                <short*>gflag.data)
            _assert_ok(status)

            return (np.asarray(gflux).reshape(shape),
                    np.asarray(gfluxerr).reshape(shape),
                    np.asarray(gxfit).reshape(shape),
                    np.asarray(gyfit).reshape(shape),
                    np.asarray(gflag).astype(np.short).reshape(shape),
                    np.asarray(gchi2).reshape(shape),
                    np.asarray(gniter).astype(np.intc).reshape(shape))

        else:
            # Non-grouped: batch call to C loop
            # Use atleast_1d to handle scalar shape (shape=())
            ashape = shape if len(shape) > 0 else (1,)
            gx = np.ascontiguousarray(
                np.broadcast_to(x, ashape).ravel(), dtype=np.float64)
            gy = np.ascontiguousarray(
                np.broadcast_to(y, ashape).ravel(), dtype=np.float64)
            gid = np.ascontiguousarray(
                np.broadcast_to(seg_id, ashape).ravel(), dtype=np.intc)

            npts = gx.shape[0]
            gflux = np.empty(npts, dtype=np.float64)
            gfluxerr = np.empty(npts, dtype=np.float64)
            gxfit = np.empty(npts, dtype=np.float64)
            gyfit = np.empty(npts, dtype=np.float64)
            gxerr = np.empty(npts, dtype=np.float64)
            gyerr = np.empty(npts, dtype=np.float64)
            gniter = np.empty(npts, dtype=np.intc)
            gchi2 = np.empty(npts, dtype=np.float64)
            gflag = np.empty(npts, dtype=np.int16)

            status = sep_psf_fit_array(
                &im, psf.ptr,
                <double*>gx.data,
                <double*>gy.data,
                npts,
                <int*>gid.data,
                0, maxiter,
                <double*>gflux.data,
                <double*>gfluxerr.data,
                <double*>gxfit.data,
                <double*>gyfit.data,
                <double*>gxerr.data,
                <double*>gyerr.data,
                <int*>gniter.data,
                <double*>gchi2.data,
                <short*>gflag.data)
            _assert_ok(status)

            return (np.asarray(gflux).reshape(shape),
                    np.asarray(gfluxerr).reshape(shape),
                    np.asarray(gxfit).reshape(shape),
                    np.asarray(gyfit).reshape(shape),
                    np.asarray(gflag).astype(np.short).reshape(shape),
                    np.asarray(gchi2).reshape(shape),
                    np.asarray(gniter).astype(np.intc).reshape(shape))
    finally:
        psf.ptr.fit_radius = old_fit_radius
        psf.ptr.damp_snthresh = old_damp_snthresh
