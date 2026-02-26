/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 *
 * This file is part of SEP
 *
 * Copyright 1993-2011 Emmanuel Bertin -- IAP/CNRS/UPMC
 * Copyright 2014-2026 SEP developers
 *
 * SEP is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SEP is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with SEP.  If not, see <http://www.gnu.org/licenses/>.
 *
 *%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sep.h"
#include "sepcore.h"

/*--------------------------------------------------------------------------*/
/* Constants for Lanczos sinc interpolation (from SExtractor image.c)       */
/*--------------------------------------------------------------------------*/
#define INTERPW 8
#define INTERPFAC 4.0
#define INTERP_LUT_SIZE 512

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Original Lanczos kernel (used only for LUT construction) */
static float interpf_exact(float x) {
  if (x < 1e-5f && x > -1e-5f) return 1.0f;
  if (x > INTERPFAC || x < -INTERPFAC) return 0.0f;
  return sinf((float)M_PI * x) * sinf((float)M_PI / INTERPFAC * x) /
         ((float)M_PI * (float)M_PI / INTERPFAC * x * x);
}

/* Fast LUT-based Lanczos kernel with linear interpolation */
static float interpf_lut(float x, const float *lut, int lut_size) {
  float ax = fabsf(x);
  if (ax > (float)INTERPFAC) return 0.0f;
  if (ax < 1e-5f) return 1.0f;
  float idx = ax * (float)(lut_size - 1) / (float)INTERPFAC;
  int i0 = (int)idx;
  if (i0 >= lut_size - 1) return lut[lut_size - 1];
  float frac = idx - (float)i0;
  return lut[i0] + frac * (lut[i0 + 1] - lut[i0]);
}

/* Build Lanczos LUT covering [0, INTERPFAC] */
static void build_interp_lut(float *lut, int size) {
  int i;
  for (i = 0; i < size; i++) {
    float x = (float)INTERPFAC * (float)i / (float)(size - 1);
    lut[i] = interpf_exact(x);
  }
}

/*--------------------------------------------------------------------------*/
/* Constants for PSF fitting                                                */
/*--------------------------------------------------------------------------*/
#define PSF_MINSHIFT 1e-3 /* convergence threshold (pixels) */
#define PSF_NA 3          /* parameters per component: flux, dx, dy */

/*--------------------------------------------------------------------------*/
/* SVD solver macros (from SExtractor psf.c, Numerical Recipes)             */
/*--------------------------------------------------------------------------*/
#define SVD_TOL 1.0e-11

static double svd_pythag(double a, double b) {
  double at = fabs(a), bt = fabs(b), ct;
  if (at > bt) {
    ct = bt / at;
    return at * sqrt(1.0 + ct * ct);
  }
  if (bt > 0.0) {
    ct = at / bt;
    return bt * sqrt(1.0 + ct * ct);
  }
  return 0.0;
}

/*--------------------------------------------------------------------------*/
/* Workspace helpers for threaded fitting                                   */
/*--------------------------------------------------------------------------*/

static void psf_workspace_nullify(sep_psf *psf) {
  if (!psf) return;
  psf->loc = NULL;
  psf->resi = NULL;
  psf->interp_mask = NULL;
  psf->interp_nmask = NULL;
  psf->interp_start = NULL;
  psf->interp_buf = NULL;
  psf->fit_mat = NULL;
  psf->fit_dvec = NULL;
  psf->fit_weight = NULL;
  psf->fit_sol = NULL;
  psf->fit_vmat = NULL;
  psf->fit_wmat = NULL;
  psf->fit_covmat = NULL;
  psf->svd_rv1 = NULL;
  psf->svd_tmp = NULL;
}

static void psf_workspace_free(sep_psf *psf) {
  if (!psf) return;
  free(psf->loc);
  free(psf->resi);
  free(psf->interp_mask);
  free(psf->interp_nmask);
  free(psf->interp_start);
  free(psf->interp_buf);
  free(psf->fit_mat);
  free(psf->fit_dvec);
  free(psf->fit_weight);
  free(psf->fit_sol);
  free(psf->fit_vmat);
  free(psf->fit_wmat);
  free(psf->fit_covmat);
  free(psf->svd_rv1);
  free(psf->svd_tmp);
  psf_workspace_nullify(psf);
}

#ifdef _OPENMP
static int psf_workspace_clone(const sep_psf *src, sep_psf *dst) {
  int status = RETURN_OK;
  int npix, rnpix, npar;

  memset(dst, 0, sizeof(*dst));
  *dst = *src;
  psf_workspace_nullify(dst);

  /* Shared read-only model arrays */
  dst->data = src->data;
  dst->interp_lut = src->interp_lut;

  npix = src->w * src->h;
  rnpix = src->rw * src->rh;
  npar = PSF_NA;

  QMALLOC(dst->loc, float, npix, status);
  QMALLOC(dst->resi, float, rnpix, status);
  QMALLOC(dst->interp_mask, float, src->interp_mask_len, status);
  QMALLOC(dst->interp_nmask, int, src->interp_nmask_len, status);
  QMALLOC(dst->interp_start, int, src->interp_nmask_len, status);
  QMALLOC(dst->interp_buf, float, src->interp_buf_len, status);
  QMALLOC(dst->fit_mat, double, rnpix * npar, status);
  QMALLOC(dst->fit_dvec, double, rnpix, status);
  QMALLOC(dst->fit_weight, double, rnpix, status);
  QMALLOC(dst->fit_sol, double, npar, status);
  QMALLOC(dst->fit_vmat, double, npar * npar, status);
  QMALLOC(dst->fit_wmat, double, npar, status);
  QMALLOC(dst->fit_covmat, double, npar * npar, status);
  QMALLOC(dst->svd_rv1, double, npar, status);
  QMALLOC(dst->svd_tmp, double, npar, status);

  return RETURN_OK;

exit:
  psf_workspace_free(dst);
  return status;
}
#endif

/*==========================================================================*/
/*                         PSF Lifecycle                                    */
/*==========================================================================*/

int sep_psf_create(sep_psf **out, const float *data, int w, int h, int ncomp,
                   int degree, double x0, double y0, double sx, double sy,
                   float pixstep, double fwhm) {
  sep_psf *psf = NULL;
  int status = RETURN_OK;
  int npix, datalen, oversamp, rw, rh, rnpix;
  int maxdim, mask_len, nmask_len, buf_len, npar;

  if (w <= 0 || h <= 0 || ncomp <= 0 || pixstep <= 0.0f) {
    return ILLEGAL_APER_PARAMS;
  }

  npix = w * h;
  datalen = ncomp * npix;

  QCALLOC(psf, sep_psf, 1, status);

  psf->w = w;
  psf->h = h;
  psf->ncomp = ncomp;
  psf->degree = degree;
  psf->x0 = x0;
  psf->y0 = y0;
  psf->sx = sx;
  psf->sy = sy;
  psf->pixstep = pixstep;
  psf->fwhm = fwhm;

  /* Compute native-resolution stamp dimensions */
  oversamp = (int)(1.0f / pixstep + 0.5f);
  if (oversamp < 1) oversamp = 1;
  rw = w / oversamp;
  rh = h / oversamp;
  if (rw < 1) rw = 1;
  if (rh < 1) rh = 1;
  psf->rw = rw;
  psf->rh = rh;
  rnpix = rw * rh;

  QMALLOC(psf->data, float, datalen, status);
  memcpy(psf->data, data, (size_t)datalen * sizeof(float));

  QCALLOC(psf->loc, float, npix, status);
  QCALLOC(psf->resi, float, rnpix, status);

  /* Pre-allocate resampling workspace.
   * mask needs max(rw, rh) * INTERPW floats (used for both x and y passes).
   * nmask/start need max(rw, rh) ints each.
   * buf needs rw * h floats (intermediate x-resampled buffer). */
  maxdim = rw > rh ? rw : rh;
  mask_len = maxdim * INTERPW;
  nmask_len = maxdim;
  buf_len = rw * h;

  QMALLOC(psf->interp_mask, float, mask_len, status);
  QMALLOC(psf->interp_nmask, int, nmask_len, status);
  QMALLOC(psf->interp_start, int, nmask_len, status);
  QCALLOC(psf->interp_buf, float, buf_len, status);
  psf->interp_mask_len = mask_len;
  psf->interp_nmask_len = nmask_len;
  psf->interp_buf_len = buf_len;

  /* Pre-compute Lanczos interpolation LUT */
  psf->interp_lut_size = INTERP_LUT_SIZE;
  QMALLOC(psf->interp_lut, float, INTERP_LUT_SIZE, status);
  build_interp_lut(psf->interp_lut, INTERP_LUT_SIZE);

  /* Pre-allocate fitting workspace */
  npar = PSF_NA;
  QMALLOC(psf->fit_mat, double, rnpix * npar, status);
  QMALLOC(psf->fit_dvec, double, rnpix, status);
  QMALLOC(psf->fit_weight, double, rnpix, status);
  QMALLOC(psf->fit_sol, double, npar, status);
  QMALLOC(psf->fit_vmat, double, npar * npar, status);
  QMALLOC(psf->fit_wmat, double, npar, status);
  QCALLOC(psf->fit_covmat, double, npar * npar, status);
  QMALLOC(psf->svd_rv1, double, npar, status);
  QMALLOC(psf->svd_tmp, double, npar, status);

  *out = psf;
  return RETURN_OK;

exit:
  sep_psf_free(psf);
  *out = NULL;
  return status;
}

void sep_psf_free(sep_psf *psf) {
  if (psf) {
    free(psf->data);
    free(psf->interp_lut);
    psf_workspace_free(psf);
  }
  free(psf);
}

/*==========================================================================*/
/*                     Polynomial PSF Evaluation                            */
/*==========================================================================*/

int sep_psf_build(sep_psf *psf, double x, double y) {
  double dx, dy, coeff;
  const float *comp;
  float *loc;
  float fcoeff;
  int npix, i, i1, i2, p;

  npix = psf->w * psf->h;
  loc = psf->loc;
  memset(loc, 0, (size_t)npix * sizeof(float));

  dx = (psf->sx != 0.0) ? (x - psf->x0) / psf->sx : 0.0;
  dy = (psf->sy != 0.0) ? (y - psf->y0) / psf->sy : 0.0;

  i = 0;
  for (i2 = 0; i2 <= psf->degree; i2++) {
    for (i1 = 0; i1 <= psf->degree - i2; i1++) {
      if (i >= psf->ncomp) break;
      coeff = pow(dx, i1) * pow(dy, i2);
      comp = psf->data + (size_t)i * npix;
      fcoeff = (float)coeff;
#if defined(_OPENMP)
#pragma omp simd
#endif
      for (p = 0; p < npix; p++) loc[p] += fcoeff * comp[p];
      i++;
    }
  }

  return RETURN_OK;
}

/*==========================================================================*/
/*         Lanczos Sinc Resampling (ported from SExtractor image.c)         */
/*==========================================================================*/

int sep_psf_resample(sep_psf *psf, double dx, double dy) {
  float *pix1, *pix2;
  float *mask, *maskt, *pix12;
  float *pixin, *pixin0, *pixout, *pixout0;
  int *start, *startt, *nmask, *nmaskt;
  const float *lut;
  int lut_size;
  int w1, h1, w2, h2;
  int i, j, k, n, t, ix, ix1, iy, iy1;
  int ixs2, iys2, dix2, diy2, nx2, ny2, iys1a, ny1, hmw, hmh;
  float xc1, xc2, yc1, yc2, xs1, ys1, x1, y1, x, y, dxm, dym;
  float val, norm, step2;

  pix1 = psf->loc;
  pix2 = psf->resi;
  w1 = psf->w;
  h1 = psf->h;
  w2 = psf->rw;
  h2 = psf->rh;
  step2 = 1.0f / psf->pixstep; /* PSF pixels per image pixel */

  /* Use pre-allocated workspace */
  mask = psf->interp_mask;
  nmask = psf->interp_nmask;
  start = psf->interp_start;
  pix12 = psf->interp_buf;
  lut = psf->interp_lut;
  lut_size = psf->interp_lut_size;

  memset(pix2, 0, (size_t)w2 * h2 * sizeof(float));

  /* Fast path for pixstep == 1.0 with zero shift: direct copy */
  if (psf->pixstep == 1.0f && fabs(dx) < 1e-7 && fabs(dy) < 1e-7) {
    int cw = w1 < w2 ? w1 : w2;
    int ch = h1 < h2 ? h1 : h2;
    int ox1 = w1 / 2 - cw / 2, oy1 = h1 / 2 - ch / 2;
    int ox2 = w2 / 2 - cw / 2, oy2 = h2 / 2 - ch / 2;
    for (j = 0; j < ch; j++) {
      memcpy(pix2 + (oy2 + j) * w2 + ox2,
             pix1 + (oy1 + j) * w1 + ox1,
             (size_t)cw * sizeof(float));
    }
    return RETURN_OK;
  }

  /* Compute coordinate mappings.
   * dx, dy: source offset from stamp center in image pixels.
   * A positive dx means "source is to the right of center", so
   * we shift the sampling window LEFT in the supersampled input
   * to make the PSF appear shifted RIGHT in the output. */
  xc1 = (float)(w1 / 2);
  xc2 = (float)(w2 / 2);
  xs1 = xc1 - (float)dx * step2 - xc2 * step2;

  if ((int)xs1 >= w1) return RETURN_OK;
  ixs2 = 0;
  if (xs1 < 0.0f) {
    dix2 = (int)(1 - xs1 / step2);
    if (dix2 >= w2) return RETURN_OK;
    ixs2 += dix2;
    xs1 += dix2 * step2;
  }
  nx2 = (int)((w1 - 1 - xs1) / step2 + 1);
  if (nx2 > (w2 - ixs2)) nx2 = w2 - ixs2;
  if (nx2 <= 0) return RETURN_OK;

  yc1 = (float)(h1 / 2);
  yc2 = (float)(h2 / 2);
  ys1 = yc1 - (float)dy * step2 - yc2 * step2;

  if ((int)ys1 >= h1) return RETURN_OK;
  iys2 = 0;
  if (ys1 < 0.0f) {
    diy2 = (int)(1 - ys1 / step2);
    if (diy2 >= h2) return RETURN_OK;
    iys2 += diy2;
    ys1 += diy2 * step2;
  }
  ny2 = (int)((h1 - 1 - ys1) / step2 + 1);
  if (ny2 > (h2 - iys2)) ny2 = h2 - iys2;
  if (ny2 <= 0) return RETURN_OK;

  /* Set y-range for x-resampling with interpolation margin */
  iys1a = (int)ys1;
  hmh = INTERPW / 2 - 1;
  if (iys1a < 0 || ((iys1a -= hmh) < 0)) iys1a = 0;
  ny1 = (int)(ys1 + ny2 * step2) + INTERPW - hmh;
  if (ny1 > h1) ny1 = h1;
  ny1 -= iys1a;
  ys1 -= (float)iys1a;

  /* Compute x interpolation kernels (using LUT) */
  hmw = INTERPW / 2 - 1;
  x1 = xs1;
  maskt = mask;
  nmaskt = nmask;
  startt = start;
  for (j = nx2; j--; x1 += step2) {
    ix = (ix1 = (int)x1) - hmw;
    dxm = ix1 - x1 - hmw;
    if (ix < 0) {
      n = INTERPW + ix;
      dxm -= (float)ix;
      ix = 0;
    } else {
      n = INTERPW;
    }
    if (n > (t = w1 - ix)) n = t;
    *(startt++) = ix;
    *(nmaskt++) = n;
    norm = 0.0f;
    for (x = dxm, i = n; i--; x += 1.0f) {
      norm += (*(maskt++) = interpf_lut(x, lut, lut_size));
    }
    norm = norm > 0.0f ? 1.0f / norm : 1.0f;
    maskt -= n;
    for (i = n; i--;) {
      *(maskt++) *= norm;
    }
  }

  /* Zero intermediate buffer */
  memset(pix12, 0, (size_t)nx2 * ny1 * sizeof(float));

  /* Interpolation in x (with transposition) */
  pixin0 = pix1 + iys1a * w1;
  pixout0 = pix12;
  for (k = ny1; k--; pixin0 += w1, pixout0++) {
    maskt = mask;
    nmaskt = nmask;
    startt = start;
    pixout = pixout0;
    for (j = nx2; j--; pixout += ny1) {
      pixin = pixin0 + *(startt++);
      n = *(nmaskt++);
      val = 0.0f;
#if defined(_OPENMP)
#pragma omp simd reduction(+ : val)
#endif
      for (i = 0; i < n; i++) val += maskt[i] * pixin[i];
      maskt += n;
      *pixout = val;
    }
  }

  /* Compute y interpolation kernels (reuse mask/nmask/start buffers) */
  hmh = INTERPW / 2 - 1;
  y1 = ys1;
  maskt = mask;
  nmaskt = nmask;
  startt = start;
  for (j = ny2; j--; y1 += step2) {
    iy = (iy1 = (int)y1) - hmh;
    dym = iy1 - y1 - hmh;
    if (iy < 0) {
      n = INTERPW + iy;
      dym -= (float)iy;
      iy = 0;
    } else {
      n = INTERPW;
    }
    if (n > (t = ny1 - iy)) n = t;
    *(startt++) = iy;
    *(nmaskt++) = n;
    norm = 0.0f;
    for (y = dym, i = n; i--; y += 1.0f) {
      norm += (*(maskt++) = interpf_lut(y, lut, lut_size));
    }
    norm = norm > 0.0f ? 1.0f / norm : 1.0f;
    maskt -= n;
    for (i = n; i--;) {
      *(maskt++) *= norm;
    }
  }

  /* Interpolation in y and transpose back */
  pixin0 = pix12;
  pixout0 = pix2 + ixs2 + iys2 * w2;
  for (k = nx2; k--; pixin0 += ny1, pixout0++) {
    maskt = mask;
    nmaskt = nmask;
    startt = start;
    pixout = pixout0;
    for (j = ny2; j--; pixout += w2) {
      pixin = pixin0 + *(startt++);
      n = *(nmaskt++);
      val = 0.0f;
#if defined(_OPENMP)
#pragma omp simd reduction(+ : val)
#endif
      for (i = 0; i < n; i++) val += maskt[i] * pixin[i];
      maskt += n;
      *pixout = val;
    }
  }

  return RETURN_OK;
}

/*==========================================================================*/
/*         SVD Solver (ported from SExtractor psf.c)                        */
/*==========================================================================*/

/*
 * General least-square fit A.x = b via Singular Value Decomposition.
 * Adapted from Numerical Recipes in C, 2nd Ed. (p. 671).
 * Note: the a and v matrices are transposed w.r.t. N.R. convention.
 *
 * Returns 0 on success, non-zero on error.
 */
static int svdfit(double *a, double *b, int m, int n, double *sol,
                  double *vmat, double *wmat, double *rv1, double *tmp) {
  double *ap, *ap0, *ap1, *ap10, *rv1p, *vp, *vp0, *vp1, *vp10, *tmpp, *bp,
      *w;
  double c, f, g, h, s, x, y, z;
  double anorm, scale, thresh, wmax;
  double maxarg1, maxarg2;
  int flag, i, its, j, jj, k, l, nm, mmi, nml;

#define SVD_MAX(a, b) \
  (maxarg1 = (a), maxarg2 = (b), (maxarg1) > (maxarg2) ? (maxarg1) : (maxarg2))
#define SVD_SIGN(a, b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

  if (m < n) return ILLEGAL_APER_PARAMS;

  anorm = g = scale = 0.0;
  l = nm = nml = 0;

  /* Householder reduction to bidiagonal form */
  for (i = 0; i < n; i++) {
    l = i + 1;
    nml = n - l;
    rv1[i] = scale * g;
    g = s = scale = 0.0;
    if ((mmi = m - i) > 0) {
      ap = ap0 = a + i * (m + 1);
      for (k = mmi; k--;) scale += fabs(*(ap++));
      if (scale) {
        for (ap = ap0, k = mmi; k--; ap++) {
          *ap /= scale;
          s += *ap * *ap;
        }
        f = *ap0;
        g = -SVD_SIGN(sqrt(s), f);
        h = f * g - s;
        *ap0 = f - g;
        ap10 = a + l * m + i;
        for (j = nml; j--; ap10 += m) {
          for (s = 0.0, ap = ap0, ap1 = ap10, k = mmi; k--;)
            s += *(ap1++) * *(ap++);
          f = s / h;
          for (ap = ap0, ap1 = ap10, k = mmi; k--;)
            *(ap1++) += f * *(ap++);
        }
        for (ap = ap0, k = mmi; k--;) *(ap++) *= scale;
      }
    }
    wmat[i] = scale * g;
    g = s = scale = 0.0;
    if (i < m && i + 1 != n) {
      ap = ap0 = a + i + m * l;
      for (k = nml; k--; ap += m) scale += fabs(*ap);
      if (scale) {
        for (ap = ap0, k = nml; k--; ap += m) {
          *ap /= scale;
          s += *ap * *ap;
        }
        f = *ap0;
        g = -SVD_SIGN(sqrt(s), f);
        h = f * g - s;
        *ap0 = f - g;
        rv1p = rv1 + l;
        for (ap = ap0, k = nml; k--; ap += m) *(rv1p++) = *ap / h;
        ap10 = a + l + m * l;
        for (j = m - l; j--; ap10++) {
          for (s = 0.0, ap = ap0, ap1 = ap10, k = nml; k--; ap += m, ap1 += m)
            s += *ap1 * *ap;
          rv1p = rv1 + l;
          for (ap1 = ap10, k = nml; k--; ap1 += m) *ap1 += s * *(rv1p++);
        }
        for (ap = ap0, k = nml; k--; ap += m) *ap *= scale;
      }
    }
    anorm =
        SVD_MAX(anorm, (fabs(wmat[i]) + fabs(rv1[i])));
  }

  /* Accumulation of right-hand transformations */
  for (i = n - 1; i >= 0; i--) {
    if (i < n - 1) {
      if (g) {
        ap0 = a + l * m + i;
        vp0 = vmat + i * n + l;
        vp10 = vmat + l * n + l;
        g *= *ap0;
        for (ap = ap0, vp = vp0, j = nml; j--; ap += m) *(vp++) = *ap / g;
        for (j = nml; j--; vp10 += n) {
          for (s = 0.0, ap = ap0, vp1 = vp10, k = nml; k--; ap += m)
            s += *ap * *(vp1++);
          for (vp = vp0, vp1 = vp10, k = nml; k--;) *(vp1++) += s * *(vp++);
        }
      }
      vp = vmat + l * n + i;
      vp1 = vmat + i * n + l;
      for (j = nml; j--; vp += n) *vp = *(vp1++) = 0.0;
    }
    vmat[i * n + i] = 1.0;
    g = rv1[i];
    l = i;
    nml = n - l;
  }

  /* Accumulation of left-hand transformations */
  for (i = (m < n ? m : n); --i >= 0;) {
    l = i + 1;
    nml = n - l;
    mmi = m - i;
    g = wmat[i];
    ap0 = a + i * m + i;
    ap10 = ap0 + m;
    for (ap = ap10, j = nml; j--; ap += m) *ap = 0.0;
    if (g) {
      g = 1.0 / g;
      for (j = nml; j--; ap10 += m) {
        for (s = 0.0, ap = ap0, ap1 = ap10, k = mmi; --k;)
          s += *(++ap) * *(++ap1);
        f = (s / (*ap0)) * g;
        for (ap = ap0, ap1 = ap10, k = mmi; k--;) *(ap1++) += f * *(ap++);
      }
      for (ap = ap0, j = mmi; j--;) *(ap++) *= g;
    } else {
      for (ap = ap0, j = mmi; j--;) *(ap++) = 0.0;
    }
    ++(*ap0);
  }

  /* Diagonalization of bidiagonal form */
  for (k = n; --k >= 0;) {
    for (its = 0; its < 100; its++) {
      flag = 1;
      for (l = k; l >= 0; l--) {
        nm = l - 1;
        if (fabs(rv1[l]) + anorm == anorm) {
          flag = 0;
          break;
        }
        if (fabs(wmat[nm]) + anorm == anorm) break;
      }
      if (flag) {
        c = 0.0;
        s = 1.0;
        ap0 = a + nm * m;
        ap10 = a + l * m;
        for (i = l; i <= k; i++, ap10 += m) {
          f = s * rv1[i];
          if (fabs(f) + anorm == anorm) break;
          g = wmat[i];
          h = svd_pythag(f, g);
          wmat[i] = h;
          h = 1.0 / h;
          c = g * h;
          s = (-f * h);
          for (ap = ap0, ap1 = ap10, j = m; j--;) {
            z = *ap1;
            y = *ap;
            *(ap++) = y * c + z * s;
            *(ap1++) = z * c - y * s;
          }
        }
      }
      z = wmat[k];
      if (l == k) {
        if (z < 0.0) {
          wmat[k] = -z;
          vp = vmat + k * n;
          for (j = n; j--; vp++) *vp = (-*vp);
        }
        break;
      }
      if (its == 99) {
        /* No convergence */
        return ILLEGAL_APER_PARAMS;
      }
      x = wmat[l];
      nm = k - 1;
      y = wmat[nm];
      g = rv1[nm];
      h = rv1[k];
      f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
      g = svd_pythag(f, 1.0);
      f = ((x - z) * (x + z) + h * ((y / (f + SVD_SIGN(g, f))) - h)) / x;
      c = s = 1.0;
      ap10 = a + l * m;
      vp10 = vmat + l * n;
      for (j = l; j <= nm; j++, ap10 += m, vp10 += n) {
        i = j + 1;
        g = rv1[i];
        y = wmat[i];
        h = s * g;
        g = c * g;
        z = svd_pythag(f, h);
        rv1[j] = z;
        c = f / z;
        s = h / z;
        f = x * c + g * s;
        g = g * c - x * s;
        h = y * s;
        y = y * c;
        for (vp = (vp1 = vp10) + n, jj = n; jj--;) {
          z = *vp;
          x = *vp1;
          *(vp1++) = x * c + z * s;
          *(vp++) = z * c - x * s;
        }
        z = svd_pythag(f, h);
        wmat[j] = z;
        if (z) {
          z = 1.0 / z;
          c = f * z;
          s = h * z;
        }
        f = c * g + s * y;
        x = c * y - s * g;
        for (ap = (ap1 = ap10) + m, jj = m; jj--;) {
          z = *ap;
          y = *ap1;
          *(ap1++) = y * c + z * s;
          *(ap++) = z * c - y * s;
        }
      }
      rv1[l] = 0.0;
      rv1[k] = f;
      wmat[k] = x;
    }
  }

  /* Threshold small singular values */
  wmax = 0.0;
  w = wmat;
  for (j = n; j--; w++)
    if (*w > wmax) wmax = *w;
  thresh = SVD_TOL * wmax;
  w = wmat;
  for (j = n; j--; w++)
    if (*w < thresh) *w = 0.0;

  /* Back-substitution: compute solution */
  w = wmat;
  ap = a;
  tmpp = tmp;
  for (j = n; j--; w++) {
    s = 0.0;
    if (*w) {
      bp = b;
      for (i = m; i--;) s += *(ap++) * *(bp++);
      s /= *w;
    } else {
      ap += m;
    }
    *(tmpp++) = s;
  }

  vp0 = vmat;
  for (j = 0; j < n; j++, vp0++) {
    s = 0.0;
    tmpp = tmp;
    for (vp = vp0, jj = n; jj--; vp += n) s += *vp * *(tmpp++);
    sol[j] = s;
  }

  return RETURN_OK;

#undef SVD_MAX
#undef SVD_SIGN
}

/*
 * Compute covariance matrix from SVD V and W matrices.
 * Adapted from Numerical Recipes in C, 2nd Ed. (p. 679).
 */
static int svdvar(double *v, double *w, int n, double *cov) {
  double wti_stack[PSF_NA * 16]; /* stack for typical group sizes */
  double *wti;
  double sum;
  int i, j, k;
  int heap = 0;

  if (n <= PSF_NA * 16) {
    wti = wti_stack;
  } else {
    wti = (double *)malloc((size_t)n * sizeof(double));
    if (!wti) return MEMORY_ALLOC_ERROR;
    heap = 1;
  }

  for (i = 0; i < n; i++) wti[i] = w[i] ? 1.0 / (w[i] * w[i]) : 0.0;

  for (i = 0; i < n; i++) {
    for (j = 0; j <= i; j++) {
      for (sum = 0.0, k = 0; k < n; k++)
        sum += v[k * n + i] * v[k * n + j] * wti[k];
      cov[j * n + i] = cov[i * n + j] = sum;
    }
  }

  if (heap) free(wti);
  return RETURN_OK;
}

/* Build normal equations for least squares with column-major A (m rows, n cols):
 *   ata = A^T A, atb = A^T b
 */
static void psf_build_normal_eq(const double *a, const double *b, int m, int n,
                                double *ata, double *atb) {
  int i, j, p;

  for (i = 0; i < n; i++) {
    const double *ai = a + (size_t)i * m;
    double rhs = 0.0;
#if defined(_OPENMP)
#pragma omp simd reduction(+ : rhs)
#endif
    for (p = 0; p < m; p++) rhs += ai[p] * b[p];
    atb[i] = rhs;

    for (j = 0; j <= i; j++) {
      const double *aj = a + (size_t)j * m;
      double sum = 0.0;
#if defined(_OPENMP)
#pragma omp simd reduction(+ : sum)
#endif
      for (p = 0; p < m; p++) sum += ai[p] * aj[p];
      ata[i * n + j] = sum;
      ata[j * n + i] = sum;
    }
  }
}

/* Cholesky factorization: A = L * L^T (in-place in lower triangle). */
static int psf_cholesky_factor(double *a, int n) {
  int i, j, k;

  for (i = 0; i < n; i++) {
    for (j = 0; j <= i; j++) {
      double sum = a[i * n + j];
      for (k = 0; k < j; k++) sum -= a[i * n + k] * a[j * n + k];

      if (i == j) {
        if (!(sum > 0.0) || !isfinite(sum)) return ILLEGAL_APER_PARAMS;
        a[i * n + i] = sqrt(sum);
      } else {
        a[i * n + j] = sum / a[j * n + j];
      }
    }

    for (j = i + 1; j < n; j++) a[i * n + j] = 0.0;
  }

  return RETURN_OK;
}

/* Solve (L * L^T) x = rhs for x, with L from psf_cholesky_factor().
 * rhs is overwritten with the forward-solve intermediate vector. */
static void psf_cholesky_solve(const double *l, int n, double *rhs, double *x) {
  int i, k;

  for (i = 0; i < n; i++) {
    double sum = rhs[i];
    for (k = 0; k < i; k++) sum -= l[i * n + k] * rhs[k];
    rhs[i] = sum / l[i * n + i];
  }

  for (i = n - 1; i >= 0; i--) {
    double sum = rhs[i];
    for (k = i + 1; k < n; k++) sum -= l[k * n + i] * x[k];
    x[i] = sum / l[i * n + i];
  }
}

/* Invert SPD matrix from Cholesky factor L (L * L^T = A). */
static int psf_cholesky_inverse(const double *l, int n, double *inv,
                                double *rhs, double *x) {
  int i, j, k;

  for (j = 0; j < n; j++) {
    for (i = 0; i < n; i++) rhs[i] = (i == j) ? 1.0 : 0.0;

    for (i = 0; i < n; i++) {
      double sum = rhs[i];
      for (k = 0; k < i; k++) sum -= l[i * n + k] * rhs[k];
      rhs[i] = sum / l[i * n + i];
    }

    for (i = n - 1; i >= 0; i--) {
      double sum = rhs[i];
      for (k = i + 1; k < n; k++) sum -= l[k * n + i] * x[k];
      x[i] = sum / l[i * n + i];
    }

    for (i = 0; i < n; i++) inv[i * n + j] = x[i];
  }

  for (i = 0; i < n; i++) {
    for (j = 0; j < i; j++) {
      double v = 0.5 * (inv[i * n + j] + inv[j * n + i]);
      inv[i * n + j] = v;
      inv[j * n + i] = v;
    }
  }

  return RETURN_OK;
}

/*==========================================================================*/
/*              PSF Flux Photometry (fixed position)                        */
/*==========================================================================*/

int sep_sum_psf(const sep_image *im, sep_psf *psf, double x, double y, int id,
                short inflag, double *sum, double *sumerr, double *area,
                short *flag) {
  (void)inflag;
  int ix0, iy0; /* integer center of stamp on image */
  int sx, sy;   /* stamp coords */
  int64_t imx, imy; /* image coords */
  int64_t pos;
  int status = RETURN_OK;
  double num, den, totarea, maskarea, pix, varpix, psfw, psfval;
  converter convert, econvert, mconvert, sconvert;
  int64_t size, esize, msize, ssize;
  const void *datat, *errort, *maskt, *segt;
  int errisarray, errisstd;

  *flag = 0;
  *sum = 0.0;
  *sumerr = 0.0;
  *area = 0.0;
  num = den = totarea = maskarea = 0.0;
  datat = maskt = segt = NULL;
  errort = im->noise;
  varpix = 1.0;
  size = esize = msize = ssize = 0;

  /* Build and resample PSF at source position */
  status = sep_psf_build(psf, x, y);
  if (status != RETURN_OK) return status;

  ix0 = (int)(x + 0.5);
  iy0 = (int)(y + 0.5);

  status = sep_psf_resample(psf, x - ix0, y - iy0);
  if (status != RETURN_OK) return status;

  /* Normalize PSF stamp */
  psfval = 0.0;
  for (sx = 0; sx < psf->rw * psf->rh; sx++) psfval += psf->resi[sx];
  if (psfval > 0.0) {
    for (sx = 0; sx < psf->rw * psf->rh; sx++)
      psf->resi[sx] /= (float)psfval;
  }

  /* Get converters */
  if ((status = get_converter(im->dtype, &convert, &size))) return status;
  if (im->mask && (status = get_converter(im->mdtype, &mconvert, &msize)))
    return status;
  if (im->segmap && (status = get_converter(im->sdtype, &sconvert, &ssize)))
    return status;

  errisarray = 0;
  errisstd = 0;
  if (im->noise_type != SEP_NOISE_NONE) {
    errisstd = (im->noise_type == SEP_NOISE_STDDEV);
    if (im->noise) {
      errisarray = 1;
      if ((status = get_converter(im->ndtype, &econvert, &esize)))
        return status;
    } else {
      varpix = (errisstd) ? im->noiseval * im->noiseval : im->noiseval;
    }
  }

  /* Iterate over PSF stamp pixels */
  for (sy = 0; sy < psf->rh; sy++) {
    imy = iy0 - psf->rh / 2 + sy;
    if (imy < 0 || imy >= im->h) {
      *flag |= SEP_APER_TRUNC;
      continue;
    }

    for (sx = 0; sx < psf->rw; sx++) {
      imx = ix0 - psf->rw / 2 + sx;
      if (imx < 0 || imx >= im->w) {
        *flag |= SEP_APER_TRUNC;
        continue;
      }

      psfw = psf->resi[sy * psf->rw + sx];
      if (psfw == 0.0) continue;

      pos = imy * im->w + imx;
      datat = (const char *)im->data + pos * size;

      /* Check mask */
      if (im->mask) {
        maskt = (const char *)im->mask + pos * msize;
        if (mconvert(maskt) > im->maskthresh) {
          *flag |= SEP_APER_HASMASKED;
          maskarea += 1.0;
          totarea += 1.0;
          continue;
        }
      }

      /* Check segmap */
      if (im->segmap) {
        segt = (const char *)im->segmap + pos * ssize;
        int segval = (int)sconvert(segt);
        if (segval != 0 && segval != id) {
          *flag |= SEP_APER_HASMASKED;
          maskarea += 1.0;
          totarea += 1.0;
          continue;
        }
      }

      pix = convert(datat);

      /* Get variance */
      if (errisarray) {
        errort = (const char *)im->noise + pos * esize;
        varpix = econvert(errort);
        if (errisstd) varpix *= varpix;
      }

      if (varpix > 0.0) {
        double total_var = varpix;
        if (im->gain > 0.0 && pix > 0.0) {
          total_var += pix / im->gain;
        }
        num += psfw * pix / total_var;
        den += psfw * psfw / total_var;
      } else {
        *flag |= SEP_APER_HASMASKED;
        maskarea += 1.0;
      }
      totarea += 1.0;
    }
  }

  if (totarea > 0.0 && maskarea >= totarea) {
    *flag |= SEP_APER_ALLMASKED;
    return status;
  }

  if (den <= 0.0) {
    *flag |= SEP_APER_ALLMASKED;
    return status;
  }

  *sum = num / den;
  {
    double var = 1.0 / den;
    if (im->gain > 0.0 && *sum > 0.0) {
      var += (*sum) / im->gain;
    }
    *sumerr = sqrt(var);
  }
  *area = totarea;

  return status;
}

/*==========================================================================*/
/*           Design Matrix Construction (ported from SExtractor)            */
/*==========================================================================*/

/*
 * Build design matrix columns for one PSF component:
 *   Column 0: weighted PSF values
 *   Column 1: weighted PSF x-derivative / 2
 *   Column 2: weighted PSF y-derivative / 2
 * Outer ring of stamp is zeroed to avoid edge effects in derivatives.
 */
static void compute_gradient(const float *psfstamp, const double *weight,
                             int width, int height, double *mat) {
  int x, y, idx, row, npix;
  int top, bottom;
  double *col0, *col1, *col2;

  npix = width * height;
  col0 = mat;
  col1 = mat + npix;
  col2 = mat + 2 * npix;

  if (width <= 2 || height <= 2) {
    memset(mat, 0, (size_t)npix * PSF_NA * sizeof(double));
    return;
  }

  /* Zero only the border; interior values are written explicitly. */
  top = 0;
  bottom = (height - 1) * width;
  for (x = 0; x < width; x++) {
    col0[top + x] = 0.0;
    col1[top + x] = 0.0;
    col2[top + x] = 0.0;
    col0[bottom + x] = 0.0;
    col1[bottom + x] = 0.0;
    col2[bottom + x] = 0.0;
  }

  for (y = 1; y < height - 1; y++) {
    row = y * width;
    col0[row] = 0.0;
    col1[row] = 0.0;
    col2[row] = 0.0;
#if defined(_OPENMP)
#pragma omp simd
#endif
    for (x = 1; x < width - 1; x++) {
      double w;
      idx = row + x;
      w = weight[idx];
      col0[idx] = psfstamp[idx] * w;
      col1[idx] = (psfstamp[idx + 1] - psfstamp[idx - 1]) * w * 0.5;
      col2[idx] = (psfstamp[idx + width] - psfstamp[idx - width]) * w * 0.5;
    }
    idx = row + width - 1;
    col0[idx] = 0.0;
    col1[idx] = 0.0;
    col2[idx] = 0.0;
  }
}

/*==========================================================================*/
/*                Iterative PSF Fitting (single source)                     */
/*==========================================================================*/

int sep_psf_fit(const sep_image *im, sep_psf *psf, double x, double y, int id,
                short inflag, int maxiter, double *flux, double *fluxerr,
                double *xfit, double *yfit, double *xerr, double *yerr,
                int *niter, double *chi2, short *flag) {
  (void)inflag;
  int status = RETURN_OK;
  int width, height, npix, npar;
  int ix0, iy0;
  int64_t imx, imy, pos;
  int sx, sy, iter;
  int convflag;
  double deltax, deltay;
  double *mat, *data_vec, *weight;
  double *sol, *vmat, *wmat, *covmat;
  double *rv1, *tmp;
  double pix, varpix, dx_update, dy_update;
  double radmax2;
  converter convert, econvert, mconvert, sconvert;
  int64_t size, esize, msize, ssize;
  const void *datat, *errort, *maskt, *segt;
  int errisarray, errisstd;

  *flag = 0;
  *flux = 0.0;
  *fluxerr = 0.0;
  *xfit = x;
  *yfit = y;
  *xerr = 0.0;
  *yerr = 0.0;
  *niter = 0;
  *chi2 = 0.0;

  width = psf->rw;
  height = psf->rh;
  npix = width * height;
  npar = PSF_NA; /* flux, dx, dy */

  if (npix < npar) return ILLEGAL_APER_PARAMS;
  if (maxiter <= 0) maxiter = 20;

  radmax2 = (double)(width / 2) * (width / 2);

  /* Use pre-allocated workspace */
  mat = psf->fit_mat;
  data_vec = psf->fit_dvec;
  weight = psf->fit_weight;
  sol = psf->fit_sol;
  vmat = psf->fit_vmat;
  wmat = psf->fit_wmat;
  covmat = psf->fit_covmat;
  rv1 = psf->svd_rv1;
  tmp = psf->svd_tmp;

  /* Get converters */
  if ((status = get_converter(im->dtype, &convert, &size))) return status;
  if (im->mask && (status = get_converter(im->mdtype, &mconvert, &msize)))
    return status;
  if (im->segmap && (status = get_converter(im->sdtype, &sconvert, &ssize)))
    return status;

  errisarray = 0;
  errisstd = 0;
  varpix = 1.0;
  if (im->noise_type != SEP_NOISE_NONE) {
    errisstd = (im->noise_type == SEP_NOISE_STDDEV);
    if (im->noise) {
      errisarray = 1;
      if ((status = get_converter(im->ndtype, &econvert, &esize)))
        return status;
    } else {
      varpix = (errisstd) ? im->noiseval * im->noiseval : im->noiseval;
    }
  }

  deltax = 0.0;
  deltay = 0.0;

  /* Extract data stamp and compute weights (once, at initial position) */
  ix0 = (int)(x + 0.5);
  iy0 = (int)(y + 0.5);

  for (sy = 0; sy < height; sy++) {
    imy = iy0 - height / 2 + sy;
    for (sx = 0; sx < width; sx++) {
      imx = ix0 - width / 2 + sx;
      int pidx = sy * width + sx;

      if (imy < 0 || imy >= im->h || imx < 0 || imx >= im->w) {
        data_vec[pidx] = 0.0;
        weight[pidx] = 0.0;
        *flag |= SEP_APER_TRUNC;
        continue;
      }

      pos = imy * im->w + imx;
      datat = (const char *)im->data + pos * size;
      pix = convert(datat);

      /* Check mask */
      if (im->mask) {
        maskt = (const char *)im->mask + pos * msize;
        if (mconvert(maskt) > im->maskthresh) {
          data_vec[pidx] = 0.0;
          weight[pidx] = 0.0;
          *flag |= SEP_APER_HASMASKED;
          continue;
        }
      }

      /* Check segmap */
      if (im->segmap) {
        segt = (const char *)im->segmap + pos * ssize;
        int segval = (int)sconvert(segt);
        if (segval != 0 && segval != id) {
          data_vec[pidx] = 0.0;
          weight[pidx] = 0.0;
          *flag |= SEP_APER_HASMASKED;
          continue;
        }
      }

      /* Compute variance and weight */
      if (errisarray) {
        errort = (const char *)im->noise + pos * esize;
        varpix = econvert(errort);
        if (errisstd) varpix *= varpix;
      }

      if (varpix > 0.0) {
        double total_var = varpix;
        if (im->gain > 0.0 && pix > 0.0) {
          total_var += pix / im->gain;
        }
        weight[pidx] = 1.0 / sqrt(total_var);
      } else {
        weight[pidx] = 0.0;
      }

      data_vec[pidx] = pix * weight[pidx];
    }
  }

  /* Iterative fitting loop */
  for (iter = 0; iter < maxiter; iter++) {
    convflag = 0;
    (*niter)++;

    /* Build PSF at current position estimate */
    status = sep_psf_build(psf, x + deltax, y + deltay);
    if (status != RETURN_OK) return status;

    status = sep_psf_resample(psf, (x + deltax) - ix0, (y + deltay) - iy0);
    if (status != RETURN_OK) return status;

    /* Normalize PSF stamp */
    {
      double psfsum = 0.0;
      for (sx = 0; sx < npix; sx++) psfsum += psf->resi[sx];
      if (psfsum > 0.0) {
        for (sx = 0; sx < npix; sx++) psf->resi[sx] /= (float)psfsum;
      }
    }

    /* Build design matrix */
    compute_gradient(psf->resi, weight, width, height, mat);

    /* Solve via SVD */
    status = svdfit(mat, data_vec, npix, npar, sol, vmat, wmat, rv1, tmp);
    if (status != RETURN_OK) return status;

    /* Extract flux and position updates */
    *flux = sol[0];
    if (fabs(*flux) > 0.0) {
      dx_update = -sol[1] / *flux;
      dy_update = -sol[2] / *flux;
    } else {
      dx_update = 0.0;
      dy_update = 0.0;
    }

    deltax += dx_update;
    deltay += dy_update;

    /* Check convergence */
    if (dx_update * dx_update + dy_update * dy_update >
        PSF_MINSHIFT * PSF_MINSHIFT) {
      convflag = 1;
    }

    /* Check if drifted too far */
    if (deltax * deltax + deltay * deltay > radmax2) {
      /* Reset to initial position */
      deltax = 0.0;
      deltay = 0.0;
      break;
    }

    if (!convflag) break;
  }

  /* Final position */
  *xfit = x + deltax;
  *yfit = y + deltay;

  /* Compute covariance from SVD */
  memset(covmat, 0, (size_t)(npar * npar) * sizeof(double));
  status = svdvar(vmat, wmat, npar, covmat);
  if (status != RETURN_OK) return status;

  /* Extract errors */
  {
    double var_flux = covmat[0];
    if (var_flux < 0.0) var_flux = 0.0;
    if (im->gain > 0.0 && *flux > 0.0) {
      var_flux += *flux / im->gain;
    }
    *fluxerr = sqrt(var_flux);

    if (fabs(*flux) > 0.0) {
      double f2 = *flux * *flux;
      double var_x = covmat[1 * npar + 1];
      double var_y = covmat[2 * npar + 2];
      *xerr = sqrt(var_x > 0.0 ? var_x / f2 : 0.0);
      *yerr = sqrt(var_y > 0.0 ? var_y / f2 : 0.0);
    }
  }

  /* Compute chi-squared */
  {
    double chi2sum = 0.0;
    int ngood = 0;

    /* Rebuild PSF at final position for residuals */
    status = sep_psf_build(psf, *xfit, *yfit);
    if (status != RETURN_OK) return status;
    status = sep_psf_resample(psf, *xfit - ix0, *yfit - iy0);
    if (status != RETURN_OK) return status;

    {
      double psfsum = 0.0;
      for (sx = 0; sx < npix; sx++) psfsum += psf->resi[sx];
      if (psfsum > 0.0) {
        for (sx = 0; sx < npix; sx++) psf->resi[sx] /= (float)psfsum;
      }
    }

    for (sx = 0; sx < npix; sx++) {
      if (weight[sx] > 0.0) {
        double model = *flux * psf->resi[sx] * weight[sx];
        double resid = data_vec[sx] - model;
        chi2sum += resid * resid;
        ngood++;
      }
    }

    if (ngood > npar) {
      *chi2 = chi2sum / (ngood - npar);
    }
  }

  return status;
}

/*==========================================================================*/
/*           Batch PSF Fitting (non-grouped, C loop)                        */
/*==========================================================================*/

int sep_psf_fit_array(const sep_image *im, sep_psf *psf, const double *x,
                      const double *y, int64_t n, const int *id, short inflag,
                      int maxiter, double *flux, double *fluxerr, double *xfit,
                      double *yfit, double *xerr, double *yerr, int *niter,
                      double *chi2, short *flag) {
#ifdef _OPENMP
  int first_status = RETURN_OK;

#pragma omp parallel
  {
    sep_psf local_psf;
    int ws_status = psf_workspace_clone(psf, &local_psf);

    if (ws_status != RETURN_OK) {
#pragma omp critical(psf_fit_status)
      {
        if (first_status == RETURN_OK) first_status = ws_status;
      }
    } else {
#pragma omp for schedule(dynamic, 32)
      for (int64_t i = 0; i < n; i++) {
        int s = sep_psf_fit(im, &local_psf, x[i], y[i], id ? id[i] : 0, inflag,
                            maxiter, &flux[i], &fluxerr[i], &xfit[i], &yfit[i],
                            &xerr[i], &yerr[i], &niter[i], &chi2[i], &flag[i]);
        if (s != RETURN_OK) {
#pragma omp critical(psf_fit_status)
          {
            if (first_status == RETURN_OK) first_status = s;
          }
        }
      }
    }

    psf_workspace_free(&local_psf);
  }

  return first_status;
#else
  int status = RETURN_OK;
  int64_t i;

  for (i = 0; i < n; i++) {
    status = sep_psf_fit(im, psf, x[i], y[i], id ? id[i] : 0, inflag, maxiter,
                         &flux[i], &fluxerr[i], &xfit[i], &yfit[i], &xerr[i],
                         &yerr[i], &niter[i], &chi2[i], &flag[i]);
    if (status != RETURN_OK) return status;
  }

  return RETURN_OK;
#endif
}

/*==========================================================================*/
/*           Union-Find for grouping (same as aperture.c)                   */
/*==========================================================================*/

static int psf_uf_find(int *parent, int i) {
  while (parent[i] != i) {
    parent[i] = parent[parent[i]]; /* path compression */
    i = parent[i];
  }
  return i;
}

static void psf_uf_union(int *parent, int *rank, int a, int b) {
  int ra = psf_uf_find(parent, a);
  int rb = psf_uf_find(parent, b);
  if (ra == rb) return;
  if (rank[ra] < rank[rb]) {
    parent[ra] = rb;
  } else if (rank[ra] > rank[rb]) {
    parent[rb] = ra;
  } else {
    parent[rb] = ra;
    rank[ra] += 1;
  }
}

typedef struct {
  double x;
  int idx;
} psf_xorder_entry;

static int psf_xorder_cmp(const void *a, const void *b) {
  const psf_xorder_entry *pa = (const psf_xorder_entry *)a;
  const psf_xorder_entry *pb = (const psf_xorder_entry *)b;
  if (pa->x < pb->x) return -1;
  if (pa->x > pb->x) return 1;
  return (pa->idx > pb->idx) - (pa->idx < pb->idx);
}

typedef struct {
  int gid;
  int count;
} psf_group_order_entry;

static int psf_group_order_cmp(const void *a, const void *b) {
  const psf_group_order_entry *ga = (const psf_group_order_entry *)a;
  const psf_group_order_entry *gb = (const psf_group_order_entry *)b;
  if (ga->count > gb->count) return -1;
  if (ga->count < gb->count) return 1;
  return (ga->gid > gb->gid) - (ga->gid < gb->gid);
}

static int psf_fit_group(const sep_image *im, sep_psf *psf, const double *x,
                         const double *y, const int *id, short inflag,
                         int maxiter, const int *group_counts,
                         const int *group_offsets, const int *members, int g,
                         double *pflux, double *pfluxerr, double *pxfit,
                         double *pyfit, double *pxerr, double *pyerr,
                         int *pniter, double *pchi2, short *pflag) {
  int status = RETURN_OK;
  int i;
  int gcount = group_counts[g];
  const int *gidx = members + group_offsets[g];
  double dx, dy;

  if (gcount == 1) {
    /* Singleton: delegate to single-source fitter */
    int idx = gidx[0];
    status = sep_psf_fit(im, psf, x[idx], y[idx], id ? id[idx] : 0, inflag,
                         maxiter, &pflux[idx], &pfluxerr[idx], &pxfit[idx],
                         &pyfit[idx], &pxerr[idx], &pyerr[idx], &pniter[idx],
                         &pchi2[idx], &pflag[idx]);
    return status;
  }

  /* Multi-member group: simultaneous fitting */
  {
    int npar = gcount * PSF_NA;
    int width = psf->rw;
    int height = psf->rh;
    int64_t gxmin, gxmax, gymin, gymax;
    int gw, gh, gnpix;
    int iter, convflag, used_svd_last, used_svd_this;
    double *gdata = NULL, *gweight = NULL;
    double *gmat = NULL, *gsol = NULL, *gvmat = NULL, *gwmat = NULL;
    double *gcovmat = NULL, *grv1 = NULL, *gtmp = NULL;
    double *deltax_arr = NULL, *deltay_arr = NULL;
    float **psfstamps = NULL;
    converter convert, econvert, mconvert, sconvert;
    int64_t sz, esz, msz, ssz;
    int errisarray, errisstd;
    double vp;
    int alloc_ok = 1;

    used_svd_last = 0;

    /* Compute group bounding box */
    gxmin = (int64_t)im->w;
    gxmax = 0;
    gymin = (int64_t)im->h;
    gymax = 0;
    for (i = 0; i < gcount; i++) {
      int idx = gidx[i];
      int64_t lxmin = (int64_t)(x[idx] + 0.5) - width / 2;
      int64_t lxmax = lxmin + width;
      int64_t lymin = (int64_t)(y[idx] + 0.5) - height / 2;
      int64_t lymax = lymin + height;
      if (lxmin < gxmin) gxmin = lxmin;
      if (lxmax > gxmax) gxmax = lxmax;
      if (lymin < gymin) gymin = lymin;
      if (lymax > gymax) gymax = lymax;
    }
    if (gxmin < 0) gxmin = 0;
    if (gymin < 0) gymin = 0;
    if (gxmax > im->w) gxmax = im->w;
    if (gymax > im->h) gymax = im->h;
    gw = (int)(gxmax - gxmin);
    gh = (int)(gymax - gymin);
    gnpix = gw * gh;

    if (gnpix < npar || gw <= 0 || gh <= 0) {
      /* Group too small, fit individually */
      for (i = 0; i < gcount; i++) {
        int idx = gidx[i];
        status = sep_psf_fit(im, psf, x[idx], y[idx], id ? id[idx] : 0, inflag,
                             maxiter, &pflux[idx], &pfluxerr[idx], &pxfit[idx],
                             &pyfit[idx], &pxerr[idx], &pyerr[idx], &pniter[idx],
                             &pchi2[idx], &pflag[idx]);
        if (status != RETURN_OK) return status;
      }
      return RETURN_OK;
    }

    /* Allocate group working arrays */
    gdata = (double *)calloc((size_t)gnpix, sizeof(double));
    gweight = (double *)calloc((size_t)gnpix, sizeof(double));
    gmat = (double *)malloc((size_t)gnpix * npar * sizeof(double));
    gsol = (double *)malloc((size_t)npar * sizeof(double));
    gvmat = (double *)malloc((size_t)npar * npar * sizeof(double));
    gwmat = (double *)malloc((size_t)npar * sizeof(double));
    gcovmat = (double *)calloc((size_t)npar * npar, sizeof(double));
    grv1 = (double *)malloc((size_t)npar * sizeof(double));
    gtmp = (double *)malloc((size_t)npar * sizeof(double));
    deltax_arr = (double *)calloc((size_t)gcount, sizeof(double));
    deltay_arr = (double *)calloc((size_t)gcount, sizeof(double));
    psfstamps = (float **)calloc((size_t)gcount, sizeof(float *));

    if (!gdata || !gweight || !gmat || !gsol || !gvmat || !gwmat || !gcovmat ||
        !grv1 || !gtmp || !deltax_arr || !deltay_arr || !psfstamps) {
      alloc_ok = 0;
    }

    if (alloc_ok) {
      for (i = 0; i < gcount; i++) {
        psfstamps[i] = (float *)calloc((size_t)gw * gh, sizeof(float));
        if (!psfstamps[i]) {
          alloc_ok = 0;
          break;
        }
      }
    }

    if (!alloc_ok) {
      status = MEMORY_ALLOC_ERROR;
      goto group_exit;
    }

    /* Get converters */
    if ((status = get_converter(im->dtype, &convert, &sz))) goto group_exit;
    errisarray = 0;
    errisstd = 0;
    vp = 1.0;
    msz = 0;
    ssz = 0;
    esz = 0;
    if (im->mask) {
      if ((status = get_converter(im->mdtype, &mconvert, &msz))) goto group_exit;
    }
    if (im->segmap) {
      if ((status = get_converter(im->sdtype, &sconvert, &ssz))) goto group_exit;
    }
    if (im->noise_type != SEP_NOISE_NONE) {
      errisstd = (im->noise_type == SEP_NOISE_STDDEV);
      if (im->noise) {
        errisarray = 1;
        if ((status = get_converter(im->ndtype, &econvert, &esz)))
          goto group_exit;
      } else {
        vp = (errisstd) ? im->noiseval * im->noiseval : im->noiseval;
      }
    }

    /* Extract data and weights for group bounding box */
    {
      int64_t ix, iy;
      int64_t pos2;
      for (iy = gymin; iy < gymax; iy++) {
        for (ix = gxmin; ix < gxmax; ix++) {
          int pidx = (int)((iy - gymin) * gw + (ix - gxmin));
          pos2 = iy * im->w + ix;
          double pix_val = ((converter)convert)((const char *)im->data + pos2 * sz);
          double var_val = vp;

          /* Check mask */
          if (im->mask) {
            if (((converter)mconvert)((const char *)im->mask + pos2 * msz) >
                im->maskthresh) {
              gdata[pidx] = 0.0;
              gweight[pidx] = 0.0;
              continue;
            }
          }

          /* Check segmap: mask pixel if it belongs to a source not in this group */
          if (im->segmap) {
            int segval =
                (int)((converter)sconvert)((const char *)im->segmap + pos2 * ssz);
            if (segval != 0) {
              int in_group = 0;
              int ki;
              for (ki = 0; ki < gcount; ki++) {
                if (id && segval == id[gidx[ki]]) {
                  in_group = 1;
                  break;
                }
              }
              if (!in_group) {
                gdata[pidx] = 0.0;
                gweight[pidx] = 0.0;
                continue;
              }
            }
          }

          if (errisarray) {
            var_val = ((converter)econvert)((const char *)im->noise + pos2 * esz);
            if (errisstd) var_val *= var_val;
          }

          if (var_val > 0.0) {
            double total_var = var_val;
            if (im->gain > 0.0 && pix_val > 0.0) {
              total_var += pix_val / im->gain;
            }
            gweight[pidx] = 1.0 / sqrt(total_var);
          } else {
            gweight[pidx] = 0.0;
          }
          gdata[pidx] = pix_val * gweight[pidx];
        }
      }
    }

    /* Compute per-source flags by checking each source's stamp */
    for (i = 0; i < gcount; i++) {
      int idx = gidx[i];
      int cix = (int)(x[idx] + 0.5);
      int ciy = (int)(y[idx] + 0.5);
      int psy, psx;
      for (psy = 0; psy < height; psy++) {
        int64_t imy2 = ciy - height / 2 + psy;
        for (psx = 0; psx < width; psx++) {
          int64_t imx2 = cix - width / 2 + psx;
          if (imy2 < 0 || imy2 >= im->h || imx2 < 0 || imx2 >= im->w) {
            pflag[idx] |= SEP_APER_TRUNC;
            continue;
          }
          int64_t pos2 = imy2 * im->w + imx2;
          if (im->mask) {
            if (((converter)mconvert)((const char *)im->mask + pos2 * msz) >
                im->maskthresh) {
              pflag[idx] |= SEP_APER_HASMASKED;
            }
          }
          if (im->segmap) {
            int segval =
                (int)((converter)sconvert)((const char *)im->segmap + pos2 * ssz);
            if (segval != 0 && (!id || segval != id[idx])) {
              pflag[idx] |= SEP_APER_HASMASKED;
            }
          }
        }
      }
    }

    /* Iterative grouped fitting */
    for (iter = 0; iter < maxiter; iter++) {
      convflag = 0;

      /* Build PSF stamps for each group member at current positions */
      for (i = 0; i < gcount; i++) {
        int idx = gidx[i];
        double cx = x[idx] + deltax_arr[i];
        double cy = y[idx] + deltay_arr[i];
        int cix = (int)(cx + 0.5);
        int ciy = (int)(cy + 0.5);
        double psfsum;

        status = sep_psf_build(psf, cx, cy);
        if (status != RETURN_OK) goto group_exit;
        status = sep_psf_resample(psf, cx - cix, cy - ciy);
        if (status != RETURN_OK) goto group_exit;

        /* Normalize */
        psfsum = 0.0;
        {
          int p;
          for (p = 0; p < psf->rw * psf->rh; p++) psfsum += psf->resi[p];
        }
        if (psfsum > 0.0) {
          int p;
          for (p = 0; p < psf->rw * psf->rh; p++) psf->resi[p] /= (float)psfsum;
        }

        /* Place PSF stamp into group-sized buffer */
        memset(psfstamps[i], 0, (size_t)gw * gh * sizeof(float));
        {
          int psy, psx;
          for (psy = 0; psy < psf->rh; psy++) {
            int64_t imy2 = ciy - psf->rh / 2 + psy;
            if (imy2 < gymin || imy2 >= gymax) continue;
            for (psx = 0; psx < psf->rw; psx++) {
              int64_t imx2 = cix - psf->rw / 2 + psx;
              if (imx2 < gxmin || imx2 >= gxmax) continue;
              int gi = (int)((imy2 - gymin) * gw + (imx2 - gxmin));
              psfstamps[i][gi] = psf->resi[psy * psf->rw + psx];
            }
          }
        }
      }

      /* Build joint design matrix: 3 columns per member */
      {
        double *mp = gmat;
        for (i = 0; i < gcount; i++) {
          compute_gradient(psfstamps[i], gweight, gw, gh, mp);
          mp += (size_t)gnpix * PSF_NA;
        }
      }

      /* Solve least squares: Cholesky on normal equations, fall back to SVD. */
      used_svd_this = 0;
      psf_build_normal_eq(gmat, gdata, gnpix, npar, gvmat, gwmat);
      status = psf_cholesky_factor(gvmat, npar);
      if (status == RETURN_OK) {
        psf_cholesky_solve(gvmat, npar, gwmat, gsol);
      } else {
        status = svdfit(gmat, gdata, gnpix, npar, gsol, gvmat, gwmat, grv1,
                        gtmp);
        if (status == RETURN_OK) used_svd_this = 1;
      }
      if (status != RETURN_OK) {
        /* Dense solve failed, fall back to individual fits */
        status = RETURN_OK;
        for (i = 0; i < gcount; i++) {
          int idx2 = gidx[i];
          int fb_status = sep_psf_fit(
              im, psf, x[idx2], y[idx2], id ? id[idx2] : 0, inflag, maxiter,
              &pflux[idx2], &pfluxerr[idx2], &pxfit[idx2], &pyfit[idx2],
              &pxerr[idx2], &pyerr[idx2], &pniter[idx2], &pchi2[idx2],
              &pflag[idx2]);
          if (fb_status != RETURN_OK) status = fb_status;
        }
        goto group_exit;
      }
      used_svd_last = used_svd_this;

      /* Update positions with damping for multi-component */
      for (i = 0; i < gcount; i++) {
        double fi = gsol[i * PSF_NA];
        pflux[gidx[i]] = fi;
        if (fabs(fi) > 0.0) {
          /* Factor of 2 damping for multi-component (SExtractor convention) */
          dx = -gsol[i * PSF_NA + 1] / (2.0 * fi);
          dy = -gsol[i * PSF_NA + 2] / (2.0 * fi);
        } else {
          dx = 0.0;
          dy = 0.0;
        }
        deltax_arr[i] += dx;
        deltay_arr[i] += dy;

        if (dx * dx + dy * dy > PSF_MINSHIFT * PSF_MINSHIFT) {
          convflag = 1;
        }
      }

      pniter[gidx[0]] = iter + 1;
      if (!convflag) break;
    }

    /* Store final positions */
    for (i = 0; i < gcount; i++) {
      int idx = gidx[i];
      pxfit[idx] = x[idx] + deltax_arr[i];
      pyfit[idx] = y[idx] + deltay_arr[i];
      pniter[idx] = pniter[gidx[0]];
    }

    /* Compute covariance and extract errors */
    if (used_svd_last) {
      status = svdvar(gvmat, gwmat, npar, gcovmat);
    } else {
      status = psf_cholesky_inverse(gvmat, npar, gcovmat, gwmat, gtmp);
    }
    if (status == RETURN_OK) {
      for (i = 0; i < gcount; i++) {
        int idx = gidx[i];
        int base = i * PSF_NA;
        double var_f = gcovmat[base * npar + base];
        if (var_f < 0.0) var_f = 0.0;
        if (im->gain > 0.0 && pflux[idx] > 0.0) {
          var_f += pflux[idx] / im->gain;
        }
        pfluxerr[idx] = sqrt(var_f);

        if (fabs(pflux[idx]) > 0.0) {
          double f2 = pflux[idx] * pflux[idx];
          double vx = gcovmat[(base + 1) * npar + (base + 1)];
          double vy = gcovmat[(base + 2) * npar + (base + 2)];
          pxerr[idx] = sqrt(vx > 0.0 ? vx / f2 : 0.0);
          pyerr[idx] = sqrt(vy > 0.0 ? vy / f2 : 0.0);
        }
      }
    }

  group_exit:
    for (i = 0; i < gcount; i++) {
      if (psfstamps) free(psfstamps[i]);
    }
    free(psfstamps);
    free(gdata);
    free(gweight);
    free(gmat);
    free(gsol);
    free(gvmat);
    free(gwmat);
    free(gcovmat);
    free(grv1);
    free(gtmp);
    free(deltax_arr);
    free(deltay_arr);
  }

  return status;
}

/*==========================================================================*/
/*                     Grouped PSF Fitting                                  */
/*==========================================================================*/

int sep_psf_fit_multi(const sep_image *im, sep_psf *psf, const double *x,
                      const double *y, int64_t n, const int *id,
                      double group_factor, short inflag, int maxiter,
                      double *pflux, double *pfluxerr, double *pxfit,
                      double *pyfit, double *pxerr, double *pyerr,
                      int *pniter, double *pchi2, short *pflag) {
  int status = RETURN_OK;
  int *parent = NULL, *rank_arr = NULL, *group_id = NULL, *root_map = NULL;
  int *group_counts = NULL, *group_offsets = NULL, *group_fill = NULL;
  int *members = NULL;
  psf_xorder_entry *xorder = NULL;
  psf_group_order_entry *group_order = NULL;
  int i, j, g, ngroups;
  int has_multi;
  double half_stamp, rsum, rsum2, dx, dy, dist2;

  if (n <= 0) return RETURN_OK;
  if (maxiter <= 0) maxiter = 20;

  half_stamp = (double)(psf->rw + psf->rh) / 4.0;

  /* Initialize outputs */
  for (i = 0; i < n; i++) {
    pflux[i] = 0.0;
    pfluxerr[i] = 0.0;
    pxfit[i] = x[i];
    pyfit[i] = y[i];
    pxerr[i] = 0.0;
    pyerr[i] = 0.0;
    pniter[i] = 0;
    pchi2[i] = 0.0;
    pflag[i] = 0;
  }

  /* Allocate grouping arrays */
  parent = (int *)malloc((size_t)n * sizeof(int));
  rank_arr = (int *)calloc((size_t)n, sizeof(int));
  group_id = (int *)malloc((size_t)n * sizeof(int));
  root_map = (int *)malloc((size_t)n * sizeof(int));
  group_counts = (int *)calloc((size_t)n, sizeof(int));
  group_offsets = (int *)malloc((size_t)(n + 1) * sizeof(int));
  group_fill = (int *)malloc((size_t)n * sizeof(int));
  members = (int *)malloc((size_t)n * sizeof(int));
  xorder = (psf_xorder_entry *)malloc((size_t)n * sizeof(psf_xorder_entry));

  if (!parent || !rank_arr || !group_id || !root_map || !group_counts ||
      !group_offsets || !group_fill || !members || !xorder) {
    status = MEMORY_ALLOC_ERROR;
    goto cleanup;
  }

  /* Initialize union-find */
  for (i = 0; i < n; i++) {
    parent[i] = i;
    xorder[i].x = x[i];
    xorder[i].idx = i;
  }
  qsort(xorder, (size_t)n, sizeof(psf_xorder_entry), psf_xorder_cmp);

  /* Build groups based on stamp overlap.
   * Sweep in x-order and only test pairs inside +/-rsum x-window. */
  rsum = group_factor * (half_stamp + half_stamp);
  rsum2 = rsum * rsum;
  if (rsum > 0.0) {
    for (i = 0; i < n; i++) {
      int ii = xorder[i].idx;
      double xi = xorder[i].x;
      for (j = i + 1; j < n; j++) {
        int jj = xorder[j].idx;
        dx = xorder[j].x - xi;
        if (dx > rsum) break;
        dy = y[ii] - y[jj];
        if (dy > rsum || dy < -rsum) continue;
        dist2 = dx * dx + dy * dy;
        if (dist2 <= rsum2) {
          psf_uf_union(parent, rank_arr, ii, jj);
        }
      }
    }
  }

  /* Map roots to sequential group IDs */
  for (i = 0; i < n; i++) root_map[i] = -1;
  g = 0;
  for (i = 0; i < n; i++) {
    int root = psf_uf_find(parent, i);
    if (root_map[root] < 0) {
      root_map[root] = g++;
    }
    group_id[i] = root_map[root];
    group_counts[group_id[i]] += 1;
  }
  ngroups = g;

  /* Build member index arrays */
  group_offsets[0] = 0;
  for (i = 0; i < ngroups; i++) {
    group_offsets[i + 1] = group_offsets[i] + group_counts[i];
    group_fill[i] = group_offsets[i];
  }
  for (i = 0; i < n; i++) {
    int gid = group_id[i];
    members[group_fill[gid]++] = i;
  }

  has_multi = 0;
  for (i = 0; i < ngroups; i++) {
    if (group_counts[i] > 1) {
      has_multi = 1;
      break;
    }
  }

  if (!has_multi) {
    status = sep_psf_fit_array(im, psf, x, y, n, id, inflag, maxiter, pflux,
                               pfluxerr, pxfit, pyfit, pxerr, pyerr, pniter,
                               pchi2, pflag);
    goto cleanup;
  }

  group_order =
      (psf_group_order_entry *)malloc((size_t)ngroups * sizeof(*group_order));
  if (!group_order) {
    status = MEMORY_ALLOC_ERROR;
    goto cleanup;
  }
  for (i = 0; i < ngroups; i++) {
    group_order[i].gid = i;
    group_order[i].count = group_counts[i];
  }
  qsort(group_order, (size_t)ngroups, sizeof(*group_order),
        psf_group_order_cmp);

  /* Process each group */
#ifdef _OPENMP
  {
    int first_status = RETURN_OK;

#pragma omp parallel
    {
      sep_psf local_psf;
      sep_psf *work_psf = psf;
      int ws_status = psf_workspace_clone(psf, &local_psf);

      if (ws_status == RETURN_OK) {
        work_psf = &local_psf;
      } else {
#pragma omp critical(psf_multi_status)
        {
          if (first_status == RETURN_OK) first_status = ws_status;
        }
      }

#pragma omp for schedule(dynamic, 1)
      for (int go = 0; go < ngroups; go++) {
        int gid = group_order[go].gid;
        int gstatus;
        if (ws_status != RETURN_OK) continue;
        gstatus = psf_fit_group(im, work_psf, x, y, id, inflag, maxiter,
                                group_counts, group_offsets, members, gid,
                                pflux, pfluxerr, pxfit, pyfit, pxerr, pyerr,
                                pniter, pchi2, pflag);
        if (gstatus != RETURN_OK) {
#pragma omp critical(psf_multi_status)
          {
            if (first_status == RETURN_OK) first_status = gstatus;
          }
        }
      }

      psf_workspace_free(&local_psf);
    }

    if (first_status != RETURN_OK) {
      status = first_status;
      goto cleanup;
    }
  }
#else
  for (i = 0; i < ngroups; i++) {
    g = group_order[i].gid;
    status = psf_fit_group(im, psf, x, y, id, inflag, maxiter, group_counts,
                           group_offsets, members, g, pflux, pfluxerr, pxfit,
                           pyfit, pxerr, pyerr, pniter, pchi2, pflag);
    if (status != RETURN_OK) goto cleanup;
  }
#endif

cleanup:
  free(group_order);
  free(xorder);
  free(parent);
  free(rank_arr);
  free(group_id);
  free(root_map);
  free(group_counts);
  free(group_offsets);
  free(group_fill);
  free(members);
  return status;
}
