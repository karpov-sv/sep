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
#define PSF_GROUP_EXACT_MAX 64
#define PSF_GROUP_CORE 16
#define PSF_GROUP_SWEEPS 2
#define PSF_GROUP_SVD_TOL 1.0e-4
#define PSF_FLUX_NNLS_MAXITER 512
#define PSF_FLUX_NNLS_TOL 1.0e-10

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

typedef struct {
  int max_gnpix;
  int max_gcount;
  int max_group_ids;
  int max_npar;
  int max_stamp_npix;
  size_t gmat_capacity;
  double *gdata;
  double *gweight;
  double *cols;
  double *ata;
  double *atb;
  double *sol;
  double *cov;
  double *rv1;
  double *tmp;
  double *deltax;
  double *deltay;
  int *ixoff;
  int *iyoff;
  int *group_ids;
  double *gmat;
} psf_group_workspace;

static void psf_group_workspace_nullify(psf_group_workspace *ws) {
  if (!ws) return;
  ws->max_gnpix = 0;
  ws->max_gcount = 0;
  ws->max_group_ids = 0;
  ws->max_npar = 0;
  ws->max_stamp_npix = 0;
  ws->gmat_capacity = 0;
  ws->gdata = NULL;
  ws->gweight = NULL;
  ws->cols = NULL;
  ws->ata = NULL;
  ws->atb = NULL;
  ws->sol = NULL;
  ws->cov = NULL;
  ws->rv1 = NULL;
  ws->tmp = NULL;
  ws->deltax = NULL;
  ws->deltay = NULL;
  ws->ixoff = NULL;
  ws->iyoff = NULL;
  ws->group_ids = NULL;
  ws->gmat = NULL;
}

static void psf_group_workspace_free(psf_group_workspace *ws) {
  if (!ws) return;
  free(ws->gdata);
  free(ws->gweight);
  free(ws->cols);
  free(ws->ata);
  free(ws->atb);
  free(ws->sol);
  free(ws->cov);
  free(ws->rv1);
  free(ws->tmp);
  free(ws->deltax);
  free(ws->deltay);
  free(ws->ixoff);
  free(ws->iyoff);
  free(ws->group_ids);
  free(ws->gmat);
  psf_group_workspace_nullify(ws);
}

static int psf_group_realloc_double(double **ptr, size_t n) {
  void *tmp = realloc(*ptr, n * sizeof(double));
  if (!tmp && n > 0) return MEMORY_ALLOC_ERROR;
  *ptr = (double *)tmp;
  return RETURN_OK;
}

static int psf_group_realloc_int(int **ptr, size_t n) {
  void *tmp = realloc(*ptr, n * sizeof(int));
  if (!tmp && n > 0) return MEMORY_ALLOC_ERROR;
  *ptr = (int *)tmp;
  return RETURN_OK;
}

static int psf_group_workspace_ensure(psf_group_workspace *ws, int gnpix,
                                      int gcount, int stamp_npix,
                                      int group_id_count) {
  int status = RETURN_OK;
  int npar = gcount * PSF_NA;

  if (gnpix > ws->max_gnpix) {
    if ((status = psf_group_realloc_double(&ws->gdata, (size_t)gnpix)))
      return status;
    if ((status = psf_group_realloc_double(&ws->gweight, (size_t)gnpix)))
      return status;
    ws->max_gnpix = gnpix;
  }

  if (stamp_npix > ws->max_stamp_npix || gcount > ws->max_gcount) {
    size_t need = (size_t)gcount * PSF_NA * (size_t)stamp_npix;
    if ((status = psf_group_realloc_double(&ws->cols, need))) return status;
    ws->max_stamp_npix = stamp_npix;
  }

  if (gcount > ws->max_gcount) {
    if ((status = psf_group_realloc_double(&ws->deltax, (size_t)gcount)))
      return status;
    if ((status = psf_group_realloc_double(&ws->deltay, (size_t)gcount)))
      return status;
    if ((status = psf_group_realloc_int(&ws->ixoff, (size_t)gcount)))
      return status;
    if ((status = psf_group_realloc_int(&ws->iyoff, (size_t)gcount)))
      return status;
    ws->max_gcount = gcount;
  }

  if (group_id_count > ws->max_group_ids) {
    if ((status =
             psf_group_realloc_int(&ws->group_ids, (size_t)group_id_count)))
      return status;
    ws->max_group_ids = group_id_count;
  }

  if (npar > ws->max_npar) {
    size_t npar2 = (size_t)npar * (size_t)npar;
    if ((status = psf_group_realloc_double(&ws->ata, npar2))) return status;
    if ((status = psf_group_realloc_double(&ws->atb, (size_t)npar)))
      return status;
    if ((status = psf_group_realloc_double(&ws->sol, (size_t)npar)))
      return status;
    if ((status = psf_group_realloc_double(&ws->cov, npar2))) return status;
    if ((status = psf_group_realloc_double(&ws->rv1, (size_t)npar)))
      return status;
    if ((status = psf_group_realloc_double(&ws->tmp, (size_t)npar)))
      return status;
    ws->max_npar = npar;
  }

  return RETURN_OK;
}

static int psf_group_workspace_ensure_gmat(psf_group_workspace *ws, int gnpix,
                                           int npar) {
  size_t need = (size_t)gnpix * (size_t)npar;
  if (need > ws->gmat_capacity) {
    int status = psf_group_realloc_double(&ws->gmat, need);
    if (status != RETURN_OK) return status;
    ws->gmat_capacity = need;
  }
  return RETURN_OK;
}

/*==========================================================================*/
/*                         PSF Lifecycle                                    */
/*==========================================================================*/

int sep_psf_create(sep_psf **out, const float *data, int w, int h, int ncomp,
                   int degree, double x0, double y0, double sx, double sy,
                   float pixstep, double fwhm) {
  sep_psf *psf = NULL;
  int status = RETURN_OK;
  int npix, datalen, rw, rh, rnpix;
  int maxdim, mask_len, nmask_len, buf_len, npar, remap_kw;

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
  psf->fit_radius = 0.0; /* 0 = use full stamp */

  /* Compute native-resolution stamp dimensions from continuous sampling. */
  rw = (int)floor((double)w * (double)pixstep + 0.5);
  rh = (int)floor((double)h * (double)pixstep + 0.5);
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
  remap_kw = (int)ceil(1.0f / pixstep) + 4;
  if (remap_kw < INTERPW) remap_kw = INTERPW;
  mask_len = maxdim * remap_kw;
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
/*         PSF Resampling                                                   */
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

  /* For supersampled PSFs, use conservative area-overlap remapping.
   * This preserves flux across sampling changes and reduces phase bias. */
  if (psf->pixstep < 1.0f) {
    float xlo, xhi, ylo, yhi, left, right, ov;

    /* Compute x overlap kernels */
    x1 = xs1;
    maskt = mask;
    nmaskt = nmask;
    startt = start;
    for (j = 0; j < nx2; j++, x1 += step2) {
      int ix_start, ix_end;
      xlo = x1 - 0.5f * step2;
      xhi = x1 + 0.5f * step2;
      ix_start = (int)floorf(xlo) - 1;
      ix_end = (int)floorf(xhi) + 1;
      if (ix_start < 0) ix_start = 0;
      if (ix_end > w1 - 1) ix_end = w1 - 1;
      n = ix_end - ix_start + 1;
      if (n <= 0) {
        *(startt++) = 0;
        *(nmaskt++) = 0;
        continue;
      }
      *(startt++) = ix_start;
      *(nmaskt++) = n;
      for (i = 0; i < n; i++) {
        ix = ix_start + i;
        left = (float)ix - 0.5f;
        right = left + 1.0f;
        ov = fminf(xhi, right) - fmaxf(xlo, left);
        *(maskt++) = ov > 0.0f ? ov : 0.0f;
      }
    }

    /* Integrate in x for every input row (store transposed) */
    memset(pix12, 0, (size_t)nx2 * h1 * sizeof(float));
    pixin0 = pix1;
    for (k = 0; k < h1; k++, pixin0 += w1) {
      maskt = mask;
      nmaskt = nmask;
      startt = start;
      for (j = 0; j < nx2; j++) {
        int sx0 = *(startt++);
        n = *(nmaskt++);
        val = 0.0f;
        if (n > 0) {
          pixin = pixin0 + sx0;
#if defined(_OPENMP)
#pragma omp simd reduction(+ : val)
#endif
          for (i = 0; i < n; i++) val += maskt[i] * pixin[i];
          maskt += n;
        }
        pix12[j * h1 + k] = val;
      }
    }

    /* Compute y overlap kernels */
    y1 = ys1;
    maskt = mask;
    nmaskt = nmask;
    startt = start;
    for (j = 0; j < ny2; j++, y1 += step2) {
      int iy_start, iy_end;
      ylo = y1 - 0.5f * step2;
      yhi = y1 + 0.5f * step2;
      iy_start = (int)floorf(ylo) - 1;
      iy_end = (int)floorf(yhi) + 1;
      if (iy_start < 0) iy_start = 0;
      if (iy_end > h1 - 1) iy_end = h1 - 1;
      n = iy_end - iy_start + 1;
      if (n <= 0) {
        *(startt++) = 0;
        *(nmaskt++) = 0;
        continue;
      }
      *(startt++) = iy_start;
      *(nmaskt++) = n;
      for (i = 0; i < n; i++) {
        iy = iy_start + i;
        left = (float)iy - 0.5f;
        right = left + 1.0f;
        ov = fminf(yhi, right) - fmaxf(ylo, left);
        *(maskt++) = ov > 0.0f ? ov : 0.0f;
      }
    }

    /* Integrate in y and transpose back to output stamp */
    pixout0 = pix2 + ixs2 + iys2 * w2;
    for (k = 0; k < nx2; k++, pixout0++) {
      pixin0 = pix12 + k * h1;
      pixout = pixout0;
      maskt = mask;
      nmaskt = nmask;
      startt = start;
      for (j = 0; j < ny2; j++, pixout += w2) {
        int sy0 = *(startt++);
        n = *(nmaskt++);
        val = 0.0f;
        if (n > 0) {
          pixin = pixin0 + sy0;
#if defined(_OPENMP)
#pragma omp simd reduction(+ : val)
#endif
          for (i = 0; i < n; i++) val += maskt[i] * pixin[i];
          maskt += n;
        }
        *pixout = val;
      }
    }

    return RETURN_OK;
  }

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
static int svdfit_tol(double *a, double *b, int m, int n, double *sol,
                      double *vmat, double *wmat, double *rv1, double *tmp,
                      double tol) {
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
  thresh = tol * wmax;
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

static int svdfit(double *a, double *b, int m, int n, double *sol,
                  double *vmat, double *wmat, double *rv1, double *tmp) {
  return svdfit_tol(a, b, m, n, sol, vmat, wmat, rv1, tmp, SVD_TOL);
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

static int psf_flux_nnls(const double *ata, const double *atb, int n,
                         double *x) {
  double max_rhs = 0.0;
  double tol;
  int iter, i, j;

  for (i = 0; i < n; i++) {
    if (x[i] < 0.0 || !isfinite(x[i])) x[i] = 0.0;
    if (fabs(atb[i]) > max_rhs) max_rhs = fabs(atb[i]);
  }
  if (max_rhs < 1.0) max_rhs = 1.0;
  tol = PSF_FLUX_NNLS_TOL * max_rhs;

  for (iter = 0; iter < PSF_FLUX_NNLS_MAXITER; iter++) {
    double max_change = 0.0;

    for (i = 0; i < n; i++) {
      double diag = ata[i * n + i];
      double rhs = atb[i];
      double newx;

      if (!(diag > 0.0) || !isfinite(diag)) {
        x[i] = 0.0;
        continue;
      }

      for (j = 0; j < n; j++) {
        if (j == i) continue;
        rhs -= ata[i * n + j] * x[j];
      }
      newx = rhs / diag;
      if (!(newx > 0.0) || !isfinite(newx)) newx = 0.0;

      if (fabs(newx - x[i]) > max_change) max_change = fabs(newx - x[i]);
      x[i] = newx;
    }

    if (max_change <= tol) break;
  }

  return RETURN_OK;
}

/*==========================================================================*/
/*              PSF Model Rendering                                         */
/*==========================================================================*/

int sep_set_psf(void *arr, int dtype, int64_t w, int64_t h, sep_psf *psf,
                double x, double y, double flux) {
  int ix0, iy0;
  int sx, sy;
  int64_t imx, imy, pos;
  int status;
  double psfval, scale;
  float fval;
  float *farr;
  double *darr;

  if (!arr || !psf || w <= 0 || h <= 0) return ILLEGAL_APER_PARAMS;

  if (dtype != SEP_TFLOAT && dtype != SEP_TDOUBLE) return ILLEGAL_DTYPE;

  status = sep_psf_build(psf, x, y);
  if (status != RETURN_OK) return status;

  ix0 = (int)(x + 0.5);
  iy0 = (int)(y + 0.5);

  status = sep_psf_resample(psf, x - ix0, y - iy0);
  if (status != RETURN_OK) return status;

  psfval = 0.0;
  for (sx = 0; sx < psf->rw * psf->rh; sx++) psfval += psf->resi[sx];
  if (psfval <= 0.0) return RETURN_OK;

  scale = flux / psfval;

  if (dtype == SEP_TFLOAT) {
    farr = (float *)arr;
    for (sy = 0; sy < psf->rh; sy++) {
      imy = iy0 - psf->rh / 2 + sy;
      if (imy < 0 || imy >= h) continue;

      for (sx = 0; sx < psf->rw; sx++) {
        imx = ix0 - psf->rw / 2 + sx;
        if (imx < 0 || imx >= w) continue;

        fval = psf->resi[sy * psf->rw + sx];
        if (fval == 0.0f) continue;

        pos = imy * w + imx;
        farr[pos] += (float)(scale * (double)fval);
      }
    }
  } else {
    darr = (double *)arr;
    for (sy = 0; sy < psf->rh; sy++) {
      imy = iy0 - psf->rh / 2 + sy;
      if (imy < 0 || imy >= h) continue;

      for (sx = 0; sx < psf->rw; sx++) {
        imx = ix0 - psf->rw / 2 + sx;
        if (imx < 0 || imx >= w) continue;

        fval = psf->resi[sy * psf->rw + sx];
        if (fval == 0.0f) continue;

        pos = imy * w + imx;
        darr[pos] += scale * (double)fval;
      }
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
  double fit_r2, dx2, dy2;
  converter convert, econvert, mconvert, sconvert;
  int64_t size, esize, msize, ssize;
  const void *datat, *errort, *maskt, *segt;
  int errisarray, errisstd;

  *flag = 0;
  *sum = 0.0;
  *sumerr = 0.0;
  *area = 0.0;
  num = den = totarea = maskarea = 0.0;
  fit_r2 = (psf->fit_radius > 0.0) ? psf->fit_radius * psf->fit_radius : 0.0;
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

      /* Apply fit_radius cutoff */
      if (fit_r2 > 0.0) {
        dx2 = (double)imx - x;
        dy2 = (double)imy - y;
        if (dx2 * dx2 + dy2 * dy2 > fit_r2) continue;
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

static int psf_int_cmp(const void *a, const void *b) {
  int ia = *(const int *)a;
  int ib = *(const int *)b;
  return (ia > ib) - (ia < ib);
}

static int psf_int_contains(const int *arr, int n, int value) {
  int lo = 0;
  int hi = n - 1;

  while (lo <= hi) {
    int mid = lo + (hi - lo) / 2;
    int cur = arr[mid];
    if (cur < value) {
      lo = mid + 1;
    } else if (cur > value) {
      hi = mid - 1;
    } else {
      return 1;
    }
  }

  return 0;
}

static int psf_stamp_pixel_in_fit_radius(int sx, int sy, int width, int height,
                                         double dx, double dy,
                                         double fit_radius) {
  double fit_r2, cx, cy, fdx, fdy;

  if (fit_radius <= 0.0) return 1;

  fit_r2 = fit_radius * fit_radius;
  cx = (double)(width / 2) + dx;
  cy = (double)(height / 2) + dy;
  fdx = (double)sx - cx;
  fdy = (double)sy - cy;

  return fdx * fdx + fdy * fdy <= fit_r2;
}

/* Build compact weighted PSF and derivative columns for one grouped source.
 * The stamp is treated as zero outside its native support, outside the
 * clipped group bounding box, and outside fit_radius. */
static void psf_build_compact_columns(const float *stamp, const double *gweight,
                                      int width, int height, int gw, int gh,
                                      int ox, int oy, double dx, double dy,
                                      double fit_radius, double *cols) {
  int sx, sy;
  int npix = width * height;
  double *col0 = cols;
  double *col1 = cols + npix;
  double *col2 = cols + 2 * npix;

  for (sy = 0; sy < height; sy++) {
    int gy = oy + sy;
    int row = sy * width;

    for (sx = 0; sx < width; sx++) {
      int gx = ox + sx;
      int idx = row + sx;
      double w;
      float left, right, up, down;

      if (gx < 0 || gx >= gw || gy < 0 || gy >= gh) {
        col0[idx] = 0.0;
        col1[idx] = 0.0;
        col2[idx] = 0.0;
        continue;
      }

      w = gweight[gy * gw + gx];
      if (w == 0.0 ||
          !psf_stamp_pixel_in_fit_radius(sx, sy, width, height, dx, dy,
                                         fit_radius)) {
        col0[idx] = 0.0;
        col1[idx] = 0.0;
        col2[idx] = 0.0;
        continue;
      }

      left = (sx > 0 && gx > 0) ? stamp[idx - 1] : 0.0f;
      right = (sx + 1 < width && gx + 1 < gw) ? stamp[idx + 1] : 0.0f;
      up = (sy > 0 && gy > 0) ? stamp[idx - width] : 0.0f;
      down = (sy + 1 < height && gy + 1 < gh) ? stamp[idx + width] : 0.0f;

      col0[idx] = stamp[idx] * w;
      col1[idx] = ((double)right - (double)left) * w * 0.5;
      col2[idx] = ((double)down - (double)up) * w * 0.5;
    }
  }
}

static void psf_accum_source_atb(const double *cols, const double *gdata,
                                 int width, int height, int gw, int gh, int ox,
                                 int oy, double *atb) {
  int sx, sy;
  int npix = width * height;
  const double *col0 = cols;
  const double *col1 = cols + npix;
  const double *col2 = cols + 2 * npix;
  double s0 = 0.0, s1 = 0.0, s2 = 0.0;

  for (sy = 0; sy < height; sy++) {
    int gy = oy + sy;
    int row = sy * width;

    if (gy < 0 || gy >= gh) continue;

    for (sx = 0; sx < width; sx++) {
      int gx = ox + sx;
      int idx;
      double d;

      if (gx < 0 || gx >= gw) continue;
      idx = row + sx;
      d = gdata[gy * gw + gx];
      s0 += col0[idx] * d;
      s1 += col1[idx] * d;
      s2 += col2[idx] * d;
    }
  }

  atb[0] += s0;
  atb[1] += s1;
  atb[2] += s2;
}

static void psf_accum_pair_ata(const double *coli, int oxi, int oyi,
                               const double *colj, int oxj, int oyj, int width,
                               int height, int gw, int gh, double *ata,
                               int npar, int base_i, int base_j) {
  int gx0, gy0, gx1, gy1;
  int gx, gy;
  int npix = width * height;
  const double *ci0 = coli;
  const double *ci1 = coli + npix;
  const double *ci2 = coli + 2 * npix;
  const double *cj0 = colj;
  const double *cj1 = colj + npix;
  const double *cj2 = colj + 2 * npix;
  double s00 = 0.0, s01 = 0.0, s02 = 0.0;
  double s10 = 0.0, s11 = 0.0, s12 = 0.0;
  double s20 = 0.0, s21 = 0.0, s22 = 0.0;

  gx0 = oxi > oxj ? oxi : oxj;
  gy0 = oyi > oyj ? oyi : oyj;
  gx1 = (oxi + width) < (oxj + width) ? (oxi + width) : (oxj + width);
  gy1 = (oyi + height) < (oyj + height) ? (oyi + height) : (oyj + height);
  if (gx0 < 0) gx0 = 0;
  if (gy0 < 0) gy0 = 0;
  if (gx1 > gw) gx1 = gw;
  if (gy1 > gh) gy1 = gh;
  if (gx0 >= gx1 || gy0 >= gy1) return;

  for (gy = gy0; gy < gy1; gy++) {
    int row_i = (gy - oyi) * width;
    int row_j = (gy - oyj) * width;

    for (gx = gx0; gx < gx1; gx++) {
      int pi = row_i + (gx - oxi);
      int pj = row_j + (gx - oxj);

      s00 += ci0[pi] * cj0[pj];
      s01 += ci0[pi] * cj1[pj];
      s02 += ci0[pi] * cj2[pj];
      s10 += ci1[pi] * cj0[pj];
      s11 += ci1[pi] * cj1[pj];
      s12 += ci1[pi] * cj2[pj];
      s20 += ci2[pi] * cj0[pj];
      s21 += ci2[pi] * cj1[pj];
      s22 += ci2[pi] * cj2[pj];
    }
  }

  ata[(base_i + 0) * npar + (base_j + 0)] += s00;
  ata[(base_i + 0) * npar + (base_j + 1)] += s01;
  ata[(base_i + 0) * npar + (base_j + 2)] += s02;
  ata[(base_i + 1) * npar + (base_j + 0)] += s10;
  ata[(base_i + 1) * npar + (base_j + 1)] += s11;
  ata[(base_i + 1) * npar + (base_j + 2)] += s12;
  ata[(base_i + 2) * npar + (base_j + 0)] += s20;
  ata[(base_i + 2) * npar + (base_j + 1)] += s21;
  ata[(base_i + 2) * npar + (base_j + 2)] += s22;

  if (base_i != base_j) {
    ata[(base_j + 0) * npar + (base_i + 0)] += s00;
    ata[(base_j + 0) * npar + (base_i + 1)] += s10;
    ata[(base_j + 0) * npar + (base_i + 2)] += s20;
    ata[(base_j + 1) * npar + (base_i + 0)] += s01;
    ata[(base_j + 1) * npar + (base_i + 1)] += s11;
    ata[(base_j + 1) * npar + (base_i + 2)] += s21;
    ata[(base_j + 2) * npar + (base_i + 0)] += s02;
    ata[(base_j + 2) * npar + (base_i + 1)] += s12;
    ata[(base_j + 2) * npar + (base_i + 2)] += s22;
  }
}

static void psf_group_build_gmat(double *gmat, int gnpix, int gcount,
                                 const double *cols, const int *ixoff,
                                 const int *iyoff, int width, int height,
                                 int gw, int gh) {
  int i, sx, sy;
  int npix = width * height;
  int npar = gcount * PSF_NA;

  memset(gmat, 0, (size_t)gnpix * (size_t)npar * sizeof(double));

  for (i = 0; i < gcount; i++) {
    const double *col = cols + (size_t)i * PSF_NA * (size_t)npix;
    double *gcol0 = gmat + (size_t)(i * PSF_NA + 0) * gnpix;
    double *gcol1 = gmat + (size_t)(i * PSF_NA + 1) * gnpix;
    double *gcol2 = gmat + (size_t)(i * PSF_NA + 2) * gnpix;
    const double *col0 = col;
    const double *col1 = col + npix;
    const double *col2 = col + 2 * npix;
    int ox = ixoff[i];
    int oy = iyoff[i];

    for (sy = 0; sy < height; sy++) {
      int gy = oy + sy;
      int row = sy * width;

      if (gy < 0 || gy >= gh) continue;

      for (sx = 0; sx < width; sx++) {
        int gx = ox + sx;
        int gi, pi;

        if (gx < 0 || gx >= gw) continue;
        gi = gy * gw + gx;
        pi = row + sx;
        gcol0[gi] = col0[pi];
        gcol1[gi] = col1[pi];
        gcol2[gi] = col2[pi];
      }
    }
  }
}

static void psf_build_compact_flux_column(const float *psf_resi,
                                          const double *gweight, int width,
                                          int height, int gw, int gh, int ox,
                                          int oy, double dx, double dy,
                                          double fit_radius, double *col) {
  int sx, sy;

  memset(col, 0, (size_t)width * (size_t)height * sizeof(double));
  for (sy = 0; sy < height; sy++) {
    int gy = oy + sy;
    int row = sy * width;

    if (gy < 0 || gy >= gh) continue;

    for (sx = 0; sx < width; sx++) {
      int gx = ox + sx;
      if (!psf_stamp_pixel_in_fit_radius(sx, sy, width, height, dx, dy,
                                         fit_radius))
        continue;
      if (gx < 0 || gx >= gw) continue;
      col[row + sx] = (double)psf_resi[row + sx] * gweight[gy * gw + gx];
    }
  }
}

static double psf_accum_flux_atb(const double *col, const double *gdata,
                                 int width, int height, int gw, int gh,
                                 int ox, int oy) {
  int sx, sy;
  double sum = 0.0;

  for (sy = 0; sy < height; sy++) {
    int gy = oy + sy;
    int row = sy * width;

    if (gy < 0 || gy >= gh) continue;

    for (sx = 0; sx < width; sx++) {
      int gx = ox + sx;
      if (gx < 0 || gx >= gw) continue;
      sum += col[row + sx] * gdata[gy * gw + gx];
    }
  }

  return sum;
}

static void psf_accum_flux_pair_ata(const double *coli, int oxi, int oyi,
                                    const double *colj, int oxj, int oyj,
                                    int width, int height, int gw, int gh,
                                    double *ata, int npar, int i, int j) {
  int xstart, xend, ystart, yend;
  int sx, sy;
  double sum = 0.0;

  xstart = oxi > oxj ? oxi : oxj;
  ystart = oyi > oyj ? oyi : oyj;
  xend = oxi + width < oxj + width ? oxi + width : oxj + width;
  yend = oyi + height < oyj + height ? oyi + height : oyj + height;

  if (xstart < 0) xstart = 0;
  if (ystart < 0) ystart = 0;
  if (xend > gw) xend = gw;
  if (yend > gh) yend = gh;

  if (xstart >= xend || ystart >= yend) return;

  for (sy = ystart; sy < yend; sy++) {
    int row_i = (sy - oyi) * width;
    int row_j = (sy - oyj) * width;

    for (sx = xstart; sx < xend; sx++) {
      sum += coli[row_i + (sx - oxi)] * colj[row_j + (sx - oxj)];
    }
  }

  ata[i * npar + j] += sum;
  if (i != j) ata[j * npar + i] += sum;
}

static void psf_source_bbox(double x, double y, int width, int height,
                            int64_t *xmin, int64_t *xmax, int64_t *ymin,
                            int64_t *ymax) {
  int cix = (int)(x + 0.5);
  int ciy = (int)(y + 0.5);
  *xmin = (int64_t)cix - width / 2;
  *xmax = *xmin + width;
  *ymin = (int64_t)ciy - height / 2;
  *ymax = *ymin + height;
}

static int psf_bbox_overlap(int64_t xmin1, int64_t xmax1, int64_t ymin1,
                            int64_t ymax1, int64_t xmin2, int64_t xmax2,
                            int64_t ymin2, int64_t ymax2) {
  return xmin1 < xmax2 && xmax1 > xmin2 && ymin1 < ymax2 && ymax1 > ymin2;
}

static int psf_subset_contains(const int *gidx, int gcount, int idx) {
  int i;
  for (i = 0; i < gcount; i++) {
    if (gidx[i] == idx) return 1;
  }
  return 0;
}

static int psf_subtract_fixed_model(const sep_psf *psf_src, sep_psf *psf,
                                    double x, double y, double flux,
                                    int64_t gxmin, int64_t gymin, int gw, int gh,
                                    const double *gweight, double *gdata) {
  int status;
  int cix, ciy;
  int sx, sy;
  int width = psf_src->rw;
  int height = psf_src->rh;
  double psfsum = 0.0;
  double dx, dy;

  if (flux == 0.0) return RETURN_OK;

  status = sep_psf_build(psf, x, y);
  if (status != RETURN_OK) return status;

  cix = (int)(x + 0.5);
  ciy = (int)(y + 0.5);
  dx = x - cix;
  dy = y - ciy;
  status = sep_psf_resample(psf, dx, dy);
  if (status != RETURN_OK) return status;

  for (sx = 0; sx < width * height; sx++) psfsum += psf->resi[sx];
  if (psfsum <= 0.0) return RETURN_OK;

  for (sy = 0; sy < height; sy++) {
    int64_t imy = (int64_t)ciy - height / 2 + sy;
    if (imy < gymin || imy >= gymin + gh) continue;

    for (sx = 0; sx < width; sx++) {
      int64_t imx = (int64_t)cix - width / 2 + sx;
      int pidx;
      double weight;

      if (!psf_stamp_pixel_in_fit_radius(sx, sy, width, height, dx, dy,
                                         psf_src->fit_radius))
        continue;
      if (imx < gxmin || imx >= gxmin + gw) continue;
      pidx = (int)((imy - gymin) * gw + (imx - gxmin));
      weight = gweight[pidx];
      if (weight == 0.0) continue;
      gdata[pidx] -= flux * ((double)psf->resi[sy * width + sx] / psfsum) * weight;
    }
  }

  return RETURN_OK;
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
  double radmax2, fit_r2;
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
  fit_r2 = (psf->fit_radius > 0.0) ? psf->fit_radius * psf->fit_radius : 0.0;

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

      /* Apply fit_radius cutoff */
      if (fit_r2 > 0.0) {
        double fdx = (double)imx - x;
        double fdy = (double)imy - y;
        if (fdx * fdx + fdy * fdy > fit_r2) {
          data_vec[pidx] = 0.0;
          weight[pidx] = 0.0;
          continue;
        }
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

static int psf_fit_flux_subset(const sep_image *im, sep_psf *psf,
                               const double *x, const double *y,
                               const int *id, int gcount, const int *gidx,
                               const int *fixed_idx, int fixed_count,
                               const double *fixed_x, const double *fixed_y,
                               const double *fixed_flux,
                               psf_group_workspace *ws, double *pflux,
                               double *pfluxerr, short *pflag);
static int psf_compute_subset_chi2(const sep_image *im, sep_psf *psf,
                                   const double *x, const double *y,
                                   const int *id, int gcount, const int *gidx,
                                   const int *fixed_idx, int fixed_count,
                                   const double *fixed_x,
                                   const double *fixed_y,
                                   const double *fixed_flux,
                                   psf_group_workspace *ws,
                                   const double *pflux, double *pchi2);

static int psf_fit_subset(const sep_image *im, sep_psf *psf, const double *x,
                          const double *y, const int *id, short inflag,
                          int maxiter, int gcount, const int *gidx,
                          const int *fixed_idx, int fixed_count,
                          const double *fixed_x, const double *fixed_y,
                          const double *fixed_flux, psf_group_workspace *ws,
                          double *pflux, double *pfluxerr, double *pxfit,
                          double *pyfit, double *pxerr, double *pyerr,
                          int *pniter, double *pchi2, short *pflag) {
  int status = RETURN_OK;
  int i, j;
  double dx, dy;

  if (gcount == 1 && fixed_count == 0) {
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
    int stamp_npix = width * height;
    int64_t gxmin, gxmax, gymin, gymax;
    int gw, gh, gnpix;
    int iter, convflag, used_svd_last, used_svd_this;
    double *gdata, *gweight, *cols;
    double *gmat = NULL, *gsol, *gvmat, *gwmat;
    double *gcovmat, *grv1, *gtmp;
    double *deltax_arr, *deltay_arr;
    int *ixoff, *iyoff, *group_ids = NULL;
    converter convert, econvert, mconvert, sconvert;
    int64_t sz, esz, msz, ssz;
    int errisarray, errisstd;
    double vp;
    double radmax2 = (double)(width / 2) * (width / 2);

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

    status = psf_group_workspace_ensure(
        ws, gnpix, gcount, stamp_npix,
        (im->segmap && id) ? (gcount + fixed_count) : 0);
    if (status != RETURN_OK) return status;

    gdata = ws->gdata;
    gweight = ws->gweight;
    cols = ws->cols;
    gsol = ws->sol;
    gvmat = ws->ata;
    gwmat = ws->atb;
    gcovmat = ws->cov;
    grv1 = ws->rv1;
    gtmp = ws->tmp;
    deltax_arr = ws->deltax;
    deltay_arr = ws->deltay;
    ixoff = ws->ixoff;
    iyoff = ws->iyoff;

    memset(deltax_arr, 0, (size_t)gcount * sizeof(double));
    memset(deltay_arr, 0, (size_t)gcount * sizeof(double));

    /* Get converters */
    if ((status = get_converter(im->dtype, &convert, &sz))) return status;
    errisarray = 0;
    errisstd = 0;
    vp = 1.0;
    msz = 0;
    ssz = 0;
    esz = 0;
    if (im->mask) {
      if ((status = get_converter(im->mdtype, &mconvert, &msz))) return status;
    }
    if (im->segmap) {
      if ((status = get_converter(im->sdtype, &sconvert, &ssz))) return status;
    }
    if (im->noise_type != SEP_NOISE_NONE) {
      errisstd = (im->noise_type == SEP_NOISE_STDDEV);
      if (im->noise) {
        errisarray = 1;
        if ((status = get_converter(im->ndtype, &econvert, &esz))) return status;
      } else {
        vp = (errisstd) ? im->noiseval * im->noiseval : im->noiseval;
      }
    }

    if (im->segmap && id) {
      int nids = 0;
      group_ids = ws->group_ids;
      for (i = 0; i < gcount; i++) group_ids[nids++] = id[gidx[i]];
      for (i = 0; i < fixed_count; i++) group_ids[nids++] = id[fixed_idx[i]];
      qsort(group_ids, (size_t)nids, sizeof(int), psf_int_cmp);
    }

    /* Extract data and weights for group bounding box */
    {
      int64_t ix, iy;
      int64_t pos2;
      memset(gdata, 0, (size_t)gnpix * sizeof(double));
      memset(gweight, 0, (size_t)gnpix * sizeof(double));
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
              int in_group =
                  id ? psf_int_contains(group_ids, gcount + fixed_count, segval)
                     : 0;
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

    if (fixed_count > 0 && fixed_x && fixed_y && fixed_flux) {
      for (i = 0; i < fixed_count; i++) {
        int idx = fixed_idx[i];
        int64_t fxmin, fxmax, fymin, fymax;

        if (psf_subset_contains(gidx, gcount, idx)) continue;
        if (fixed_flux[idx] == 0.0) continue;

        psf_source_bbox(fixed_x[idx], fixed_y[idx], width, height, &fxmin, &fxmax,
                        &fymin, &fymax);
        if (!psf_bbox_overlap(gxmin, gxmax, gymin, gymax, fxmin, fxmax, fymin,
                              fymax))
          continue;

        status = psf_subtract_fixed_model(psf, psf, fixed_x[idx], fixed_y[idx],
                                          fixed_flux[idx], gxmin, gymin, gw, gh,
                                          gweight, gdata);
        if (status != RETURN_OK) return status;
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

    /* Initialize fluxes with a non-negative flux-only solve at fixed positions.
     * This gives the full grouped fit a stable starting point in close blends. */
    status = psf_fit_flux_subset(im, psf, x, y, id, gcount, gidx, fixed_idx,
                                 fixed_count, fixed_x, fixed_y, fixed_flux, ws,
                                 pflux, pfluxerr, pflag);
    if (status != RETURN_OK) return status;
    memset(deltax_arr, 0, (size_t)gcount * sizeof(double));
    memset(deltay_arr, 0, (size_t)gcount * sizeof(double));

    /* Iterative grouped fitting */
    for (iter = 0; iter < maxiter; iter++) {
      convflag = 0;
      memset(gvmat, 0, (size_t)npar * (size_t)npar * sizeof(double));
      memset(gwmat, 0, (size_t)npar * sizeof(double));

      /* Build compact columns for each group member at current positions. */
      for (i = 0; i < gcount; i++) {
        double *col = cols + (size_t)i * PSF_NA * (size_t)stamp_npix;
        int idx = gidx[i];
        double cx = x[idx] + deltax_arr[i];
        double cy = y[idx] + deltay_arr[i];
        int cix = (int)(cx + 0.5);
        int ciy = (int)(cy + 0.5);
        double psfsum;

        status = sep_psf_build(psf, cx, cy);
        if (status != RETURN_OK) return status;
        status = sep_psf_resample(psf, cx - cix, cy - ciy);
        if (status != RETURN_OK) return status;

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
        ixoff[i] = cix - width / 2 - (int)gxmin;
        iyoff[i] = ciy - height / 2 - (int)gymin;
        psf_build_compact_columns(psf->resi, gweight, width, height, gw, gh,
                                  ixoff[i], iyoff[i], cx - cix, cy - ciy,
                                  psf->fit_radius, col);
        psf_accum_source_atb(col, gdata, width, height, gw, gh, ixoff[i],
                             iyoff[i], gwmat + i * PSF_NA);
      }

      for (i = 0; i < gcount; i++) {
        const double *coli = cols + (size_t)i * PSF_NA * (size_t)stamp_npix;
        int base_i = i * PSF_NA;
        for (j = 0; j <= i; j++) {
          const double *colj = cols + (size_t)j * PSF_NA * (size_t)stamp_npix;
          int base_j = j * PSF_NA;
          psf_accum_pair_ata(coli, ixoff[i], iyoff[i], colj, ixoff[j], iyoff[j],
                             width, height, gw, gh, gvmat, npar, base_i,
                             base_j);
        }
      }

      /* Solve least squares: direct normal equations first, SVD only on fallback. */
      used_svd_this = 0;
      status = psf_cholesky_factor(gvmat, npar);
      if (status == RETURN_OK) {
        psf_cholesky_solve(gvmat, npar, gwmat, gsol);
      } else {
        status = psf_group_workspace_ensure_gmat(ws, gnpix, npar);
        if (status != RETURN_OK) return status;
        gmat = ws->gmat;
        psf_group_build_gmat(gmat, gnpix, gcount, cols, ixoff, iyoff, width,
                             height, gw, gh);
        status = svdfit_tol(gmat, gdata, gnpix, npar, gsol, gvmat, gwmat, grv1,
                            gtmp, PSF_GROUP_SVD_TOL);
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
        return status;
      }
      used_svd_last = used_svd_this;

      /* Update positions with damping for multi-component */
      for (i = 0; i < gcount; i++) {
        int idx = gidx[i];
        double prev_dx = deltax_arr[i];
        double prev_dy = deltay_arr[i];
        double fi = gsol[i * PSF_NA];
        if (fi < 0.0 || !isfinite(fi)) fi = 0.0;
        if (fi > 1.0e-12) {
          /* Factor of 2 damping for multi-component (SExtractor convention) */
          dx = -gsol[i * PSF_NA + 1] / (2.0 * fi);
          dy = -gsol[i * PSF_NA + 2] / (2.0 * fi);
        } else {
          dx = 0.0;
          dy = 0.0;
        }

        deltax_arr[i] = prev_dx + dx;
        deltay_arr[i] = prev_dy + dy;
        if (deltax_arr[i] * deltax_arr[i] + deltay_arr[i] * deltay_arr[i] >
            radmax2) {
          deltax_arr[i] = prev_dx;
          deltay_arr[i] = prev_dy;
          dx = 0.0;
          dy = 0.0;
        }
        if (!isfinite(deltax_arr[i]) || !isfinite(deltay_arr[i])) {
          deltax_arr[i] = prev_dx;
          deltay_arr[i] = prev_dy;
          fi = 0.0;
          dx = 0.0;
          dy = 0.0;
        }
        pflux[idx] = fi;

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

    /* Re-solve fluxes with NNLS at the converged positions. This keeps the
     * final flux partition non-negative and gives flux errors from the
     * reduced active-set covariance. */
    status = psf_fit_flux_subset(im, psf, pxfit, pyfit, id, gcount, gidx,
                                 fixed_idx, fixed_count, fixed_x, fixed_y,
                                 fixed_flux, ws, pflux, pfluxerr, pflag);
    if (status != RETURN_OK) return status;

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
        if (fabs(pflux[idx]) > 0.0) {
          double f2 = pflux[idx] * pflux[idx];
          double vx = gcovmat[(base + 1) * npar + (base + 1)];
          double vy = gcovmat[(base + 2) * npar + (base + 2)];
          pxerr[idx] = sqrt(vx > 0.0 ? vx / f2 : 0.0);
          pyerr[idx] = sqrt(vy > 0.0 ? vy / f2 : 0.0);
        } else {
          pxerr[idx] = 0.0;
          pyerr[idx] = 0.0;
        }
      }
    }

    if (status != RETURN_OK) return status;

    status = psf_compute_subset_chi2(im, psf, pxfit, pyfit, id, gcount, gidx,
                                     fixed_idx, fixed_count, fixed_x, fixed_y,
                                     fixed_flux, ws, pflux, pchi2);
  }

  return status;
}

static int psf_fit_flux_subset(const sep_image *im, sep_psf *psf,
                               const double *x, const double *y,
                               const int *id, int gcount, const int *gidx,
                               const int *fixed_idx, int fixed_count,
                               const double *fixed_x, const double *fixed_y,
                               const double *fixed_flux,
                               psf_group_workspace *ws, double *pflux,
                               double *pfluxerr, short *pflag) {
  int status = RETURN_OK;
  int i, j;
  int width = psf->rw;
  int height = psf->rh;
  int stamp_npix = width * height;
  int64_t gxmin, gxmax, gymin, gymax;
  int gw, gh, gnpix;
  double *gdata, *gweight, *cols;
  double *gsol, *gata, *gatb;
  int *ixoff, *iyoff, *group_ids = NULL;
  double *diagata, *active_l, *active_tmp;
  converter convert, econvert, mconvert, sconvert;
  int64_t sz, esz, msz, ssz;
  int errisarray, errisstd;
  double vp;
  int nactive;

  if (gcount <= 0) return RETURN_OK;

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

  if (gnpix < gcount || gw <= 0 || gh <= 0) return RETURN_OK;

  status = psf_group_workspace_ensure(
      ws, gnpix, gcount, stamp_npix, (im->segmap && id) ? (gcount + fixed_count)
                                                        : 0);
  if (status != RETURN_OK) return status;

  gdata = ws->gdata;
  gweight = ws->gweight;
  cols = ws->cols;
  gsol = ws->sol;
  gata = ws->ata;
  gatb = ws->atb;
  ixoff = ws->ixoff;
  iyoff = ws->iyoff;
  diagata = ws->deltax;
  active_l = ws->cov;
  active_tmp = ws->tmp;

  if ((status = get_converter(im->dtype, &convert, &sz))) return status;
  errisarray = 0;
  errisstd = 0;
  vp = 1.0;
  msz = 0;
  ssz = 0;
  esz = 0;
  if (im->mask) {
    if ((status = get_converter(im->mdtype, &mconvert, &msz))) return status;
  }
  if (im->segmap) {
    if ((status = get_converter(im->sdtype, &sconvert, &ssz))) return status;
  }
  if (im->noise_type != SEP_NOISE_NONE) {
    errisstd = (im->noise_type == SEP_NOISE_STDDEV);
    if (im->noise) {
      errisarray = 1;
      if ((status = get_converter(im->ndtype, &econvert, &esz))) return status;
    } else {
      vp = (errisstd) ? im->noiseval * im->noiseval : im->noiseval;
    }
  }

  if (im->segmap && id) {
    int nids = 0;
    group_ids = ws->group_ids;
    for (i = 0; i < gcount; i++) group_ids[nids++] = id[gidx[i]];
    for (i = 0; i < fixed_count; i++) group_ids[nids++] = id[fixed_idx[i]];
    qsort(group_ids, (size_t)nids, sizeof(int), psf_int_cmp);
  }

  {
    int64_t ix, iy;
    int64_t pos2;
    memset(gdata, 0, (size_t)gnpix * sizeof(double));
    memset(gweight, 0, (size_t)gnpix * sizeof(double));
    for (iy = gymin; iy < gymax; iy++) {
      for (ix = gxmin; ix < gxmax; ix++) {
        int pidx = (int)((iy - gymin) * gw + (ix - gxmin));
        double pix_val, var_val;

        pos2 = iy * im->w + ix;
        pix_val = ((converter)convert)((const char *)im->data + pos2 * sz);
        var_val = vp;

        if (im->mask) {
          if (((converter)mconvert)((const char *)im->mask + pos2 * msz) >
              im->maskthresh) {
            continue;
          }
        }

        if (im->segmap) {
          int segval =
              (int)((converter)sconvert)((const char *)im->segmap + pos2 * ssz);
          if (segval != 0) {
            int in_group =
                id ? psf_int_contains(group_ids, gcount + fixed_count, segval)
                   : 0;
            if (!in_group) continue;
          }
        }

        if (errisarray) {
          var_val = ((converter)econvert)((const char *)im->noise + pos2 * esz);
          if (errisstd) var_val *= var_val;
        }

        if (var_val > 0.0) {
          double total_var = var_val;
          if (im->gain > 0.0 && pix_val > 0.0) total_var += pix_val / im->gain;
          gweight[pidx] = 1.0 / sqrt(total_var);
          gdata[pidx] = pix_val * gweight[pidx];
        }
      }
    }
  }

  if (fixed_count > 0 && fixed_x && fixed_y && fixed_flux) {
    for (i = 0; i < fixed_count; i++) {
      int idx = fixed_idx[i];
      int64_t fxmin, fxmax, fymin, fymax;

      if (psf_subset_contains(gidx, gcount, idx)) continue;
      if (fixed_flux[idx] == 0.0) continue;

      psf_source_bbox(fixed_x[idx], fixed_y[idx], width, height, &fxmin, &fxmax,
                      &fymin, &fymax);
      if (!psf_bbox_overlap(gxmin, gxmax, gymin, gymax, fxmin, fxmax, fymin,
                            fymax))
        continue;

      status = psf_subtract_fixed_model(psf, psf, fixed_x[idx], fixed_y[idx],
                                        fixed_flux[idx], gxmin, gymin, gw, gh,
                                        gweight, gdata);
      if (status != RETURN_OK) return status;
    }
  }

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
        if (im->mask) {
          int64_t pos2 = imy2 * im->w + imx2;
          if (((converter)mconvert)((const char *)im->mask + pos2 * msz) >
              im->maskthresh) {
            pflag[idx] |= SEP_APER_HASMASKED;
          }
        }
      }
    }
  }

  memset(gata, 0, (size_t)gcount * (size_t)gcount * sizeof(double));
  memset(gatb, 0, (size_t)gcount * sizeof(double));
  for (i = 0; i < gcount; i++) {
    double *col = cols + (size_t)i * PSF_NA * (size_t)stamp_npix;
    int idx = gidx[i];
    int cix = (int)(x[idx] + 0.5);
    int ciy = (int)(y[idx] + 0.5);
    double psfsum = 0.0;

    status = sep_psf_build(psf, x[idx], y[idx]);
    if (status != RETURN_OK) return status;
    status = sep_psf_resample(psf, x[idx] - cix, y[idx] - ciy);
    if (status != RETURN_OK) return status;

    for (j = 0; j < stamp_npix; j++) psfsum += psf->resi[j];
    if (psfsum > 0.0) {
      for (j = 0; j < stamp_npix; j++) psf->resi[j] /= (float)psfsum;
    }
    ixoff[i] = cix - width / 2 - (int)gxmin;
    iyoff[i] = ciy - height / 2 - (int)gymin;
    psf_build_compact_flux_column(psf->resi, gweight, width, height, gw, gh,
                                  ixoff[i], iyoff[i], x[idx] - cix,
                                  y[idx] - ciy, psf->fit_radius, col);
    gatb[i] = psf_accum_flux_atb(col, gdata, width, height, gw, gh, ixoff[i],
                                 iyoff[i]);
  }

  for (i = 0; i < gcount; i++) {
    const double *coli = cols + (size_t)i * PSF_NA * (size_t)stamp_npix;
    for (j = 0; j <= i; j++) {
      const double *colj = cols + (size_t)j * PSF_NA * (size_t)stamp_npix;
      psf_accum_flux_pair_ata(coli, ixoff[i], iyoff[i], colj, ixoff[j],
                              iyoff[j], width, height, gw, gh, gata, gcount,
                              i, j);
    }
  }
  for (i = 0; i < gcount; i++) diagata[i] = gata[i * gcount + i];

  for (i = 0; i < gcount; i++) {
    int idx = gidx[i];
    gsol[i] = pflux[idx] > 0.0 ? pflux[idx] : 0.0;
  }
  status = psf_flux_nnls(gata, gatb, gcount, gsol);
  if (status != RETURN_OK) return status;

  nactive = 0;
  for (i = 0; i < gcount; i++) iyoff[i] = -1;
  for (i = 0; i < gcount; i++) {
    if (gsol[i] > 0.0) {
      iyoff[i] = nactive;
      ixoff[nactive++] = i;
    }
  }

  if (nactive > 0) {
    int ai, aj;

    for (ai = 0; ai < nactive; ai++) {
      for (aj = 0; aj < nactive; aj++) {
        active_l[ai * nactive + aj] =
            gata[ixoff[ai] * gcount + ixoff[aj]];
      }
    }

    status = psf_cholesky_factor(active_l, nactive);
    if (status == RETURN_OK) {
      status = psf_cholesky_inverse(active_l, nactive, gata, gatb, active_tmp);
      if (status != RETURN_OK) return status;
    } else {
      nactive = 0;
      status = RETURN_OK;
    }
  }

  for (i = 0; i < gcount; i++) {
    int idx = gidx[i];
    double var_f;
    pflux[idx] = gsol[i];
    var_f = 0.0;
    if (pflux[idx] > 0.0) {
      int ai = iyoff[i];
      if (nactive > 0 && ai >= 0) {
        var_f = gata[ai * nactive + ai];
      }
      if (var_f == 0.0 && diagata[i] > 0.0) {
        var_f = 1.0 / diagata[i];
      }
    }
    if (im->gain > 0.0 && pflux[idx] > 0.0) var_f += pflux[idx] / im->gain;
    pfluxerr[idx] = sqrt(var_f);
  }

  return RETURN_OK;
}

static int psf_compute_subset_chi2(const sep_image *im, sep_psf *psf,
                                   const double *x, const double *y,
                                   const int *id, int gcount, const int *gidx,
                                   const int *fixed_idx, int fixed_count,
                                   const double *fixed_x,
                                   const double *fixed_y,
                                   const double *fixed_flux,
                                   psf_group_workspace *ws,
                                   const double *pflux, double *pchi2) {
  int status = RETURN_OK;
  int i;
  int width = psf->rw;
  int height = psf->rh;
  int64_t gxmin, gxmax, gymin, gymax;
  int gw, gh, gnpix;
  double *gdata, *gweight;
  int *group_ids = NULL;
  converter convert, econvert, mconvert, sconvert;
  int64_t sz, esz, msz, ssz;
  int errisarray, errisstd;
  double vp;

  if (gcount <= 0) return RETURN_OK;

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

  for (i = 0; i < gcount; i++) pchi2[gidx[i]] = 0.0;
  if (gnpix <= 0 || gw <= 0 || gh <= 0) return RETURN_OK;

  status = psf_group_workspace_ensure(
      ws, gnpix, gcount, width * height, (im->segmap && id) ? (gcount + fixed_count)
                                                            : 0);
  if (status != RETURN_OK) return status;

  gdata = ws->gdata;
  gweight = ws->gweight;

  if ((status = get_converter(im->dtype, &convert, &sz))) return status;
  errisarray = 0;
  errisstd = 0;
  vp = 1.0;
  msz = 0;
  ssz = 0;
  esz = 0;
  if (im->mask) {
    if ((status = get_converter(im->mdtype, &mconvert, &msz))) return status;
  }
  if (im->segmap) {
    if ((status = get_converter(im->sdtype, &sconvert, &ssz))) return status;
  }
  if (im->noise_type != SEP_NOISE_NONE) {
    errisstd = (im->noise_type == SEP_NOISE_STDDEV);
    if (im->noise) {
      errisarray = 1;
      if ((status = get_converter(im->ndtype, &econvert, &esz))) return status;
    } else {
      vp = (errisstd) ? im->noiseval * im->noiseval : im->noiseval;
    }
  }

  if (im->segmap && id) {
    int nids = 0;
    group_ids = ws->group_ids;
    for (i = 0; i < gcount; i++) group_ids[nids++] = id[gidx[i]];
    for (i = 0; i < fixed_count; i++) group_ids[nids++] = id[fixed_idx[i]];
    qsort(group_ids, (size_t)nids, sizeof(int), psf_int_cmp);
  }

  {
    int64_t ix, iy;

    memset(gdata, 0, (size_t)gnpix * sizeof(double));
    memset(gweight, 0, (size_t)gnpix * sizeof(double));
    for (iy = gymin; iy < gymax; iy++) {
      for (ix = gxmin; ix < gxmax; ix++) {
        int pidx = (int)((iy - gymin) * gw + (ix - gxmin));
        int64_t pos2 = iy * im->w + ix;
        double pix_val = ((converter)convert)((const char *)im->data + pos2 * sz);
        double var_val = vp;

        if (im->mask) {
          if (((converter)mconvert)((const char *)im->mask + pos2 * msz) >
              im->maskthresh) {
            continue;
          }
        }

        if (im->segmap) {
          int segval =
              (int)((converter)sconvert)((const char *)im->segmap + pos2 * ssz);
          if (segval != 0) {
            int in_group =
                id ? psf_int_contains(group_ids, gcount + fixed_count, segval)
                   : 0;
            if (!in_group) continue;
          }
        }

        if (errisarray) {
          var_val = ((converter)econvert)((const char *)im->noise + pos2 * esz);
          if (errisstd) var_val *= var_val;
        }

        if (var_val > 0.0) {
          double total_var = var_val;
          if (im->gain > 0.0 && pix_val > 0.0) total_var += pix_val / im->gain;
          gweight[pidx] = 1.0 / sqrt(total_var);
          gdata[pidx] = pix_val * gweight[pidx];
        }
      }
    }
  }

  if (fixed_count > 0 && fixed_x && fixed_y && fixed_flux) {
    for (i = 0; i < fixed_count; i++) {
      int idx = fixed_idx[i];
      int64_t fxmin, fxmax, fymin, fymax;

      if (psf_subset_contains(gidx, gcount, idx)) continue;
      if (fixed_flux[idx] == 0.0) continue;

      psf_source_bbox(fixed_x[idx], fixed_y[idx], width, height, &fxmin, &fxmax,
                      &fymin, &fymax);
      if (!psf_bbox_overlap(gxmin, gxmax, gymin, gymax, fxmin, fxmax, fymin,
                            fymax))
        continue;

      status = psf_subtract_fixed_model(psf, psf, fixed_x[idx], fixed_y[idx],
                                        fixed_flux[idx], gxmin, gymin, gw, gh,
                                        gweight, gdata);
      if (status != RETURN_OK) return status;
    }
  }

  for (i = 0; i < gcount; i++) {
    int idx = gidx[i];

    status = psf_subtract_fixed_model(psf, psf, x[idx], y[idx], pflux[idx],
                                      gxmin, gymin, gw, gh, gweight, gdata);
    if (status != RETURN_OK) return status;
  }

  for (i = 0; i < gcount; i++) {
    int idx = gidx[i];
    int cix = (int)(x[idx] + 0.5);
    int ciy = (int)(y[idx] + 0.5);
    int sx, sy, ngood = 0;
    double chi2sum = 0.0;
    double dx = x[idx] - cix;
    double dy = y[idx] - ciy;

    for (sy = 0; sy < height; sy++) {
      int gy = ciy - height / 2 + sy;
      if (gy < gymin || gy >= gymax) continue;

      for (sx = 0; sx < width; sx++) {
        int gx = cix - width / 2 + sx;
        int pidx;
        double resid;

        if (!psf_stamp_pixel_in_fit_radius(sx, sy, width, height, dx, dy,
                                           psf->fit_radius))
          continue;
        if (gx < gxmin || gx >= gxmax) continue;
        pidx = (gy - gymin) * gw + (gx - gxmin);
        if (gweight[pidx] == 0.0) continue;
        resid = gdata[pidx];
        chi2sum += resid * resid;
        ngood++;
      }
    }

    if (ngood > PSF_NA) pchi2[idx] = chi2sum / (ngood - PSF_NA);
  }

  return RETURN_OK;
}

static int psf_fit_group_localized(const sep_image *im, sep_psf *psf,
                                   const double *x, const double *y,
                                   const int *id, double local_radius,
                                   short inflag, int maxiter,
                                   int fit_positions,
                                   int gcount,
                                   const int *gidx,
                                   psf_group_workspace *ws, double *pflux,
                                   double *pfluxerr, double *pxfit,
                                   double *pyfit, double *pxerr,
                                   double *pyerr, int *pniter, double *pchi2,
                                   short *pflag) {
  int status = RETURN_OK;
  int i, sweep, block_count;
  int refine_maxiter = maxiter;
  int max_idx = -1;
  size_t arr_len;
  double *work_flux = NULL, *tmp_flux = NULL, *tmp_fluxerr = NULL;
  short *tmp_flag = NULL;
  /* For flux-only mode, use original x/y for positions throughout;
   * for fit_positions mode, use fitted pxfit/pyfit. */
  const double *pos_x = fit_positions ? pxfit : x;
  const double *pos_y = fit_positions ? pyfit : y;

  if (refine_maxiter > 4) refine_maxiter = 4;

  for (i = 0; i < gcount; i++) {
    int idx = gidx[i];
    if (idx > max_idx) max_idx = idx;
    if (fit_positions) {
      status = sep_psf_fit(im, psf, x[idx], y[idx], id ? id[idx] : 0, inflag,
                           maxiter, &pflux[idx], &pfluxerr[idx], &pxfit[idx],
                           &pyfit[idx], &pxerr[idx], &pyerr[idx], &pniter[idx],
                           &pchi2[idx], &pflag[idx]);
    } else {
      double area;
      status = sep_sum_psf(im, psf, x[idx], y[idx], id ? id[idx] : 0, inflag,
                           &pflux[idx], &pfluxerr[idx], &area, &pflag[idx]);
    }
    if (status != RETURN_OK) return status;
  }

  arr_len = (size_t)max_idx + 1;
  work_flux = (double *)calloc(arr_len, sizeof(double));
  tmp_flux = (double *)calloc(arr_len, sizeof(double));
  tmp_fluxerr = (double *)calloc(arr_len, sizeof(double));
  tmp_flag = (short *)calloc(arr_len, sizeof(short));
  if (!work_flux || !tmp_flux || !tmp_fluxerr || !tmp_flag) {
    status = MEMORY_ALLOC_ERROR;
    goto cleanup;
  }

  for (i = 0; i < gcount; i++) {
    int idx = gidx[i];
    work_flux[idx] = pflux[idx];
  }

  block_count = (gcount + PSF_GROUP_CORE - 1) / PSF_GROUP_CORE;
  for (sweep = 0; sweep < PSF_GROUP_SWEEPS; sweep++) {
    int block;

    for (block = 0; block < block_count; block++) {
      int start = block * PSF_GROUP_CORE;
      int count = gcount - start;
      int left, right, active_count, k;
      int active_idx[PSF_GROUP_EXACT_MAX];
      int64_t core_xmin, core_xmax, core_ymin, core_ymax;
      int64_t ext_xmin, ext_xmax, ext_ymin, ext_ymax;
      if (count > PSF_GROUP_CORE) count = PSF_GROUP_CORE;

      core_xmin = (int64_t)im->w;
      core_xmax = 0;
      core_ymin = (int64_t)im->h;
      core_ymax = 0;
      for (i = 0; i < count; i++) {
        int idx = gidx[start + i];
        int64_t sxmin, sxmax, symin, symax;
        psf_source_bbox(pos_x[idx], pos_y[idx], psf->rw, psf->rh, &sxmin, &sxmax,
                        &symin, &symax);
        if (sxmin < core_xmin) core_xmin = sxmin;
        if (sxmax > core_xmax) core_xmax = sxmax;
        if (symin < core_ymin) core_ymin = symin;
        if (symax > core_ymax) core_ymax = symax;
      }
      ext_xmin = core_xmin - (int64_t)(local_radius + 0.5);
      ext_xmax = core_xmax + (int64_t)(local_radius + 0.5);
      ext_ymin = core_ymin - (int64_t)(local_radius + 0.5);
      ext_ymax = core_ymax + (int64_t)(local_radius + 0.5);

      left = start;
      while (left > 0 &&
             pos_x[gidx[left - 1]] + psf->rw / 2.0 >= (double)ext_xmin)
        left--;
      right = start + count;
      while (right < gcount &&
             pos_x[gidx[right]] - psf->rw / 2.0 <= (double)ext_xmax)
        right++;

      active_count = 0;
      for (k = left; k < right && active_count < PSF_GROUP_EXACT_MAX; k++) {
        int idx = gidx[k];
        int64_t sxmin, sxmax, symin, symax;
        psf_source_bbox(pos_x[idx], pos_y[idx], psf->rw, psf->rh, &sxmin, &sxmax,
                        &symin, &symax);
        if (!psf_bbox_overlap(ext_xmin, ext_xmax, ext_ymin, ext_ymax, sxmin,
                              sxmax, symin, symax))
          continue;
        active_idx[active_count++] = idx;
      }

      for (i = 0; i < active_count; i++) {
        int idx = active_idx[i];
        tmp_flux[idx] = work_flux[idx];
        tmp_fluxerr[idx] = pfluxerr[idx];
        tmp_flag[idx] = pflag[idx];
      }

      status = psf_fit_flux_subset(im, psf, pos_x, pos_y, id, active_count,
                                   active_idx, gidx, gcount, pos_x,
                                   pos_y, work_flux, ws, tmp_flux, tmp_fluxerr,
                                   tmp_flag);
      if (status != RETURN_OK) goto cleanup;

      for (i = 0; i < active_count; i++) {
        int idx = active_idx[i];
        work_flux[idx] = tmp_flux[idx];
      }
      for (i = 0; i < count; i++) {
        int idx = gidx[start + i];
        pflux[idx] = tmp_flux[idx];
        pfluxerr[idx] = tmp_fluxerr[idx];
        pflag[idx] = tmp_flag[idx];
      }
    }

    for (block = block_count - 1; block >= 0; block--) {
      int start = block * PSF_GROUP_CORE;
      int count = gcount - start;
      int left, right, active_count, k;
      int active_idx[PSF_GROUP_EXACT_MAX];
      int64_t core_xmin, core_xmax, core_ymin, core_ymax;
      int64_t ext_xmin, ext_xmax, ext_ymin, ext_ymax;
      if (count > PSF_GROUP_CORE) count = PSF_GROUP_CORE;

      core_xmin = (int64_t)im->w;
      core_xmax = 0;
      core_ymin = (int64_t)im->h;
      core_ymax = 0;
      for (i = 0; i < count; i++) {
        int idx = gidx[start + i];
        int64_t sxmin, sxmax, symin, symax;
        psf_source_bbox(pos_x[idx], pos_y[idx], psf->rw, psf->rh, &sxmin, &sxmax,
                        &symin, &symax);
        if (sxmin < core_xmin) core_xmin = sxmin;
        if (sxmax > core_xmax) core_xmax = sxmax;
        if (symin < core_ymin) core_ymin = symin;
        if (symax > core_ymax) core_ymax = symax;
      }
      ext_xmin = core_xmin - (int64_t)(local_radius + 0.5);
      ext_xmax = core_xmax + (int64_t)(local_radius + 0.5);
      ext_ymin = core_ymin - (int64_t)(local_radius + 0.5);
      ext_ymax = core_ymax + (int64_t)(local_radius + 0.5);

      left = start;
      while (left > 0 &&
             pos_x[gidx[left - 1]] + psf->rw / 2.0 >= (double)ext_xmin)
        left--;
      right = start + count;
      while (right < gcount &&
             pos_x[gidx[right]] - psf->rw / 2.0 <= (double)ext_xmax)
        right++;

      active_count = 0;
      for (k = left; k < right && active_count < PSF_GROUP_EXACT_MAX; k++) {
        int idx = gidx[k];
        int64_t sxmin, sxmax, symin, symax;
        psf_source_bbox(pos_x[idx], pos_y[idx], psf->rw, psf->rh, &sxmin, &sxmax,
                        &symin, &symax);
        if (!psf_bbox_overlap(ext_xmin, ext_xmax, ext_ymin, ext_ymax, sxmin,
                              sxmax, symin, symax))
          continue;
        active_idx[active_count++] = idx;
      }

      for (i = 0; i < active_count; i++) {
        int idx = active_idx[i];
        tmp_flux[idx] = work_flux[idx];
        tmp_fluxerr[idx] = pfluxerr[idx];
        tmp_flag[idx] = pflag[idx];
      }

      status = psf_fit_flux_subset(im, psf, pos_x, pos_y, id, active_count,
                                   active_idx, gidx, gcount, pos_x,
                                   pos_y, work_flux, ws, tmp_flux, tmp_fluxerr,
                                   tmp_flag);
      if (status != RETURN_OK) goto cleanup;

      for (i = 0; i < active_count; i++) {
        int idx = active_idx[i];
        work_flux[idx] = tmp_flux[idx];
      }
      for (i = 0; i < count; i++) {
        int idx = gidx[start + i];
        pflux[idx] = tmp_flux[idx];
        pfluxerr[idx] = tmp_fluxerr[idx];
        pflag[idx] = tmp_flag[idx];
      }
    }
  }

  if (fit_positions) {
    for (i = 0; i < gcount; i++) {
      status = psf_fit_subset(im, psf, pxfit, pyfit, id, inflag, refine_maxiter,
                              1, gidx + i, gidx, gcount, pxfit, pyfit, pflux, ws,
                              pflux, pfluxerr, pxfit, pyfit, pxerr, pyerr, pniter,
                              pchi2, pflag);
      if (status != RETURN_OK) goto cleanup;
    }
  }

cleanup:
  free(work_flux);
  free(tmp_flux);
  free(tmp_fluxerr);
  free(tmp_flag);
  return status;
}

/*==========================================================================*/
/*                     Grouped PSF Fitting                                  */
/*==========================================================================*/

int sep_psf_fit_multi(const sep_image *im, sep_psf *psf, const double *x,
                      const double *y, int64_t n, const int *id,
                      double group_factor, short inflag, int maxiter,
                      int fit_positions,
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
  double half_stamp, link_rsum, local_rsum, rsum2, dx, dy, dist2;

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
    pchi2[i] = fit_positions ? 0.0 : NAN;
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
  local_rsum = group_factor * (half_stamp + half_stamp);
  link_rsum = local_rsum;
  if (link_rsum > half_stamp + half_stamp) link_rsum = half_stamp + half_stamp;
  rsum2 = link_rsum * link_rsum;
  if (link_rsum > 0.0) {
    for (i = 0; i < n; i++) {
      int ii = xorder[i].idx;
      double xi = xorder[i].x;
      for (j = i + 1; j < n; j++) {
        int jj = xorder[j].idx;
        dx = xorder[j].x - xi;
        if (dx > link_rsum) break;
        dy = y[ii] - y[jj];
        if (dy > link_rsum || dy < -link_rsum) continue;
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
    int idx = xorder[i].idx;
    int gid = group_id[idx];
    members[group_fill[gid]++] = idx;
  }

  has_multi = 0;
  for (i = 0; i < ngroups; i++) {
    if (group_counts[i] > 1) {
      has_multi = 1;
      break;
    }
  }

  if (!has_multi) {
    if (fit_positions) {
      status = sep_psf_fit_array(im, psf, x, y, n, id, inflag, maxiter, pflux,
                                 pfluxerr, pxfit, pyfit, pxerr, pyerr, pniter,
                                 pchi2, pflag);
    } else {
      /* Flux-only: individual matched-filter for each source */
      for (i = 0; i < n; i++) {
        double area;
        status = sep_sum_psf(im, psf, x[i], y[i], id ? id[i] : 0, inflag,
                             &pflux[i], &pfluxerr[i], &area, &pflag[i]);
        if (status != RETURN_OK) goto cleanup;
      }
    }
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
      psf_group_workspace group_ws;
      sep_psf *work_psf = psf;
      int ws_status = psf_workspace_clone(psf, &local_psf);

      psf_group_workspace_nullify(&group_ws);

      if (ws_status == RETURN_OK) {
        work_psf = &local_psf;
      } else {
#pragma omp critical(psf_multi_status)
        {
          if (first_status == RETURN_OK) first_status = ws_status;
        }
      }

#pragma omp for schedule(guided, 1)
      for (int go = 0; go < ngroups; go++) {
        int gid = group_order[go].gid;
        int gstatus;
        if (ws_status != RETURN_OK) continue;
        {
          int gstart = group_offsets[gid];
          int gcount = group_counts[gid];
          const int *gidx = members + gstart;

          if (!fit_positions) {
            /* Flux-only grouped path */
            if (gcount == 1) {
              int idx0 = gidx[0];
              double area;
              gstatus = sep_sum_psf(im, work_psf, x[idx0], y[idx0],
                                    id ? id[idx0] : 0, inflag,
                                    &pflux[idx0], &pfluxerr[idx0], &area,
                                    &pflag[idx0]);
            } else if (gcount <= PSF_GROUP_EXACT_MAX) {
              gstatus = psf_fit_flux_subset(im, work_psf, x, y, id,
                                            gcount, gidx, NULL, 0,
                                            NULL, NULL, NULL,
                                            &group_ws, pflux, pfluxerr,
                                            pflag);
            } else {
              gstatus = psf_fit_group_localized(
                  im, work_psf, x, y, id, local_rsum, inflag, maxiter,
                  0, gcount, gidx,
                  &group_ws, pflux, pfluxerr, pxfit, pyfit, pxerr, pyerr,
                  pniter, pchi2, pflag);
            }
          } else if (gcount <= PSF_GROUP_EXACT_MAX) {
            gstatus = psf_fit_subset(im, work_psf, x, y, id, inflag, maxiter,
                                     gcount, gidx, NULL, 0, NULL, NULL, NULL,
                                     &group_ws, pflux, pfluxerr, pxfit, pyfit,
                                     pxerr, pyerr, pniter, pchi2, pflag);
          } else {
            gstatus = psf_fit_group_localized(
                im, work_psf, x, y, id, local_rsum, inflag, maxiter,
                fit_positions, gcount, gidx,
                &group_ws, pflux, pfluxerr, pxfit, pyfit, pxerr, pyerr, pniter,
                pchi2, pflag);
          }
        }
        if (gstatus != RETURN_OK) {
#pragma omp critical(psf_multi_status)
          {
            if (first_status == RETURN_OK) first_status = gstatus;
          }
        }
      }

      psf_group_workspace_free(&group_ws);
      psf_workspace_free(&local_psf);
    }

    if (first_status != RETURN_OK) {
      status = first_status;
      goto cleanup;
    }
  }
#else
  psf_group_workspace group_ws;
  psf_group_workspace_nullify(&group_ws);
  for (i = 0; i < ngroups; i++) {
    int gstart, gcount;
    const int *gidx;

    g = group_order[i].gid;
    gstart = group_offsets[g];
    gcount = group_counts[g];
    gidx = members + gstart;
    if (!fit_positions) {
      /* Flux-only grouped path */
      if (gcount == 1) {
        int idx0 = gidx[0];
        double area;
        status = sep_sum_psf(im, psf, x[idx0], y[idx0],
                             id ? id[idx0] : 0, inflag,
                             &pflux[idx0], &pfluxerr[idx0], &area,
                             &pflag[idx0]);
      } else if (gcount <= PSF_GROUP_EXACT_MAX) {
        status = psf_fit_flux_subset(im, psf, x, y, id,
                                     gcount, gidx, NULL, 0,
                                     NULL, NULL, NULL,
                                     &group_ws, pflux, pfluxerr, pflag);
      } else {
        status = psf_fit_group_localized(
            im, psf, x, y, id, local_rsum, inflag, maxiter,
            0, gcount, gidx,
            &group_ws, pflux, pfluxerr, pxfit, pyfit, pxerr, pyerr,
            pniter, pchi2, pflag);
      }
    } else if (gcount <= PSF_GROUP_EXACT_MAX) {
      status = psf_fit_subset(im, psf, x, y, id, inflag, maxiter, gcount, gidx,
                              NULL, 0, NULL, NULL, NULL, &group_ws, pflux,
                              pfluxerr, pxfit, pyfit, pxerr, pyerr, pniter,
                              pchi2, pflag);
    } else {
      status = psf_fit_group_localized(
          im, psf, x, y, id, local_rsum, inflag, maxiter,
          fit_positions, gcount, gidx,
          &group_ws, pflux, pfluxerr, pxfit, pyfit, pxerr, pyerr, pniter,
          pchi2, pflag);
    }
    if (status != RETURN_OK) {
      psf_group_workspace_free(&group_ws);
      goto cleanup;
    }
  }
  psf_group_workspace_free(&group_ws);
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
