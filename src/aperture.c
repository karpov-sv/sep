/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 *
 * This file is part of SEP
 *
 * Copyright 1993-2011 Emmanuel Bertin -- IAP/CNRS/UPMC
 * Copyright 2014 SEP developers
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
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "overlap.h"
#include "sep.h"
#include "sepcore.h"

#define FLUX_RADIUS_BUFSIZE 64

#define WINPOS_NITERMAX 16 /* Maximum number of steps */
#define WINPOS_NSIG 4 /* Measurement radius */
#define WINPOS_STEPMIN 0.0001 /* Minimum change in position for continuing */
#define WINPOS_FAC 2.0 /* Centroid offset factor (2 for a Gaussian) */

/*
  Adding (void *) pointers is a GNU C extension, not part of standard C.
  When compiling on Windows with MS VIsual C compiler need to cast the
  (void *) to something the size of one byte.
*/
#if defined(_MSC_VER)
#define MSVC_VOID_CAST (char *)
#else
#define MSVC_VOID_CAST
#endif

/****************************************************************************/
/* conversions between ellipse representations */

/* return ellipse semi-major and semi-minor axes and position angle, given
   representation: cxx*x^2 + cyy*y^2 + cxy*x*y = 1
   derived from http://mathworld.wolfram.com/Ellipse.html

   Input requirements:
   cxx*cyy - cxy*cxy/4. > 0.
   cxx + cyy > 0.
*/
int sep_ellipse_axes(
    double cxx, double cyy, double cxy, double * a, double * b, double * theta
) {
  double p, q, t;

  p = cxx + cyy;
  q = cxx - cyy;
  t = sqrt(q * q + cxy * cxy);

  /* Ensure that parameters actually describe an ellipse. */
  if ((cxx * cyy - cxy * cxy / 4. <= 0.) || (p <= 0.)) {
    return NON_ELLIPSE_PARAMS;
  }

  *a = sqrt(2. / (p - t));
  *b = sqrt(2. / (p + t));

  /* theta = 0 if cxy == 0, else (1/2) acot(q/cxy) */
  *theta = (cxy == 0.) ? 0. : (q == 0. ? 0. : atan(cxy / q)) / 2.;
  if (cxx > cyy) {
    *theta += PI / 2.;
  }
  if (*theta > PI / 2.) {
    *theta -= PI;
  }

  return RETURN_OK;
}

void sep_ellipse_coeffs(
    double a, double b, double theta, double * cxx, double * cyy, double * cxy
) {
  double costheta, sintheta;

  costheta = cos(theta);
  sintheta = sin(theta);

  *cxx = costheta * costheta / (a * a) + sintheta * sintheta / (b * b);
  *cyy = sintheta * sintheta / (a * a) + costheta * costheta / (b * b);
  *cxy = 2. * costheta * sintheta * (1. / (a * a) - 1. / (b * b));
}

/*****************************************************************************/
/* Helper functions for aperture functions */

/* determine the extent of the box that just contains the circle with
 * parameters x, y, r. xmin is inclusive and xmax is exclusive.
 * Ensures that box is within image bound and sets a flag if it is not.
 */
static void boxextent(
    double x,
    double y,
    double rx,
    double ry,
    int64_t w,
    int64_t h,
    int64_t * xmin,
    int64_t * xmax,
    int64_t * ymin,
    int64_t * ymax,
    short * flag
) {
  *xmin = (int64_t)(x - rx + 0.5);
  *xmax = (int64_t)(x + rx + 1.4999999);
  *ymin = (int64_t)(y - ry + 0.5);
  *ymax = (int64_t)(y + ry + 1.4999999);
  if (*xmin < 0) {
    *xmin = 0;
    *flag |= SEP_APER_TRUNC;
  }
  if (*xmax > w) {
    *xmax = w;
    *flag |= SEP_APER_TRUNC;
  }
  if (*ymin < 0) {
    *ymin = 0;
    *flag |= SEP_APER_TRUNC;
  }
  if (*ymax > h) {
    *ymax = h;
    *flag |= SEP_APER_TRUNC;
  }
}


static void boxextent_ellipse(
    double x,
    double y,
    double cxx,
    double cyy,
    double cxy,
    double r,
    int64_t w,
    int64_t h,
    int64_t * xmin,
    int64_t * xmax,
    int64_t * ymin,
    int64_t * ymax,
    short * flag
) {
  double dxlim, dylim;

  dxlim = cxx - cxy * cxy / (4.0 * cyy);
  dxlim = dxlim > 0.0 ? r / sqrt(dxlim) : 0.0;
  dylim = cyy - cxy * cxy / (4.0 * cxx);
  dylim = dylim > 0.0 ? r / sqrt(dylim) : 0.0;
  boxextent(x, y, dxlim, dylim, w, h, xmin, xmax, ymin, ymax, flag);
}

/* determine oversampled annulus for a circle */
static void oversamp_ann_circle(double r, double * r_in2, double * r_out2) {
  *r_in2 = r - 0.7072;
  *r_in2 = (*r_in2 > 0.0) ? (*r_in2) * (*r_in2) : 0.0;
  *r_out2 = r + 0.7072;
  *r_out2 = (*r_out2) * (*r_out2);
}

/* determine oversampled "annulus" for an ellipse */
static void oversamp_ann_ellipse(double r, double b, double * r_in2, double * r_out2) {
  *r_in2 = r - 0.7072 / b;
  *r_in2 = (*r_in2 > 0.0) ? (*r_in2) * (*r_in2) : 0.0;
  *r_out2 = r + 0.7072 / b;
  *r_out2 = (*r_out2) * (*r_out2);
}

typedef struct {
  double v;
  double w;
} valweight;

static int cmp_valweight(const void *a, const void *b) {
  double va = ((const valweight *)a)->v;
  double vb = ((const valweight *)b)->v;
  if (va < vb) {
    return -1;
  }
  if (va > vb) {
    return 1;
  }
  return 0;
}

static double weighted_median_sorted(const valweight *arr, int64_t n, double totw) {
  double acc = 0.0;
  int64_t i;

  if (n == 0 || totw <= 0.0) {
    return NAN;
  }

  for (i = 0; i < n; i++) {
    acc += arr[i].w;
    if (acc >= 0.5 * totw) {
      return arr[i].v;
    }
  }

  return arr[n - 1].v;
}

static double gaussian_pixel_integral(double dx, double dy, double sigma) {
  double inv = 1.0 / (sqrt(2.0) * sigma);
  double ex = erf((dx + 0.5) * inv) - erf((dx - 0.5) * inv);
  double ey = erf((dy + 0.5) * inv) - erf((dy - 0.5) * inv);
  return 0.25 * ex * ey;
}

static int cholesky_decomp(double *a, int n) {
  int i, j, k;

  for (i = 0; i < n; i++) {
    for (j = 0; j <= i; j++) {
      double sum = a[i * n + j];
      for (k = 0; k < j; k++) {
        sum -= a[i * n + k] * a[j * n + k];
      }
      if (i == j) {
        if (sum <= 0.0) {
          return 1;
        }
        a[i * n + i] = sqrt(sum);
      } else {
        a[i * n + j] = sum / a[j * n + j];
      }
    }
    for (j = i + 1; j < n; j++) {
      a[i * n + j] = 0.0;
    }
  }

  return 0;
}

static void cholesky_solve(const double *L, const double *b, double *x, int n,
                           double *work) {
  int i, k;

  /* forward solve: L * y = b */
  for (i = 0; i < n; i++) {
    double sum = b[i];
    for (k = 0; k < i; k++) {
      sum -= L[i * n + k] * work[k];
    }
    work[i] = sum / L[i * n + i];
  }

  /* backward solve: L^T * x = y */
  for (i = n - 1; i >= 0; i--) {
    double sum = work[i];
    for (k = i + 1; k < n; k++) {
      sum -= L[k * n + i] * x[k];
    }
    x[i] = sum / L[i * n + i];
  }
}

static int uf_find(int *parent, int i) {
  while (parent[i] != i) {
    parent[i] = parent[parent[i]];
    i = parent[i];
  }
  return i;
}

static void uf_union(int *parent, int *rank, int a, int b) {
  int ra = uf_find(parent, a);
  int rb = uf_find(parent, b);
  if (ra == rb) {
    return;
  }
  if (rank[ra] < rank[rb]) {
    parent[ra] = rb;
  } else if (rank[ra] > rank[rb]) {
    parent[rb] = ra;
  } else {
    parent[rb] = ra;
    rank[ra] += 1;
  }
}

/*****************************************************************************/
/* circular aperture */

#define APER_NAME sep_sum_circle
#define APER_ARGS double r
#define APER_DECL double r2, r_in2, r_out2
#define APER_CHECKS \
  if (r < 0.0)      \
  return ILLEGAL_APER_PARAMS
#define APER_INIT \
  r2 = r * r;     \
  oversamp_ann_circle(r, &r_in2, &r_out2)
#define APER_BOXEXTENT \
  boxextent(x, y, r, r, im->w, im->h, &xmin, &xmax, &ymin, &ymax, flag)
#define APER_EXACT circoverlap(dx - 0.5, dy - 0.5, dx + 0.5, dy + 0.5, r)
#define APER_RPIX2 dx * dx + dy * dy
#define APER_RPIX2_SUBPIX dx1 * dx1 + dy2
#define APER_COMPARE1 rpix2 < r_out2
#define APER_COMPARE2 rpix2 > r_in2
#define APER_COMPARE3 rpix2 < r2
#include "aperture.i"
#undef APER_NAME
#undef APER_ARGS
#undef APER_DECL
#undef APER_CHECKS
#undef APER_INIT
#undef APER_BOXEXTENT
#undef APER_EXACT
#undef APER_RPIX2
#undef APER_RPIX2_SUBPIX
#undef APER_COMPARE1
#undef APER_COMPARE2
#undef APER_COMPARE3

/*****************************************************************************/
/* circular aperture with optimal extraction */

int sep_sum_circle_optimal(
    const sep_image * im,
    double x,
    double y,
    double r,
    double fwhm,
    int id,
    int subpix,
    short inflag,
    double * sum,
    double * sumerr,
    double * area,
    short * flag
) {
  PIXTYPE pix, varpix;
  double dx, dy, dx1, dy2, offset, scale, scale2, rpix2, overlap;
  double r2, r_in2, r_out2, sigma, num, den, totarea, maskarea;
  double psf, var;
  int64_t ix, iy, xmin, xmax, ymin, ymax, sx, sy, pos, size, esize, msize, ssize;
  int ismasked, status;
  short errisarray, errisstd;
  const BYTE *datat, *errort, *maskt, *segt;
  converter convert, econvert = NULL, mconvert, sconvert;

  if (r < 0.0 || !(fwhm > 0.0)) {
    return ILLEGAL_APER_PARAMS;
  }
  if (subpix < 0) {
    return ILLEGAL_SUBPIX;
  }

  size = esize = msize = ssize = 0;
  num = den = totarea = maskarea = 0.0;
  datat = maskt = segt = NULL;
  errort = im->noise;
  *flag = 0;
  varpix = 1.0;

  if (subpix > 0) {
    scale = 1.0 / subpix;
    scale2 = scale * scale;
    offset = 0.5 * (scale - 1.0);
  } else {
    scale = 0.0;
    scale2 = 0.0;
    offset = 0.0;
  }

  sigma = fwhm / 2.354820045;
  if (!(sigma > 0.0)) {
    return ILLEGAL_APER_PARAMS;
  }

  r2 = r * r;
  oversamp_ann_circle(r, &r_in2, &r_out2);

  if ((status = get_converter(im->dtype, &convert, &size))) {
    return status;
  }
  if (im->mask && (status = get_converter(im->mdtype, &mconvert, &msize))) {
    return status;
  }
  if (im->segmap && (status = get_converter(im->sdtype, &sconvert, &ssize))) {
    return status;
  }

  errisarray = 0;
  errisstd = 0;
  if (im->noise_type != SEP_NOISE_NONE) {
    errisstd = (im->noise_type == SEP_NOISE_STDDEV);
    if (im->noise) {
      errisarray = 1;
      if ((status = get_converter(im->ndtype, &econvert, &esize))) {
        return status;
      }
    } else {
      varpix = (errisstd) ? im->noiseval * im->noiseval : im->noiseval;
    }
  }

  boxextent(x, y, r, r, im->w, im->h, &xmin, &xmax, &ymin, &ymax, flag);

  for (iy = ymin; iy < ymax; iy++) {
    pos = (iy % im->h) * im->w + xmin;
    datat = MSVC_VOID_CAST im->data + pos * size;
    if (errisarray) {
      errort = MSVC_VOID_CAST im->noise + pos * esize;
    }
    if (im->mask) {
      maskt = MSVC_VOID_CAST im->mask + pos * msize;
    }
    if (im->segmap) {
      segt = MSVC_VOID_CAST im->segmap + pos * ssize;
    }

    for (ix = xmin; ix < xmax; ix++) {
      dx = ix - x;
      dy = iy - y;
      double dx0 = dx;
      double dy0 = dy;
      rpix2 = dx * dx + dy * dy;
      if (rpix2 < r_out2) {
        if (rpix2 > r_in2) {
          if (subpix == 0) {
            overlap = circoverlap(dx - 0.5, dy - 0.5, dx + 0.5, dy + 0.5, r);
          } else {
            dx += offset;
            dy += offset;
            overlap = 0.0;
            for (sy = subpix; sy--; dy += scale) {
              dx1 = dx;
              dy2 = dy * dy;
              for (sx = subpix; sx--; dx1 += scale) {
                if (dx1 * dx1 + dy2 < r2) {
                  overlap += scale2;
                }
              }
            }
          }
        } else {
          overlap = 1.0;
        }

        pix = convert(datat);
        if (errisarray) {
          varpix = econvert(errort);
          if (errisstd) {
            varpix *= varpix;
          }
        }

        ismasked = 0;
        if (im->mask && (mconvert(maskt) > im->maskthresh)) {
          ismasked = 1;
        }

        if (im->segmap) {
          if (id > 0) {
            if ((sconvert(segt) > 0.) && (sconvert(segt) != id)) {
              ismasked = 1;
            }
          } else {
            if (sconvert(segt) != -1 * id) {
              ismasked = 1;
            }
          }
        }

        if (ismasked) {
          *flag |= SEP_APER_HASMASKED;
          maskarea += overlap;
        } else {
          if (varpix > 0.0) {
            psf = gaussian_pixel_integral(dx0, dy0, sigma) * overlap;
            num += psf * pix / varpix;
            den += psf * psf / varpix;
          } else {
            *flag |= SEP_APER_HASMASKED;
            maskarea += overlap;
          }
        }

        totarea += overlap;
      }

      datat += size;
      if (errisarray) {
        errort += esize;
      }
      maskt += msize;
      segt += ssize;
    }
  }

  if (im->mask) {
    if (totarea > 0.0 && maskarea >= totarea) {
      *flag |= SEP_APER_ALLMASKED;
      *sum = 0.0;
      *sumerr = 0.0;
      *area = 0.0;
      return status;
    } else if (inflag & SEP_MASK_IGNORE) {
      totarea -= maskarea;
    }
  }

  if (den <= 0.0) {
    *flag |= SEP_APER_ALLMASKED;
    *sum = 0.0;
    *sumerr = 0.0;
    *area = 0.0;
    return status;
  }

  *sum = num / den;
  var = 1.0 / den;
  if (im->gain > 0.0 && *sum > 0.0) {
    var += (*sum) / im->gain;
  }
  *sumerr = sqrt(var);
  *area = totarea;

  return status;
}

/*****************************************************************************/
/* circular aperture optimal extraction with auto-grouping */

static int sep_sum_circle_optimal_multi_impl(
    const sep_image * im,
    const double * x,
    const double * y,
    const double * r,
    const double * fwhm,
    int64_t n,
    const int * id,
    int subpix,
    short inflag,
    const double * bkg_mean,
    const double * bkg_mean_err,
    const double * bkg_weight,
    double * sum,
    double * sumerr,
    double * area,
    short * flag
) {
  PIXTYPE pix, varpix;
  double dx, dy, dx1, dy2, offset, scale, scale2, rpix2, overlap;
  double tmp, var, dist2, rsum, sigma;
  int64_t ix, iy, xmin, xmax, ymin, ymax, sx, sy, pos;
  int64_t size, esize, msize, ssize;
  int i, j, g, gi;
  int status, ismasked;
  short errisarray, errisstd;
  const BYTE *datat, *errort, *maskt, *segt;
  converter convert, econvert = NULL, mconvert, sconvert;
  int *parent, *rank, *group_id, *root_map, *group_counts, *group_offsets, *group_fill;
  int *members;
  double *r2, *r_in2, *r_out2, *sigma_arr;
  double *M, *b, *work, *sol, *totarea, *maskarea, *ai;
  int use_bkg;

  if (n < 1) {
    return ILLEGAL_APER_PARAMS;
  }
  if (subpix < 0) {
    return ILLEGAL_SUBPIX;
  }

  size = esize = msize = ssize = 0;
  datat = maskt = segt = NULL;
  errort = im->noise;
  errisarray = 0;
  errisstd = 0;
  use_bkg = (bkg_mean != NULL);

  if ((status = get_converter(im->dtype, &convert, &size))) {
    return status;
  }
  if (im->mask && (status = get_converter(im->mdtype, &mconvert, &msize))) {
    return status;
  }
  if (im->segmap && (status = get_converter(im->sdtype, &sconvert, &ssize))) {
    return status;
  }

  if (im->noise_type != SEP_NOISE_NONE) {
    errisstd = (im->noise_type == SEP_NOISE_STDDEV);
    if (im->noise) {
      errisarray = 1;
      if ((status = get_converter(im->ndtype, &econvert, &esize))) {
        return status;
      }
    }
  }

  parent = (int *)malloc((size_t)n * sizeof(int));
  rank = (int *)calloc((size_t)n, sizeof(int));
  group_id = (int *)malloc((size_t)n * sizeof(int));
  root_map = (int *)malloc((size_t)n * sizeof(int));
  group_counts = (int *)calloc((size_t)n, sizeof(int));
  group_offsets = (int *)malloc((size_t)(n + 1) * sizeof(int));
  group_fill = (int *)malloc((size_t)n * sizeof(int));
  members = (int *)malloc((size_t)n * sizeof(int));
  r2 = (double *)malloc((size_t)n * sizeof(double));
  r_in2 = (double *)malloc((size_t)n * sizeof(double));
  r_out2 = (double *)malloc((size_t)n * sizeof(double));
  sigma_arr = (double *)malloc((size_t)n * sizeof(double));

  if (!parent || !rank || !group_id || !root_map || !group_counts || !group_offsets
      || !group_fill || !members || !r2 || !r_in2 || !r_out2 || !sigma_arr)
  {
    status = MEMORY_ALLOC_ERROR;
    goto cleanup;
  }

  for (i = 0; i < n; i++) {
    parent[i] = i;
    if (r[i] < 0.0 || !(fwhm[i] > 0.0)) {
      status = ILLEGAL_APER_PARAMS;
      goto cleanup;
    }
    r2[i] = r[i] * r[i];
    oversamp_ann_circle(r[i], &r_in2[i], &r_out2[i]);
    sigma = fwhm[i] / 2.354820045;
    sigma_arr[i] = sigma;
    if (!(sigma_arr[i] > 0.0)) {
      status = ILLEGAL_APER_PARAMS;
      goto cleanup;
    }
    flag[i] = 0;
    sum[i] = 0.0;
    sumerr[i] = 0.0;
    area[i] = 0.0;
  }

  for (i = 0; i < n; i++) {
    for (j = i + 1; j < n; j++) {
      dx = x[i] - x[j];
      dy = y[i] - y[j];
      rsum = r[i] + r[j];
      dist2 = dx * dx + dy * dy;
      if (dist2 <= rsum * rsum) {
        uf_union(parent, rank, i, j);
      }
    }
  }

  for (i = 0; i < n; i++) {
    root_map[i] = -1;
  }
  g = 0;
  for (i = 0; i < n; i++) {
    int root = uf_find(parent, i);
    if (root_map[root] < 0) {
      root_map[root] = g++;
    }
    group_id[i] = root_map[root];
    group_counts[group_id[i]] += 1;
  }

  group_offsets[0] = 0;
  for (i = 0; i < g; i++) {
    group_offsets[i + 1] = group_offsets[i] + group_counts[i];
    group_fill[i] = group_offsets[i];
  }
  for (i = 0; i < n; i++) {
    int gid = group_id[i];
    members[group_fill[gid]++] = i;
  }

  if (subpix > 0) {
    scale = 1.0 / subpix;
    scale2 = scale * scale;
    offset = 0.5 * (scale - 1.0);
  } else {
    scale = 0.0;
    scale2 = 0.0;
    offset = 0.0;
  }

  for (gi = 0; gi < g; gi++) {
    int gcount = group_counts[gi];
    int *gidx = members + group_offsets[gi];
    double group_mean = 0.0;
    double group_err = 0.0;
    int has_pos = 0;
    int has_neg = 0;
    int n_pos = 0;
    int n_neg = 0;
    int *pos_ids = NULL;
    int *neg_ids = NULL;

    if (use_bkg) {
      double wsum = 0.0;
      double werr2 = 0.0;
      for (i = 0; i < gcount; i++) {
        int idx = gidx[i];
        double w = bkg_weight ? bkg_weight[idx] : 1.0;
        if (!(w > 0.0)) {
          continue;
        }
        if (!(bkg_mean[idx] == bkg_mean[idx])) {
          continue;
        }
        wsum += w;
        group_mean += w * bkg_mean[idx];
        if (bkg_mean_err) {
          double err = bkg_mean_err[idx];
          if (err > 0.0) {
            double we = w * err;
            werr2 += we * we;
          }
        }
      }

      if (!(wsum > 0.0)) {
        put_errdetail("group background annulus has no valid pixels");
        status = ILLEGAL_APER_PARAMS;
        goto cleanup;
      }

      group_mean /= wsum;
      if (bkg_mean_err) {
        group_err = sqrt(werr2) / wsum;
      }
    }

    if (gcount == 1) {
      int idx = gidx[0];
      status = sep_sum_circle_optimal(
          im, x[idx], y[idx], r[idx], fwhm[idx], id ? id[idx] : 0, subpix, inflag,
          &sum[idx], &sumerr[idx], &area[idx], &flag[idx]
      );
      if (status != RETURN_OK) {
        goto cleanup;
      }
      if (use_bkg && area[idx] > 0.0) {
        double berr;
        sum[idx] -= group_mean * area[idx];
        if (group_err > 0.0) {
          berr = group_err * area[idx];
          sumerr[idx] = sqrt(sumerr[idx] * sumerr[idx] + berr * berr);
        }
      }
      continue;
    }

    xmin = im->w;
    xmax = 0;
    ymin = im->h;
    ymax = 0;

    for (i = 0; i < gcount; i++) {
      int idx = gidx[i];
      int64_t lxmin, lxmax, lymin, lymax;
      short lflag = 0;
      boxextent(x[idx], y[idx], r[idx], r[idx], im->w, im->h, &lxmin, &lxmax, &lymin,
                &lymax, &lflag);
      flag[idx] |= lflag;
      if (lxmin < xmin) {
        xmin = lxmin;
      }
      if (lxmax > xmax) {
        xmax = lxmax;
      }
      if (lymin < ymin) {
        ymin = lymin;
      }
      if (lymax > ymax) {
        ymax = lymax;
      }
    }

    M = (double *)calloc((size_t)(gcount * gcount), sizeof(double));
    b = (double *)calloc((size_t)gcount, sizeof(double));
    work = (double *)malloc((size_t)gcount * sizeof(double));
    sol = (double *)malloc((size_t)gcount * sizeof(double));
    totarea = (double *)calloc((size_t)gcount, sizeof(double));
    maskarea = (double *)calloc((size_t)gcount, sizeof(double));
    ai = (double *)malloc((size_t)gcount * sizeof(double));

    if (!M || !b || !work || !sol || !totarea || !maskarea || !ai) {
      status = MEMORY_ALLOC_ERROR;
      free(M);
      free(b);
      free(work);
      free(sol);
      free(totarea);
      free(maskarea);
      free(ai);
      goto cleanup;
    }

    if (im->segmap && id) {
      pos_ids = (int *)malloc((size_t)gcount * sizeof(int));
      neg_ids = (int *)malloc((size_t)gcount * sizeof(int));
      if (!pos_ids || !neg_ids) {
        status = MEMORY_ALLOC_ERROR;
        free(M);
        free(b);
        free(work);
        free(sol);
        free(totarea);
        free(maskarea);
        free(ai);
        free(pos_ids);
        free(neg_ids);
        goto cleanup;
      }
      for (i = 0; i < gcount; i++) {
        int idx = gidx[i];
        if (id[idx] > 0) {
          has_pos = 1;
          pos_ids[n_pos++] = id[idx];
        } else if (id[idx] < 0) {
          has_neg = 1;
          neg_ids[n_neg++] = -id[idx];
        }
      }
    }

    for (iy = ymin; iy < ymax; iy++) {
      pos = (iy % im->h) * im->w + xmin;
      datat = MSVC_VOID_CAST im->data + pos * size;
      if (errisarray) {
        errort = MSVC_VOID_CAST im->noise + pos * esize;
      }
      if (im->mask) {
        maskt = MSVC_VOID_CAST im->mask + pos * msize;
      }
      if (im->segmap) {
        segt = MSVC_VOID_CAST im->segmap + pos * ssize;
      }

      for (ix = xmin; ix < xmax; ix++) {
        ismasked = 0;
        if (im->mask && (mconvert(maskt) > im->maskthresh)) {
          ismasked = 1;
        }

        if (im->segmap) {
          int seg_masked = 0;
          double segval = sconvert(segt);
          if (has_pos) {
            if (segval > 0.0) {
              seg_masked = 1;
              for (i = 0; i < n_pos; i++) {
                if (segval == pos_ids[i]) {
                  seg_masked = 0;
                  break;
                }
              }
            }
          } else if (has_neg) {
            seg_masked = 1;
            for (i = 0; i < n_neg; i++) {
              if (segval == neg_ids[i]) {
                seg_masked = 0;
                break;
              }
            }
          }
          if (seg_masked) {
            ismasked = 1;
          }
        }

        pix = convert(datat);
        if (errisarray) {
          varpix = econvert(errort);
          if (errisstd) {
            varpix *= varpix;
          }
        } else if (im->noise_type != SEP_NOISE_NONE) {
          varpix = (errisstd) ? im->noiseval * im->noiseval : im->noiseval;
        } else {
          varpix = 1.0;
        }

        if (varpix <= 0.0) {
          ismasked = 1;
        }

        double union_overlap = 0.0;
        for (i = 0; i < gcount; i++) {
          int idx = gidx[i];
          dx = ix - x[idx];
          dy = iy - y[idx];
          double dx0 = dx;
          double dy0 = dy;
          rpix2 = dx * dx + dy * dy;
          overlap = 0.0;
          if (rpix2 < r_out2[idx]) {
            if (rpix2 > r_in2[idx]) {
              if (subpix == 0) {
                overlap =
                    circoverlap(dx - 0.5, dy - 0.5, dx + 0.5, dy + 0.5, r[idx]);
              } else {
                dx += offset;
                dy += offset;
                overlap = 0.0;
                for (sy = subpix; sy--; dy += scale) {
                  dx1 = dx;
                  dy2 = dy * dy;
                  for (sx = subpix; sx--; dx1 += scale) {
                    if (dx1 * dx1 + dy2 < r2[idx]) {
                      overlap += scale2;
                    }
                  }
                }
              }
            } else {
              overlap = 1.0;
            }
          }

          if (overlap > 0.0) {
            totarea[i] += overlap;
            if (ismasked) {
              flag[idx] |= SEP_APER_HASMASKED;
              maskarea[i] += overlap;
            }
            if (overlap > union_overlap) {
              union_overlap = overlap;
            }
          }

          ai[i] = gaussian_pixel_integral(dx0, dy0, sigma_arr[idx]);
        }

        if (!ismasked && union_overlap > 0.0) {
          for (i = 0; i < gcount; i++) {
            ai[i] *= union_overlap;
          }
        } else {
          for (i = 0; i < gcount; i++) {
            ai[i] = 0.0;
          }
        }

        if (!ismasked) {
          double w = 1.0 / varpix;
          for (i = 0; i < gcount; i++) {
            if (ai[i] <= 0.0) {
              continue;
            }
            b[i] += w * ai[i] * pix;
            for (j = 0; j <= i; j++) {
              if (ai[j] > 0.0) {
                M[i * gcount + j] += w * ai[i] * ai[j];
              }
            }
          }
        }

        datat += size;
        if (errisarray) {
          errort += esize;
        }
        maskt += msize;
        segt += ssize;
      }
    }

    for (i = 0; i < gcount; i++) {
      for (j = 0; j < i; j++) {
        M[j * gcount + i] = M[i * gcount + j];
      }
    }

    for (i = 0; i < gcount; i++) {
      int idx = gidx[i];
      if (im->mask) {
        if (totarea[i] > 0.0 && maskarea[i] >= totarea[i]) {
          flag[idx] |= SEP_APER_ALLMASKED;
          area[idx] = 0.0;
        } else if (inflag & SEP_MASK_IGNORE) {
          area[idx] = totarea[i] - maskarea[i];
        } else {
          area[idx] = totarea[i];
        }
      } else {
        area[idx] = totarea[i];
      }
    }

    tmp = 0.0;
    for (i = 0; i < gcount; i++) {
      tmp += M[i * gcount + i];
    }
    if (!(tmp > 0.0) || cholesky_decomp(M, gcount)) {
      for (i = 0; i < gcount; i++) {
        int idx = gidx[i];
        status = sep_sum_circle_optimal(
            im,
            x[idx],
            y[idx],
            r[idx],
            fwhm[idx],
            id ? id[idx] : 0,
            subpix,
            inflag,
            &sum[idx],
            &sumerr[idx],
            &area[idx],
            &flag[idx]
        );
        if (status != RETURN_OK) {
          free(M);
          free(b);
          free(work);
          free(sol);
          free(totarea);
          free(maskarea);
          free(ai);
          goto cleanup;
        }
      }
    } else {
      cholesky_solve(M, b, sol, gcount, work);
      for (i = 0; i < gcount; i++) {
        int idx = gidx[i];
        sum[idx] = sol[i];
      }

      for (i = 0; i < gcount; i++) {
        memset(work, 0, (size_t)gcount * sizeof(double));
        work[i] = 1.0;
        cholesky_solve(M, work, sol, gcount, b);
        var = sol[i];
        if (var < 0.0) {
          var = 0.0;
        }
        if (im->gain > 0.0 && sum[gidx[i]] > 0.0) {
          var += sum[gidx[i]] / im->gain;
        }
        sumerr[gidx[i]] = sqrt(var);
      }
    }

    if (use_bkg) {
      for (i = 0; i < gcount; i++) {
        int idx = gidx[i];
        if (area[idx] > 0.0) {
          double berr;
          sum[idx] -= group_mean * area[idx];
          if (group_err > 0.0) {
            berr = group_err * area[idx];
            sumerr[idx] = sqrt(sumerr[idx] * sumerr[idx] + berr * berr);
          }
        }
      }
    }

    free(M);
    free(b);
    free(work);
    free(sol);
    free(totarea);
    free(maskarea);
    free(ai);
    free(pos_ids);
    free(neg_ids);
  }

  status = RETURN_OK;

cleanup:
  free(parent);
  free(rank);
  free(group_id);
  free(root_map);
  free(group_counts);
  free(group_offsets);
  free(group_fill);
  free(members);
  free(r2);
  free(r_in2);
  free(r_out2);
  free(sigma_arr);

  return status;
}

/*****************************************************************************/
/* circular aperture optimal extraction with auto-grouping */

int sep_sum_circle_optimal_multi(
    const sep_image * im,
    const double * x,
    const double * y,
    const double * r,
    const double * fwhm,
    int64_t n,
    const int * id,
    int subpix,
    short inflag,
    double * sum,
    double * sumerr,
    double * area,
    short * flag
) {
  return sep_sum_circle_optimal_multi_impl(
      im, x, y, r, fwhm, n, id, subpix, inflag,
      NULL, NULL, NULL, sum, sumerr, area, flag
  );
}

int sep_sum_circle_optimal_multi_bkg(
    const sep_image * im,
    const double * x,
    const double * y,
    const double * r,
    const double * fwhm,
    int64_t n,
    const int * id,
    int subpix,
    short inflag,
    const double * bkg_mean,
    const double * bkg_mean_err,
    const double * bkg_weight,
    double * sum,
    double * sumerr,
    double * area,
    short * flag
) {
  return sep_sum_circle_optimal_multi_impl(
      im, x, y, r, fwhm, n, id, subpix, inflag,
      bkg_mean, bkg_mean_err, bkg_weight, sum, sumerr, area, flag
  );
}

/*****************************************************************************/
/* elliptical aperture */

#define APER_NAME sep_sum_ellipse
#define APER_ARGS double a, double b, double theta, double r
#define APER_DECL double cxx, cyy, cxy, r2, r_in2, r_out2
#define APER_CHECKS                                                               \
  if (!(r >= 0.0 && b >= 0.0 && a >= b && theta >= -PI / 2. && theta <= PI / 2.)) \
  return ILLEGAL_APER_PARAMS
#define APER_INIT                                    \
  r2 = r * r;                                        \
  oversamp_ann_ellipse(r, b, &r_in2, &r_out2);       \
  sep_ellipse_coeffs(a, b, theta, &cxx, &cyy, &cxy); \
  a *= r;                                            \
  b *= r
#define APER_BOXEXTENT                                                       \
  boxextent_ellipse(                                                         \
      x, y, cxx, cyy, cxy, r, im->w, im->h, &xmin, &xmax, &ymin, &ymax, flag \
  )
#define APER_EXACT ellipoverlap(dx - 0.5, dy - 0.5, dx + 0.5, dy + 0.5, a, b, theta)
#define APER_RPIX2 cxx * dx * dx + cyy * dy * dy + cxy * dx * dy
#define APER_RPIX2_SUBPIX cxx * dx1 * dx1 + cyy * dy2 + cxy * dx1 * dy
#define APER_COMPARE1 rpix2 < r_out2
#define APER_COMPARE2 rpix2 > r_in2
#define APER_COMPARE3 rpix2 < r2
#include "aperture.i"
#undef APER_NAME
#undef APER_ARGS
#undef APER_DECL
#undef APER_CHECKS
#undef APER_INIT
#undef APER_BOXEXTENT
#undef APER_EXACT
#undef APER_RPIX2
#undef APER_RPIX2_SUBPIX
#undef APER_COMPARE1
#undef APER_COMPARE2
#undef APER_COMPARE3

/*****************************************************************************/
/* circular annulus aperture */

#define APER_NAME sep_sum_circann
#define APER_ARGS double rin, double rout
#define APER_DECL double rin2, rin_in2, rin_out2, rout2, rout_in2, rout_out2
#define APER_CHECKS                 \
  if (!(rin >= 0.0 && rout >= rin)) \
  return ILLEGAL_APER_PARAMS
#define APER_INIT                                \
  rin2 = rin * rin;                              \
  oversamp_ann_circle(rin, &rin_in2, &rin_out2); \
  rout2 = rout * rout;                           \
  oversamp_ann_circle(rout, &rout_in2, &rout_out2)
#define APER_BOXEXTENT \
  boxextent(x, y, rout, rout, im->w, im->h, &xmin, &xmax, &ymin, &ymax, flag)
#define APER_EXACT                                           \
  (circoverlap(dx - 0.5, dy - 0.5, dx + 0.5, dy + 0.5, rout) \
   - circoverlap(dx - 0.5, dy - 0.5, dx + 0.5, dy + 0.5, rin))
#define APER_RPIX2 dx * dx + dy * dy
#define APER_RPIX2_SUBPIX dx1 * dx1 + dy2
#define APER_COMPARE1 (rpix2 < rout_out2) && (rpix2 > rin_in2)
#define APER_COMPARE2 (rpix2 > rout_in2) || (rpix2 < rin_out2)
#define APER_COMPARE3 (rpix2 < rout2) && (rpix2 > rin2)
#include "aperture.i"
#undef APER_NAME
#undef APER_ARGS
#undef APER_DECL
#undef APER_CHECKS
#undef APER_INIT
#undef APER_BOXEXTENT
#undef APER_EXACT
#undef APER_RPIX2
#undef APER_RPIX2_SUBPIX
#undef APER_COMPARE1
#undef APER_COMPARE2
#undef APER_COMPARE3

int sep_stats_circann(
    const sep_image * im,
    double x,
    double y,
    double rin,
    double rout,
    int id,
    int subpix,
    short inflag,
    double clip_sigma,
    int clip_iters,
    double * mean,
    double * std,
    double * median,
    double * mad_std,
    double * mean_clip,
    double * area,
    double * sumerr,
    short * flag
) {
  PIXTYPE pix;
  double dx, dy, dx1, dy2, offset, scale, scale2, rpix2, overlap;
  double rin2, rin_in2, rin_out2, rout2, rout_in2, rout_out2;
  double totw, sumw, varw, diff, wsum, vsum;
  double tv, sigtv, varpix, varpix_const;
  double med, madv, sig, totw_keep;
  double lo, hi;
  int64_t ix, iy, i, xmin, xmax, ymin, ymax, sx, sy, pos, size, msize, ssize, nkeep;
  int64_t nvals, cap, esize;
  int status, ismasked, changed, iter;
  short errisarray, errisstd;
  const BYTE *datat, *errort, *maskt, *segt;
  converter convert, econvert = NULL, mconvert, sconvert;
  valweight *vw = NULL;
  valweight *tmp = NULL;
  char *keep = NULL;
  const double mad_scale = 1.4826;

  /* input checks */
  if (!(rin >= 0.0 && rout >= rin)) {
    return ILLEGAL_APER_PARAMS;
  }
  if (subpix < 0) {
    return ILLEGAL_SUBPIX;
  }
  if (clip_sigma <= 0.0 || clip_iters < 0) {
    return ILLEGAL_APER_PARAMS;
  }

  (void)inflag;

  *flag = 0;
  *mean = NAN;
  *std = NAN;
  *median = NAN;
  *mad_std = NAN;
  *mean_clip = NAN;
  *area = 0.0;
  *sumerr = 0.0;

  rin2 = rin * rin;
  rout2 = rout * rout;
  oversamp_ann_circle(rin, &rin_in2, &rin_out2);
  oversamp_ann_circle(rout, &rout_in2, &rout_out2);

  if (subpix > 0) {
    scale = 1.0 / subpix;
    scale2 = scale * scale;
    offset = 0.5 * (scale - 1.0);
  } else {
    scale = 0.0;
    scale2 = 0.0;
    offset = 0.0;
  }

  maskt = NULL;
  segt = NULL;
  msize = 0;
  ssize = 0;
  errort = im->noise;
  esize = 0;
  errisarray = 0;
  errisstd = 0;
  varpix_const = 0.0;

  /* get data converter(s) for input array(s) */
  if ((status = get_converter(im->dtype, &convert, &size))) {
    return status;
  }
  if (im->mask && (status = get_converter(im->mdtype, &mconvert, &msize))) {
    return status;
  }
  if (im->segmap && (status = get_converter(im->sdtype, &sconvert, &ssize))) {
    return status;
  }

  /* get extent of box */
  boxextent(x, y, rout, rout, im->w, im->h, &xmin, &xmax, &ymin, &ymax, flag);

  cap = (xmax - xmin) * (ymax - ymin);
  if (cap <= 0) {
    *flag |= SEP_APER_ALLMASKED;
    return RETURN_OK;
  }

  status = RETURN_OK;
  QMALLOC(vw, valweight, cap, status);

  nvals = 0;
  totw = 0.0;
  tv = 0.0;
  sigtv = 0.0;

  /* loop over rows in the box */
  for (iy = ymin; iy < ymax; iy++) {
    /* set pointers to the start of this row */
    pos = (iy % im->h) * im->w + xmin;
    datat = MSVC_VOID_CAST im->data + pos * size;
    if (errisarray) {
      errort = MSVC_VOID_CAST im->noise + pos * esize;
    }
    if (im->mask) {
      maskt = MSVC_VOID_CAST im->mask + pos * msize;
    }
    if (im->segmap) {
      segt = MSVC_VOID_CAST im->segmap + pos * ssize;
    }

    /* loop over pixels in this row */
    for (ix = xmin; ix < xmax; ix++) {
      dx = ix - x;
      dy = iy - y;
      rpix2 = dx * dx + dy * dy;
      if ((rpix2 < rout_out2) && (rpix2 > rin_in2)) {
        if ((rpix2 > rout_in2) || (rpix2 < rin_out2)) {
          if (subpix == 0) {
            overlap = circoverlap(dx - 0.5, dy - 0.5, dx + 0.5, dy + 0.5, rout)
                      - circoverlap(dx - 0.5, dy - 0.5, dx + 0.5, dy + 0.5, rin);
          } else {
            dx += offset;
            dy += offset;
            overlap = 0.0;
            for (sy = subpix; sy--; dy += scale) {
              dx1 = dx;
              dy2 = dy * dy;
              for (sx = subpix; sx--; dx1 += scale) {
                rpix2 = dx1 * dx1 + dy2;
                if ((rpix2 < rout2) && (rpix2 > rin2)) {
                  overlap += scale2;
                }
              }
            }
          }
        } else {
          overlap = 1.0;
        }

        if (overlap > 0.0) {
          pix = convert(datat);

          ismasked = 0;
          if (im->mask && (mconvert(maskt) > im->maskthresh)) {
            ismasked = 1;
          }

          /* Segmentation image:

               If `id` is negative, require segmented pixels within the
               aperture.

               If `id` is positive, mask pixels with nonzero segment ids
               not equal to `id`.

          */
          if (im->segmap) {
            if (id > 0) {
              if ((sconvert(segt) > 0.) && (sconvert(segt) != id)) {
                ismasked = 1;
              }
            } else {
              if (sconvert(segt) != -1 * id) {
                ismasked = 1;
              }
            }
          }

          if (ismasked) {
            *flag |= SEP_APER_HASMASKED;
          } else {
            vw[nvals].v = pix;
            vw[nvals].w = overlap;
            totw += overlap;
            tv += pix * overlap;
            if (errisarray) {
              varpix = econvert(errort);
              if (errisstd) {
                varpix *= varpix;
              }
            } else {
              varpix = varpix_const;
            }
            sigtv += varpix * overlap;
            nvals++;
          }
        }
      }

      /* increment pointers by one element */
      datat += size;
      if (errisarray) {
        errort += esize;
      }
      maskt += msize;
      segt += ssize;
    }
  }

  if (nvals == 0 || totw <= 0.0) {
    *flag |= SEP_APER_ALLMASKED;
    goto exit;
  }

  /* mean and standard deviation */
  sumw = 0.0;
  for (i = 0; i < nvals; i++) {
    if (vw[i].w > 0.0) {
      sumw += vw[i].w * vw[i].v;
    }
  }
  *mean = sumw / totw;

  varw = 0.0;
  for (i = 0; i < nvals; i++) {
    if (vw[i].w > 0.0) {
      diff = vw[i].v - *mean;
      varw += vw[i].w * diff * diff;
    }
  }
  *std = sqrt(varw / totw);

  /* sigma-clipped mean using median/MAD (3-sigma, max 5 iterations) */
  status = RETURN_OK;
  QMALLOC(keep, char, nvals, status);
  QMALLOC(tmp, valweight, nvals, status);
  for (i = 0; i < nvals; i++) {
    keep[i] = 1;
  }

  changed = 1;
  for (iter = 0; iter < clip_iters; iter++) {
    nkeep = 0;
    totw_keep = 0.0;
    for (i = 0; i < nvals; i++) {
      if (keep[i] && vw[i].w > 0.0) {
        tmp[nkeep] = vw[i];
        totw_keep += vw[i].w;
        nkeep++;
      }
    }
    if (nkeep == 0 || totw_keep <= 0.0) {
      break;
    }

    qsort(tmp, (size_t)nkeep, sizeof(valweight), cmp_valweight);
    med = weighted_median_sorted(tmp, nkeep, totw_keep);
    for (i = 0; i < nkeep; i++) {
      tmp[i].v = fabs(tmp[i].v - med);
    }
    qsort(tmp, (size_t)nkeep, sizeof(valweight), cmp_valweight);
    madv = weighted_median_sorted(tmp, nkeep, totw_keep);
    sig = madv * mad_scale;
    if (sig <= 0.0) {
      break;
    }
    lo = med - clip_sigma * sig;
    hi = med + clip_sigma * sig;
    changed = 0;
    for (i = 0; i < nvals; i++) {
      if (keep[i] && (vw[i].v < lo || vw[i].v > hi)) {
        keep[i] = 0;
        changed = 1;
      }
    }
    if (!changed) {
      break;
    }
  }

  wsum = 0.0;
  vsum = 0.0;
  for (i = 0; i < nvals; i++) {
    if (keep[i] && vw[i].w > 0.0) {
      wsum += vw[i].w;
      vsum += vw[i].w * vw[i].v;
    }
  }
  *mean_clip = (wsum > 0.0) ? (vsum / wsum) : NAN;

  /* weighted median */
  qsort(vw, (size_t)nvals, sizeof(valweight), cmp_valweight);
  *median = weighted_median_sorted(vw, nvals, totw);

  /* weighted MAD (unscaled) */
  for (i = 0; i < nvals; i++) {
    vw[i].v = fabs(vw[i].v - *median);
  }
  qsort(vw, (size_t)nvals, sizeof(valweight), cmp_valweight);
  madv = weighted_median_sorted(vw, nvals, totw);
  *mad_std = madv * mad_scale;

  if (im->gain > 0.0 && tv > 0.0) {
    sigtv += tv / im->gain;
  }
  if (sigtv < 0.0) {
    sigtv = 0.0;
  }
  *sumerr = sqrt(sigtv);
  *area = totw;

exit:
  if (keep) {
    free(keep);
  }
  if (tmp) {
    free(tmp);
  }
  if (vw) {
    free(vw);
  }

  return status;
}

/*****************************************************************************/
/* elliptical annulus aperture */

#define APER_NAME sep_sum_ellipann
#define APER_ARGS double a, double b, double theta, double rin, double rout
#define APER_DECL       \
  double cxx, cyy, cxy; \
  double rin2, rin_in2, rin_out2, rout2, rout_in2, rout_out2
#define APER_CHECKS                                                          \
  if (!(rin >= 0.0 && rout >= rin && b >= 0.0 && a >= b && theta >= -PI / 2. \
        && theta <= PI / 2.))                                                \
  return ILLEGAL_APER_PARAMS
#define APER_INIT                                       \
  rin2 = rin * rin;                                     \
  oversamp_ann_ellipse(rin, b, &rin_in2, &rin_out2);    \
  rout2 = rout * rout;                                  \
  oversamp_ann_ellipse(rout, b, &rout_in2, &rout_out2); \
  sep_ellipse_coeffs(a, b, theta, &cxx, &cyy, &cxy)
#define APER_BOXEXTENT                                                          \
  boxextent_ellipse(                                                            \
      x, y, cxx, cyy, cxy, rout, im->w, im->h, &xmin, &xmax, &ymin, &ymax, flag \
  )
#define APER_EXACT                                                                 \
  (ellipoverlap(dx - 0.5, dy - 0.5, dx + 0.5, dy + 0.5, a * rout, b * rout, theta) \
   - ellipoverlap(dx - 0.5, dy - 0.5, dx + 0.5, dy + 0.5, a * rin, b * rin, theta))
#define APER_RPIX2 cxx * dx * dx + cyy * dy * dy + cxy * dx * dy
#define APER_RPIX2_SUBPIX cxx * dx1 * dx1 + cyy * dy2 + cxy * dx1 * dy
#define APER_COMPARE1 (rpix2 < rout_out2) && (rpix2 > rin_in2)
#define APER_COMPARE2 (rpix2 > rout_in2) || (rpix2 < rin_out2)
#define APER_COMPARE3 (rpix2 < rout2) && (rpix2 > rin2)
#include "aperture.i"
#undef APER_NAME
#undef APER_ARGS
#undef APER_DECL
#undef APER_CHECKS
#undef APER_INIT
#undef APER_BOXEXTENT
#undef APER_EXACT
#undef APER_RPIX2
#undef APER_RPIX2_SUBPIX
#undef APER_COMPARE1
#undef APER_COMPARE2
#undef APER_COMPARE3


/*****************************************************************************/
/*
 * This is just different enough from the other aperture functions
 * that it doesn't quite make sense to use aperture.i.
 */
int sep_sum_circann_multi(
    const sep_image * im,
    double x,
    double y,
    double rmax,
    int64_t n,
    int id,
    int subpix,
    short inflag,
    double * sum,
    double * sumvar,
    double * area,
    double * maskarea,
    short * flag
) {
  PIXTYPE pix, varpix;
  double dx, dy, dx1, dy2, offset, scale, scale2, tmp, rpix2;
  int64_t ix, iy, xmin, xmax, ymin, ymax, sx, sy, size, esize, msize, ssize, pos;
  int status;
  short errisarray, errisstd;
  const BYTE *datat, *errort, *maskt, *segt;
  converter convert, econvert, mconvert, sconvert;
  double rpix, r_out, r_out2, d, prevbinmargin, nextbinmargin, step, stepdens;
  int64_t j, ismasked;

  /* input checks */
  if (rmax < 0.0 || n < 1) {
    return ILLEGAL_APER_PARAMS;
  }
  if (subpix < 1) {
    return ILLEGAL_SUBPIX;
  }

  /* clear results arrays */
  memset(sum, 0, (size_t)(n * sizeof(double)));
  memset(sumvar, 0, (size_t)(n * sizeof(double)));
  memset(area, 0, (size_t)(n * sizeof(double)));
  if (im->mask) {
    memset(maskarea, 0, (size_t)(n * sizeof(double)));
  }

  /* initializations */
  size = esize = msize = ssize = 0;
  datat = maskt = segt = NULL;
  errort = im->noise;
  *flag = 0;
  varpix = 0.0;
  scale = 1.0 / subpix;
  scale2 = scale * scale;
  offset = 0.5 * (scale - 1.0);

  r_out = rmax + 1.5; /* margin for interpolation */
  r_out2 = r_out * r_out;
  step = rmax / n;
  stepdens = 1.0 / step;
  prevbinmargin = 0.7072;
  nextbinmargin = step - 0.7072;
  j = 0;
  d = 0.;
  ismasked = 0;
  errisarray = 0;
  errisstd = 0;

  /* get data converter(s) for input array(s) */
  if ((status = get_converter(im->dtype, &convert, &size))) {
    return status;
  }
  if (im->mask && (status = get_converter(im->mdtype, &mconvert, &msize))) {
    return status;
  }
  if (im->segmap && (status = get_converter(im->sdtype, &sconvert, &ssize))) {
    return status;
  }

  /* get image noise */
  if (im->noise_type != SEP_NOISE_NONE) {
    errisstd = (im->noise_type == SEP_NOISE_STDDEV);
    if (im->noise) {
      errisarray = 1;
      if ((status = get_converter(im->ndtype, &econvert, &esize))) {
        return status;
      }
    } else {
      varpix = (errisstd) ? im->noiseval * im->noiseval : im->noiseval;
    }
  }


  /* get extent of box */
  boxextent(x, y, r_out, r_out, im->w, im->h, &xmin, &xmax, &ymin, &ymax, flag);

  /* loop over rows in the box */
  for (iy = ymin; iy < ymax; iy++) {
    /* set pointers to the start of this row */
    pos = (iy % im->h) * im->w + xmin;
    datat = MSVC_VOID_CAST im->data + pos * size;
    if (errisarray) {
      errort = MSVC_VOID_CAST im->noise + pos * esize;
    }
    if (im->mask) {
      maskt = MSVC_VOID_CAST im->mask + pos * msize;
    }
    if (im->segmap) {
      segt = MSVC_VOID_CAST im->segmap + pos * ssize;
    }

    /* loop over pixels in this row */
    for (ix = xmin; ix < xmax; ix++) {
      dx = ix - x;
      dy = iy - y;
      rpix2 = dx * dx + dy * dy;
      if (rpix2 < r_out2) {
        /* get pixel values */
        pix = convert(datat);
        if (errisarray) {
          varpix = econvert(errort);
          if (errisstd) {
            varpix *= varpix;
          }
        }

        ismasked = 0;
        if (im->mask) {
          if (mconvert(maskt) > im->maskthresh) {
            *flag |= SEP_APER_HASMASKED;
            ismasked = 1;
          }
        }

        /* Segmentation image:

             If `id` is negative, require segmented pixels within the
             aperture.

             If `id` is positive, mask pixels with nonzero segment ids
             not equal to `id`.

        */
        if (im->segmap) {
          if (id > 0) {
            if ((sconvert(segt) > 0.) && (sconvert(segt) != id)) {
              *flag |= SEP_APER_HASMASKED;
              ismasked = 1;
            }
          } else {
            if (sconvert(segt) != -1 * id) {
              *flag |= SEP_APER_HASMASKED;
              ismasked = 1;
            }
          }
        }

        /* check if oversampling is needed (close to bin boundary?) */
        rpix = sqrt(rpix2);
        d = fmod(rpix, step);
        if (d < prevbinmargin || d > nextbinmargin) {
          dx += offset;
          dy += offset;
          for (sy = subpix; sy--; dy += scale) {
            dx1 = dx;
            dy2 = dy * dy;
            for (sx = subpix; sx--; dx1 += scale) {
              j = (int64_t)(sqrt(dx1 * dx1 + dy2) * stepdens);
              if (j < n) {
                if (ismasked) {
                  maskarea[j] += scale2;
                } else {
                  sum[j] += scale2 * pix;
                  sumvar[j] += scale2 * varpix;
                }
                area[j] += scale2;
              }
            }
          }
        } else
        /* pixel not close to bin boundary */
        {
          j = (int64_t)(rpix * stepdens);
          if (j < n) {
            if (ismasked) {
              maskarea[j] += 1.0;
            } else {
              sum[j] += pix;
              sumvar[j] += varpix;
            }
            area[j] += 1.0;
          }
        }
      } /* closes "if pixel might be within aperture" */

      /* increment pointers by one element */
      datat += size;
      if (errisarray) {
        errort += esize;
      }
      maskt += msize;
      segt += ssize;
    }
  }


  /* correct for masked values */
  if (im->mask) {
    if (inflag & SEP_MASK_IGNORE) {
      for (j = n; j--;) {
        area[j] -= maskarea[j];
      }
    } else {
      for (j = n; j--;) {
        tmp = area[j] == maskarea[j] ? 0.0 : area[j] / (area[j] - maskarea[j]);
        sum[j] *= tmp;
        sumvar[j] *= tmp;
      }
    }
  }

  /* add poisson noise, only if gain > 0 */
  if (im->gain > 0.0) {
    for (j = n; j--;) {
      if (sum[j] > 0.0) {
        sumvar[j] += sum[j] / im->gain;
      }
    }
  }

  return status;
}


/* for use in flux_radius */
static double inverse(double xmax, const double * y, int64_t n, double ytarg) {
  double step;
  int64_t i;

  step = xmax / n;
  i = 0;

  /* increment i until y[i] is >= to ytarg */
  while (i < n && y[i] < ytarg) {
    i++;
  }

  if (i == 0) {
    if (ytarg <= 0. || y[0] == 0.) {
      return 0.;
    }
    return step * ytarg / y[0];
  }
  if (i == n) {
    return xmax;
  }

  /* note that y[i-1] corresponds to x=step*i. */
  return step * (i + (ytarg - y[i - 1]) / (y[i] - y[i - 1]));
}

int sep_flux_radius(
    const sep_image * im,
    double x,
    double y,
    double rmax,
    int id,
    int subpix,
    short inflag,
    const double * fluxtot,
    const double * fluxfrac,
    int64_t n,
    double * r,
    short * flag
) {
  int status;
  int64_t i;
  double f;
  double sumbuf[FLUX_RADIUS_BUFSIZE] = {0.};
  double sumvarbuf[FLUX_RADIUS_BUFSIZE];
  double areabuf[FLUX_RADIUS_BUFSIZE];
  double maskareabuf[FLUX_RADIUS_BUFSIZE];

  /* measure FLUX_RADIUS_BUFSIZE annuli out to rmax. */
  status = sep_sum_circann_multi(
      im,
      x,
      y,
      rmax,
      FLUX_RADIUS_BUFSIZE,
      id,
      subpix,
      inflag,
      sumbuf,
      sumvarbuf,
      areabuf,
      maskareabuf,
      flag
  );

  /* sum up sumbuf array */
  for (i = 1; i < FLUX_RADIUS_BUFSIZE; i++) {
    sumbuf[i] += sumbuf[i - 1];
  }

  /* if given, use "total flux", else, use sum within rmax. */
  f = fluxtot ? *fluxtot : sumbuf[FLUX_RADIUS_BUFSIZE - 1];

  /* Use inverse to get the radii corresponding to the requested flux fracs */
  for (i = 0; i < n; i++) {
    r[i] = inverse(rmax, sumbuf, FLUX_RADIUS_BUFSIZE, fluxfrac[i] * f);
  }

  return status;
}

/*****************************************************************************/
/* calculate Kron radius from pixels within an ellipse. */
int sep_kron_radius(
    const sep_image * im,
    double x,
    double y,
    double cxx,
    double cyy,
    double cxy,
    double r,
    int id,
    double * kronrad,
    short * flag
) {
  float pix;
  double r1, v1, r2, area, rpix2, dx, dy;
  int64_t ix, iy, xmin, xmax, ymin, ymax, pos, size, msize, ssize;
  int status;
  int ismasked;

  const BYTE *datat, *maskt, *segt;
  converter convert, mconvert, sconvert;

  r2 = r * r;
  r1 = v1 = 0.0;
  area = 0.0;
  *flag = 0;
  datat = maskt = segt = NULL;
  size = msize = ssize = 0;

  /* get data converter(s) for input array(s) */
  if ((status = get_converter(im->dtype, &convert, &size))) {
    return status;
  }
  if (im->mask && (status = get_converter(im->mdtype, &mconvert, &msize))) {
    return status;
  }
  if (im->segmap && (status = get_converter(im->sdtype, &sconvert, &ssize))) {
    return status;
  }

  /* get extent of ellipse in x and y */
  boxextent_ellipse(
      x, y, cxx, cyy, cxy, r, im->w, im->h, &xmin, &xmax, &ymin, &ymax, flag
  );

  /* loop over rows in the box */
  for (iy = ymin; iy < ymax; iy++) {
    /* set pointers to the start of this row */
    pos = (iy % im->h) * im->w + xmin;
    datat = MSVC_VOID_CAST im->data + pos * size;
    if (im->mask) {
      maskt = MSVC_VOID_CAST im->mask + pos * msize;
    }
    if (im->segmap) {
      segt = MSVC_VOID_CAST im->segmap + pos * ssize;
    }

    /* loop over pixels in this row */
    for (ix = xmin; ix < xmax; ix++) {
      dx = ix - x;
      dy = iy - y;
      rpix2 = cxx * dx * dx + cyy * dy * dy + cxy * dx * dy;
      if (rpix2 <= r2) {
        pix = convert(datat);
        ismasked = 0;
        if ((pix < -BIG) || (im->mask && mconvert(maskt) > im->maskthresh)) {
          ismasked = 1;
        }

        /* Segmentation image:

             If `id` is negative, require segmented pixels within the
             aperture.

             If `id` is positive, mask pixels with nonzero segment ids
             not equal to `id`.

        */
        if (im->segmap) {
          if (id > 0) {
            if ((sconvert(segt) > 0.) && (sconvert(segt) != id)) {
              ismasked = 1;
            }
          } else {
            if (sconvert(segt) != -1 * id) {
              ismasked = 1;
            }
          }
        }

        if (ismasked > 0) {
          *flag |= SEP_APER_HASMASKED;
        } else {
          r1 += sqrt(rpix2) * pix;
          v1 += pix;
          area++;
        }
      }

      /* increment pointers by one element */
      datat += size;
      maskt += msize;
      segt += ssize;
    }
  }

  if (area == 0) {
    *flag |= SEP_APER_ALLMASKED;
    *kronrad = 0.0;
  } else if (r1 <= 0.0 || v1 <= 0.0) {
    *flag |= SEP_APER_NONPOSITIVE;
    *kronrad = 0.0;
  } else {
    *kronrad = r1 / v1;
  }

  return RETURN_OK;
}


/* set array values within an ellipse (uc = unsigned char array) */
void sep_set_ellipse(
    unsigned char * arr,
    int64_t w,
    int64_t h,
    double x,
    double y,
    double cxx,
    double cyy,
    double cxy,
    double r,
    unsigned char val
) {
  unsigned char * arrt;
  int64_t xmin, xmax, ymin, ymax, xi, yi;
  double r2, dx, dy, dy2;
  short flag; /* not actually used, but input to boxextent */

  flag = 0;
  r2 = r * r;

  boxextent_ellipse(x, y, cxx, cyy, cxy, r, w, h, &xmin, &xmax, &ymin, &ymax, &flag);

  for (yi = ymin; yi < ymax; yi++) {
    arrt = arr + (yi * w + xmin);
    dy = yi - y;
    dy2 = dy * dy;
    for (xi = xmin; xi < xmax; xi++, arrt++) {
      dx = xi - x;
      if ((cxx * dx * dx + cyy * dy2 + cxy * dx * dy) <= r2) {
        *arrt = val;
      }
    }
  }
}

/*****************************************************************************/
/*
 * As with `sep_sum_circann_multi`, this is different enough from the other
 * aperture functions that it doesn't quite make sense to use aperture.i.
 *
 * To reproduce SExtractor:
 * - use sig = obj.hl_radius * 2/2.35.
 * - use obj.posx/posy for initial position
 *
 */

int sep_windowed(
    const sep_image * im,
    double x,
    double y,
    double sig,
    int subpix,
    short inflag,
    int id,
    double maxstep,
    double * xout,
    double * yout,
    int * niter,
    short * flag
) {
  PIXTYPE pix, varpix;
  double dx, dy, dx1, dy2, offset, scale, scale2, tmp, dxpos, dypos, weight;
  double maskarea, maskweight, maskdxpos, maskdypos;
  double r, tv, twv, sigtv, totarea, overlap, rpix2, invtwosig2;
  double wpix, step, step_scale;
  int64_t ix, iy, xmin, xmax, ymin, ymax, sx, sy, pos, size, esize, msize, ssize;
  int i, status, ismasked;
  short errisarray, errisstd;
  const BYTE *datat, *errort, *maskt, *segt;
  converter convert, econvert, mconvert, sconvert;
  double r2, r_in2, r_out2;

  /* input checks */
  if (sig < 0.0) {
    return ILLEGAL_APER_PARAMS;
  }
  if (subpix < 0) {
    return ILLEGAL_SUBPIX;
  }

  /* initializations */
  size = esize = msize = ssize = 0;
  tv = sigtv = 0.0;
  overlap = totarea = maskweight = 0.0;
  datat = maskt = segt = NULL;
  errort = im->noise;
  *flag = 0;
  varpix = 0.0;
  if (subpix > 0) {
    scale = 1.0 / subpix;
    scale2 = scale * scale;
    offset = 0.5 * (scale - 1.0);
  } else {
    scale = 0.0;
    scale2 = 0.0;
    offset = 0.0;
  }
  invtwosig2 = 1.0 / (2.0 * sig * sig);
  errisarray = 0;
  errisstd = 0;

  /* Integration radius */
  r = WINPOS_NSIG * sig;

  /* calculate oversampled annulus */
  r2 = r * r;
  oversamp_ann_circle(r, &r_in2, &r_out2);

  /* get data converter(s) for input array(s) */
  if ((status = get_converter(im->dtype, &convert, &size))) {
    return status;
  }
  if (im->mask && (status = get_converter(im->mdtype, &mconvert, &msize))) {
    return status;
  }
  if (im->segmap && (status = get_converter(im->sdtype, &sconvert, &ssize))) {
    return status;
  }

  /* get image noise */
  if (im->noise_type != SEP_NOISE_NONE) {
    errisstd = (im->noise_type == SEP_NOISE_STDDEV);
    if (im->noise) {
      errisarray = 1;
      if ((status = get_converter(im->ndtype, &econvert, &esize))) {
        return status;
      }
    } else {
      varpix = (errisstd) ? im->noiseval * im->noiseval : im->noiseval;
    }
  }

  /* iteration loop */
  for (i = 0; i < WINPOS_NITERMAX; i++) {
    /* get extent of box */
    boxextent(x, y, r, r, im->w, im->h, &xmin, &xmax, &ymin, &ymax, flag);

    /* TODO: initialize values */
    // mx2ph
    // my2ph
    //  esum, emxy, emx2, emy2, mx2, my2, mxy
    tv = twv = sigtv = 0.0;
    overlap = totarea = maskarea = maskweight = 0.0;
    dxpos = dypos = 0.0;
    maskdxpos = maskdypos = 0.0;

    /* loop over rows in the box */
    for (iy = ymin; iy < ymax; iy++) {
      /* set pointers to the start of this row */
      pos = (iy % im->h) * im->w + xmin;
      datat = MSVC_VOID_CAST im->data + pos * size;
      if (errisarray) {
        errort = MSVC_VOID_CAST im->noise + pos * esize;
      }
      if (im->mask) {
        maskt = MSVC_VOID_CAST im->mask + pos * msize;
      }
      if (im->segmap) {
        segt = MSVC_VOID_CAST im->segmap + pos * ssize;
      }

      /* loop over pixels in this row */
      for (ix = xmin; ix < xmax; ix++) {
        dx = ix - x;
        dy = iy - y;
        rpix2 = dx * dx + dy * dy;
        if (rpix2 < r_out2) {
          if (rpix2 > r_in2) /* might be partially in aperture */ {
            if (subpix == 0) {
              overlap = circoverlap(dx - 0.5, dy - 0.5, dx + 0.5, dy + 0.5, r);
            } else {
              dx += offset;
              dy += offset;
              overlap = 0.0;
              for (sy = subpix; sy--; dy += scale) {
                dx1 = dx;
                dy2 = dy * dy;
                for (sx = subpix; sx--; dx1 += scale) {
                  if (dx1 * dx1 + dy2 < r2) {
                    overlap += scale2;
                  }
                }
              }
            }
          } else {
            /* definitely fully in aperture */
            overlap = 1.0;
          }

          /* get pixel value and variance value */
          pix = convert(datat);
          if (errisarray) {
            varpix = econvert(errort);
            if (errisstd) {
              varpix *= varpix;
            }
          }

          /* offset of this pixel from center */
          dx = ix - x;
          dy = iy - y;

          /* weight by gaussian */
          weight = exp(-rpix2 * invtwosig2);

          ismasked = 0;
          if (im->mask && (mconvert(maskt) > im->maskthresh)) {
            ismasked = 1;
          }

          /* Segmentation image:

               If `id` is negative, require segmented pixels within the
               aperture.

               If `id` is positive, mask pixels with nonzero segment ids
               not equal to `id`.

          */
          if (im->segmap) {
            if (id > 0) {
              if ((sconvert(segt) > 0.) && (sconvert(segt) != id)) {
                ismasked = 1;
              }
            } else {
              if (sconvert(segt) != -1 * id) {
                ismasked = 1;
              }
            }
          }

          if (ismasked) {
            *flag |= SEP_APER_HASMASKED;
            maskarea += overlap;
            maskweight += overlap * weight;
            maskdxpos += overlap * weight * dx;
            maskdypos += overlap * weight * dy;
          } else {
            tv += pix * overlap;
            wpix = pix * overlap * weight;
            twv += wpix;
            dxpos += wpix * dx;
            dypos += wpix * dy;
          }

          totarea += overlap;

        } /* closes "if pixel might be within aperture" */

        /* increment pointers by one element */
        datat += size;
        if (errisarray) {
          errort += esize;
        }
        maskt += msize;
        segt += ssize;
      } /* closes loop over x */
    } /* closes loop over y */

    /* we're done looping over pixels for this iteration.
     * Our summary statistics are:
     *
     * tv : total value
     * twv : total weighted value
     * dxpos : weighted dx
     * dypos : weighted dy
     */

    /* Correct for masked values: This effectively makes it as if
     * the masked pixels had the value of the average unmasked value
     * in the aperture.
     */
    if (im->mask || im->segmap) {
      if (totarea > 0.0 && maskarea >= totarea) {
        *flag |= SEP_APER_ALLMASKED;
        break;
      }
      /* this option will probably not yield accurate values */
      if (inflag & SEP_MASK_IGNORE) {
        totarea -= maskarea;
      } else {
        tmp = tv / (totarea - maskarea); /* avg unmasked pixel value */
        twv += tmp * maskweight;
        dxpos += tmp * maskdxpos;
        dypos += tmp * maskdypos;
      }
    }

    /* update center */
    if (twv > 0.0) {
      dxpos = (dxpos / twv) * WINPOS_FAC;
      dypos = (dypos / twv) * WINPOS_FAC;
      if (maxstep > 0.0) {
        step = sqrt(dxpos * dxpos + dypos * dypos);
        if (step > maxstep) {
          step_scale = maxstep / step;
          dxpos *= step_scale;
          dypos *= step_scale;
        }
      }
      x += dxpos;
      y += dypos;
    } else {
      break;
    }

    /* Stop here if position does not change */
    if (dxpos * dxpos + dypos * dypos < WINPOS_STEPMIN * WINPOS_STEPMIN) {
      break;
    }

  } /* closes loop over interations */

  /* assign output results */
  *xout = x;
  *yout = y;
  *niter = i + 1;

  return status;
}
