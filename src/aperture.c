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

static void cholesky_inverse_diagonal_inplace(double *L, int n, double *work,
                                              double *diag) {
  int i, j, k;

  for (i = 0; i < n; i++) {
    for (k = 0; k <= i; k++) work[k] = L[i * n + k];

    for (j = 0; j < i; j++) {
      double sum = 0.0;
      for (k = j; k < i; k++) sum += work[k] * L[k * n + j];
      L[i * n + j] = -sum / work[i];
    }
    L[i * n + i] = 1.0 / work[i];
  }

  for (i = 0; i < n; i++) {
    double sum = 0.0;
    for (k = i; k < n; k++) {
      double v = L[k * n + i];
      sum += v * v;
    }
    diag[i] = sum;
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

#define OPT_GROUP_EXACT_MAX 32
#define OPT_GROUP_CORE 4
#define OPT_GROUP_FIXED_MAX 96
#define OPT_GROUP_EDGE_REFINE_MAX 1024
#define OPT_GROUP_MAX_SWEEPS 8
#define OPT_GROUP_CONV_RTOL 1.0e-4
#define OPT_GROUP_CONV_ATOL 1.0e-6

typedef struct {
  double x;
  int idx;
} opt_xorder_entry;

typedef struct {
  int gid;
  int count;
} opt_group_order_entry;

typedef struct {
  size_t pix_cap;
  size_t src_cap;
  size_t nnz_cap;
  size_t id_cap;
  double *gdata;
  double *gweight;
  double *col_vals;
  double *M;
  double *b;
  double *sol;
  double *work;
  int *pix_idx;
  int *col_starts;
  int *group_ids;
  int *pos_ids;
  int *neg_ids;
} opt_group_workspace;

static int opt_xorder_cmp(const void *a, const void *b) {
  const opt_xorder_entry *pa = (const opt_xorder_entry *)a;
  const opt_xorder_entry *pb = (const opt_xorder_entry *)b;
  if (pa->x < pb->x) return -1;
  if (pa->x > pb->x) return 1;
  return (pa->idx > pb->idx) - (pa->idx < pb->idx);
}

static int opt_group_order_cmp(const void *a, const void *b) {
  const opt_group_order_entry *ga = (const opt_group_order_entry *)a;
  const opt_group_order_entry *gb = (const opt_group_order_entry *)b;
  if (ga->count > gb->count) return -1;
  if (ga->count < gb->count) return 1;
  return (ga->gid > gb->gid) - (ga->gid < gb->gid);
}

static int opt_int_cmp(const void *a, const void *b) {
  int ia = *(const int *)a;
  int ib = *(const int *)b;
  return (ia > ib) - (ia < ib);
}

static int opt_int_contains(const int *arr, int n, int value) {
  int lo = 0, hi = n - 1;
  while (lo <= hi) {
    int mid = lo + (hi - lo) / 2;
    int v = arr[mid];
    if (v < value) {
      lo = mid + 1;
    } else if (v > value) {
      hi = mid - 1;
    } else {
      return 1;
    }
  }
  return 0;
}

static int opt_realloc_bytes(void **ptr, size_t nbytes) {
  void *tmp;

  if (nbytes == 0) return RETURN_OK;
  tmp = realloc(*ptr, nbytes);
  if (!tmp) return MEMORY_ALLOC_ERROR;
  *ptr = tmp;
  return RETURN_OK;
}

static int optimal_group_workspace_ensure(opt_group_workspace *ws, size_t gnpix,
                                          int gcount, size_t nnz_cap,
                                          int id_cap) {
  int status;

  if (gnpix > ws->pix_cap) {
    if ((status = opt_realloc_bytes((void **)&ws->gdata,
                                    gnpix * sizeof(double))))
      return status;
    if ((status = opt_realloc_bytes((void **)&ws->gweight,
                                    gnpix * sizeof(double))))
      return status;
    ws->pix_cap = gnpix;
  }
  if ((size_t)gcount > ws->src_cap) {
    if ((status = opt_realloc_bytes((void **)&ws->M,
                                    (size_t)gcount * (size_t)gcount *
                                        sizeof(double))))
      return status;
    if ((status = opt_realloc_bytes((void **)&ws->b,
                                    (size_t)gcount * sizeof(double))))
      return status;
    if ((status = opt_realloc_bytes((void **)&ws->sol,
                                    (size_t)gcount * sizeof(double))))
      return status;
    if ((status = opt_realloc_bytes((void **)&ws->work,
                                    (size_t)gcount * sizeof(double))))
      return status;
    if ((status = opt_realloc_bytes((void **)&ws->col_starts,
                                    (size_t)(gcount + 1) * sizeof(int))))
      return status;
    ws->src_cap = (size_t)gcount;
  }
  if (nnz_cap > ws->nnz_cap) {
    if ((status = opt_realloc_bytes((void **)&ws->pix_idx,
                                    nnz_cap * sizeof(int))))
      return status;
    if ((status = opt_realloc_bytes((void **)&ws->col_vals,
                                    nnz_cap * sizeof(double))))
      return status;
    ws->nnz_cap = nnz_cap;
  }
  if ((size_t)id_cap > ws->id_cap) {
    if ((status = opt_realloc_bytes((void **)&ws->group_ids,
                                    (size_t)id_cap * sizeof(int))))
      return status;
    if ((status = opt_realloc_bytes((void **)&ws->pos_ids,
                                    (size_t)id_cap * sizeof(int))))
      return status;
    if ((status = opt_realloc_bytes((void **)&ws->neg_ids,
                                    (size_t)id_cap * sizeof(int))))
      return status;
    ws->id_cap = (size_t)id_cap;
  }

  return RETURN_OK;
}

static void optimal_group_workspace_free(opt_group_workspace *ws) {
  free(ws->gdata);
  free(ws->gweight);
  free(ws->col_vals);
  free(ws->M);
  free(ws->b);
  free(ws->sol);
  free(ws->work);
  free(ws->pix_idx);
  free(ws->col_starts);
  free(ws->group_ids);
  free(ws->pos_ids);
  free(ws->neg_ids);
}

static void optimal_source_bbox(double x, double y, double r, int64_t *xmin,
                                int64_t *xmax, int64_t *ymin, int64_t *ymax) {
  *xmin = (int64_t)(x - r + 0.5);
  *xmax = (int64_t)(x + r + 1.4999999);
  *ymin = (int64_t)(y - r + 0.5);
  *ymax = (int64_t)(y + r + 1.4999999);
}

static int optimal_bbox_overlap(int64_t axmin, int64_t axmax, int64_t aymin,
                                int64_t aymax, int64_t bxmin, int64_t bxmax,
                                int64_t bymin, int64_t bymax) {
  return (axmin < bxmax && bxmin < axmax && aymin < bymax && bymin < aymax);
}

static double optimal_circle_overlap(double dx, double dy, double r, double r2,
                                     double r_in2, double r_out2, int subpix,
                                     double scale, double scale2,
                                     double offset) {
  double dx1, dy2, rpix2, overlap;
  int64_t sx, sy;

  rpix2 = dx * dx + dy * dy;
  if (!(rpix2 < r_out2)) return 0.0;

  if (rpix2 > r_in2) {
    if (subpix == 0) {
      return circoverlap(dx - 0.5, dy - 0.5, dx + 0.5, dy + 0.5, r);
    }
    dx += offset;
    dy += offset;
    overlap = 0.0;
    for (sy = subpix; sy--; dy += scale) {
      dx1 = dx;
      dy2 = dy * dy;
      for (sx = subpix; sx--; dx1 += scale) {
        if (dx1 * dx1 + dy2 < r2) overlap += scale2;
      }
    }
    return overlap;
  }

  return 1.0;
}

static int optimal_group_lower_bound(const double *x, const int *gidx, int n,
                                     double value) {
  int lo = 0, hi = n;
  while (lo < hi) {
    int mid = lo + (hi - lo) / 2;
    if (x[gidx[mid]] < value) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

static int optimal_group_upper_bound(const double *x, const int *gidx, int n,
                                     double value) {
  int lo = 0, hi = n;
  while (lo < hi) {
    int mid = lo + (hi - lo) / 2;
    if (x[gidx[mid]] <= value) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

static int optimal_group_split_active(const double *x, const double *y,
                                      const double *r, int active_count,
                                      const int *active_idx, int *parent,
                                      int *rank, int *group_id,
                                      int *group_counts, int *group_offsets,
                                      int *group_fill, int *members) {
  double max_r = 0.0;
  int i, j, ngroups;

  for (i = 0; i < active_count; i++) {
    int idx = active_idx[i];
    parent[i] = i;
    rank[i] = 0;
    group_counts[i] = 0;
    group_fill[i] = -1;
    if (r[idx] > max_r) max_r = r[idx];
  }

  for (i = 0; i < active_count; i++) {
    int ii = active_idx[i];
    double xi = x[ii];
    double link_limit = r[ii] + max_r;
    for (j = i + 1; j < active_count; j++) {
      int jj = active_idx[j];
      double dx = x[jj] - xi;
      double dy, rsum;
      if (dx > link_limit) break;
      rsum = r[ii] + r[jj];
      dy = y[ii] - y[jj];
      if (dy > rsum || dy < -rsum) continue;
      if (dx * dx + dy * dy <= rsum * rsum) uf_union(parent, rank, i, j);
    }
  }

  ngroups = 0;
  for (i = 0; i < active_count; i++) {
    int root = uf_find(parent, i);
    if (group_fill[root] < 0) group_fill[root] = ngroups++;
    group_id[i] = group_fill[root];
    group_counts[group_id[i]] += 1;
  }

  group_offsets[0] = 0;
  for (i = 0; i < ngroups; i++) {
    group_offsets[i + 1] = group_offsets[i] + group_counts[i];
    group_fill[i] = group_offsets[i];
  }
  for (i = 0; i < active_count; i++) {
    int gid = group_id[i];
    members[group_fill[gid]++] = active_idx[i];
  }

  return ngroups;
}

static void optimal_mark_tiles_for_bbox(unsigned char *dirty, int nx, int ny,
                                        int64_t xbase, int64_t ybase,
                                        int64_t core_span, int64_t xmin,
                                        int64_t xmax, int64_t ymin,
                                        int64_t ymax) {
  int tx0, tx1, ty0, ty1, tx, ty;

  tx0 = (int)floor(((double)xmin - (double)xbase) / (double)core_span);
  tx1 = (int)floor(((double)(xmax - 1) - (double)xbase) / (double)core_span);
  ty0 = (int)floor(((double)ymin - (double)ybase) / (double)core_span);
  ty1 = (int)floor(((double)(ymax - 1) - (double)ybase) / (double)core_span);

  if (tx0 < 0) tx0 = 0;
  if (ty0 < 0) ty0 = 0;
  if (tx1 >= nx) tx1 = nx - 1;
  if (ty1 >= ny) ty1 = ny - 1;
  if (tx0 > tx1 || ty0 > ty1) return;

  for (ty = ty0; ty <= ty1; ty++) {
    for (tx = tx0; tx <= tx1; tx++) {
      dirty[ty * nx + tx] = 1;
    }
  }
}

static double optimal_sparse_dot(const int *pix_idx, const double *col_vals,
                                 int ia, int ib, int ja, int jb) {
  double acc = 0.0;

  while (ia < ib && ja < jb) {
    int ip = pix_idx[ia];
    int jp = pix_idx[ja];
    if (ip < jp) {
      ia++;
    } else if (jp < ip) {
      ja++;
    } else {
      acc += col_vals[ia] * col_vals[ja];
      ia++;
      ja++;
    }
  }

  return acc;
}

static int optimal_group_solve_compact(
    const sep_image *im, const double *x, const double *y, const double *r,
    const double *r2, const double *r_in2, const double *r_out2,
    const double *sigma_arr, const int64_t *sxmin_arr, const int64_t *sxmax_arr,
    const int64_t *symin_arr, const int64_t *symax_arr,
    const unsigned char *trunc_arr, const int *id, int subpix, short inflag,
    int gcount, const int *gidx, const int *fixed_idx, int fixed_count,
    const double *fixed_flux, opt_group_workspace *ws, double *sum,
    double *sumerr, short *flag) {
  PIXTYPE pix, varpix;
  double dx, dy, overlap, scale, scale2, offset, var;
  int64_t gxmin, gxmax, gymin, gymax;
  int64_t ix, iy;
  size_t nnz_cap = 0;
  size_t gnpix;
  int gw, gh;
  int i, j, k, status, ismasked, has_pos = 0, has_neg = 0;
  int errisarray, errisstd, n_pos = 0, n_neg = 0, nsolve_ids = 0;
  int *group_ids = NULL, *pos_ids = NULL, *neg_ids = NULL;
  double *gdata = NULL, *gweight = NULL, *col_vals = NULL;
  double *M = NULL, *b = NULL, *sol = NULL, *work = NULL;
  int *pix_idx = NULL, *col_starts = NULL;
  converter convert, econvert = NULL, mconvert, sconvert;
  int64_t size, esize, msize, ssize;

  if (gcount <= 0) return RETURN_OK;

  gxmin = im->w;
  gxmax = 0;
  gymin = im->h;
  gymax = 0;
  for (i = 0; i < gcount; i++) {
    int idx = gidx[i];
    int64_t lxmin = sxmin_arr[idx];
    int64_t lxmax = sxmax_arr[idx];
    int64_t lymin = symin_arr[idx];
    int64_t lymax = symax_arr[idx];
    if (trunc_arr && trunc_arr[idx]) flag[idx] |= SEP_APER_TRUNC;
    if (lxmin < 0) lxmin = 0;
    if (lymin < 0) lymin = 0;
    if (lxmax > im->w) lxmax = im->w;
    if (lymax > im->h) lymax = im->h;
    if (lxmin < gxmin) gxmin = lxmin;
    if (lxmax > gxmax) gxmax = lxmax;
    if (lymin < gymin) gymin = lymin;
    if (lymax > gymax) gymax = lymax;
    nnz_cap += (size_t)(lxmax - lxmin) * (size_t)(lymax - lymin);
  }

  if (gxmin >= gxmax || gymin >= gymax) return RETURN_OK;

  gw = (int)(gxmax - gxmin);
  gh = (int)(gymax - gymin);
  gnpix = (size_t)gw * (size_t)gh;

  status = optimal_group_workspace_ensure(
      ws, gnpix, gcount, nnz_cap ? nnz_cap : 1,
      (im->segmap && id) ? (gcount + fixed_count) : 0);
  if (status != RETURN_OK) return status;

  gdata = ws->gdata;
  gweight = ws->gweight;
  col_vals = ws->col_vals;
  M = ws->M;
  b = ws->b;
  sol = ws->sol;
  work = ws->work;
  pix_idx = ws->pix_idx;
  col_starts = ws->col_starts;
  group_ids = ws->group_ids;
  pos_ids = ws->pos_ids;
  neg_ids = ws->neg_ids;

  if ((status = get_converter(im->dtype, &convert, &size))) return status;
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
      if ((status = get_converter(im->ndtype, &econvert, &esize))) return status;
    }
  }

  if (im->segmap && id) {
    for (i = 0; i < gcount; i++) group_ids[nsolve_ids++] = id[gidx[i]];
    for (i = 0; i < fixed_count; i++) group_ids[nsolve_ids++] = id[fixed_idx[i]];
    qsort(group_ids, (size_t)nsolve_ids, sizeof(int), opt_int_cmp);
    for (i = 0; i < nsolve_ids; i++) {
      int gid = group_ids[i];
      if (gid > 0) {
        has_pos = 1;
        pos_ids[n_pos++] = gid;
      } else if (gid < 0) {
        has_neg = 1;
        neg_ids[n_neg++] = -gid;
      }
    }
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

  memset(gweight, 0, gnpix * sizeof(double));

  for (iy = gymin; iy < gymax; iy++) {
    for (ix = gxmin; ix < gxmax; ix++) {
      int pidx = (int)((iy - gymin) * gw + (ix - gxmin));
      int64_t pos = iy * im->w + ix;

      ismasked = 0;
      if (im->mask &&
          (mconvert(MSVC_VOID_CAST im->mask + pos * msize) > im->maskthresh)) {
        ismasked = 1;
      }

      if (im->segmap) {
        int seg_masked = 0;
        double segval = sconvert(MSVC_VOID_CAST im->segmap + pos * ssize);
        if (has_pos) {
          if (segval > 0.0 && !opt_int_contains(pos_ids, n_pos, (int)segval)) {
            seg_masked = 1;
          }
        } else if (has_neg) {
          if (!opt_int_contains(neg_ids, n_neg, (int)segval)) seg_masked = 1;
        }
        if (seg_masked) ismasked = 1;
      }

      pix = convert(MSVC_VOID_CAST im->data + pos * size);
      if (errisarray) {
        varpix = econvert(MSVC_VOID_CAST im->noise + pos * esize);
        if (errisstd) varpix *= varpix;
      } else if (im->noise_type != SEP_NOISE_NONE) {
        varpix = errisstd ? im->noiseval * im->noiseval : im->noiseval;
      } else {
        varpix = 1.0;
      }

      if (varpix <= 0.0) ismasked = 1;
      if (ismasked) continue;

      gweight[pidx] = 1.0 / sqrt(varpix);
      gdata[pidx] = pix * gweight[pidx];
    }
  }

  if (fixed_idx && fixed_flux) {
    for (i = 0; i < fixed_count; i++) {
      int idx = fixed_idx[i];
      int64_t lxmin = sxmin_arr[idx];
      int64_t lxmax = sxmax_arr[idx];
      int64_t lymin = symin_arr[idx];
      int64_t lymax = symax_arr[idx];

      if (fixed_flux[idx] == 0.0) continue;
      if (lxmin < gxmin) lxmin = gxmin;
      if (lymin < gymin) lymin = gymin;
      if (lxmax > gxmax) lxmax = gxmax;
      if (lymax > gymax) lymax = gymax;
      if (lxmin >= lxmax || lymin >= lymax) continue;

      for (iy = lymin; iy < lymax; iy++) {
        int row0 = (int)((iy - gymin) * gw - gxmin);
        for (ix = lxmin; ix < lxmax; ix++) {
          int pidx = row0 + (int)ix;
          if (!(gweight[pidx] > 0.0)) continue;
          dx = ix - x[idx];
          dy = iy - y[idx];
          overlap = optimal_circle_overlap(
              dx, dy, r[idx], r2[idx], r_in2[idx], r_out2[idx], subpix, scale,
              scale2, offset);
          if (overlap <= 0.0) continue;
          gdata[pidx] -= fixed_flux[idx] *
                         gaussian_pixel_integral(dx, dy, sigma_arr[idx]) *
                         gweight[pidx];
        }
      }
    }
  }

  k = 0;
  for (i = 0; i < gcount; i++) {
    int idx = gidx[i];
    int64_t lxmin = sxmin_arr[idx];
    int64_t lxmax = sxmax_arr[idx];
    int64_t lymin = symin_arr[idx];
    int64_t lymax = symax_arr[idx];

    col_starts[i] = k;
    if (lxmin < gxmin) lxmin = gxmin;
    if (lymin < gymin) lymin = gymin;
    if (lxmax > gxmax) lxmax = gxmax;
    if (lymax > gymax) lymax = gymax;

    for (iy = lymin; iy < lymax; iy++) {
      int row0 = (int)((iy - gymin) * gw - gxmin);
      for (ix = lxmin; ix < lxmax; ix++) {
        int pidx = row0 + (int)ix;
        if (!(gweight[pidx] > 0.0)) continue;
        dx = ix - x[idx];
        dy = iy - y[idx];
        overlap = optimal_circle_overlap(
            dx, dy, r[idx], r2[idx], r_in2[idx], r_out2[idx], subpix, scale,
            scale2, offset);
        if (overlap <= 0.0) continue;
        pix_idx[k] = pidx;
        col_vals[k] = gaussian_pixel_integral(dx, dy, sigma_arr[idx]) *
                      gweight[pidx];
        k++;
      }
    }
  }
  col_starts[gcount] = k;

  memset(M, 0, (size_t)gcount * (size_t)gcount * sizeof(double));
  memset(b, 0, (size_t)gcount * sizeof(double));
  for (i = 0; i < gcount; i++) {
    int ia = col_starts[i];
    int ib = col_starts[i + 1];
    int idx = gidx[i];
    for (j = ia; j < ib; j++) b[i] += col_vals[j] * gdata[pix_idx[j]];
    M[i * gcount + i] = optimal_sparse_dot(pix_idx, col_vals, ia, ib, ia, ib);
    for (j = 0; j < i; j++) {
      int jdx = gidx[j];
      if (!optimal_bbox_overlap(sxmin_arr[idx], sxmax_arr[idx], symin_arr[idx],
                                symax_arr[idx], sxmin_arr[jdx], sxmax_arr[jdx],
                                symin_arr[jdx], symax_arr[jdx])) {
        continue;
      }
      double dot = optimal_sparse_dot(pix_idx, col_vals, ia, ib, col_starts[j],
                                      col_starts[j + 1]);
      M[i * gcount + j] = dot;
      M[j * gcount + i] = dot;
    }
  }

  var = 0.0;
  for (i = 0; i < gcount; i++) var += M[i * gcount + i];
  if (!(var > 0.0) || cholesky_decomp(M, gcount)) {
    for (i = 0; i < gcount; i++) {
      int idx = gidx[i];
      double area_tmp;
      status = sep_sum_circle_optimal(
          im, x[idx], y[idx], r[idx], sigma_arr[idx] * 2.354820045,
          id ? id[idx] : 0, subpix, inflag, &sum[idx], &sumerr[idx], &area_tmp,
          &flag[idx]);
      if (status != RETURN_OK) return status;
    }
    return RETURN_OK;
  }

  cholesky_solve(M, b, sol, gcount, work);
  for (i = 0; i < gcount; i++) {
    int idx = gidx[i];
    sum[idx] = sol[i];
  }

  cholesky_inverse_diagonal_inplace(M, gcount, work, b);
  for (i = 0; i < gcount; i++) {
    int idx = gidx[i];
    var = b[i];
    if (var < 0.0) var = 0.0;
    if (im->gain > 0.0 && sum[idx] > 0.0) var += sum[idx] / im->gain;
    sumerr[idx] = sqrt(var);
  }

  return RETURN_OK;
}

static int optimal_group_solve_exact(
    const sep_image *im, const double *x, const double *y, const double *r,
    const double *r2, const double *r_in2, const double *r_out2,
    const double *sigma_arr, const int64_t *sxmin_arr, const int64_t *sxmax_arr,
    const int64_t *symin_arr, const int64_t *symax_arr,
    const unsigned char *trunc_arr, const int *id, int subpix, short inflag,
    int gcount, const int *gidx, const int *fixed_idx, int fixed_count,
    const double *fixed_flux, double *sum, double *sumerr, double *area,
    short *flag) {
  PIXTYPE pix, varpix;
  double dx, dy, overlap, scale, scale2, offset, tmp, var;
  int64_t ix, iy, xmin, xmax, ymin, ymax, size, esize, msize, ssize;
  int i, j, status, ismasked, use_area;
  short errisarray, errisstd;
  converter convert, econvert = NULL, mconvert, sconvert;
  int has_pos = 0, has_neg = 0, n_pos = 0, n_neg = 0, n_group_ids = 0;
  int *pos_ids = NULL, *neg_ids = NULL, *group_ids = NULL;
  int nsolve_ids = 0;
  double *M = NULL, *b = NULL, *work = NULL, *sol = NULL;
  double *totarea = NULL, *maskarea = NULL, *ai = NULL, *overlaps = NULL;

  if (gcount <= 0) return RETURN_OK;

  size = esize = msize = ssize = 0;
  status = RETURN_OK;
  errisarray = 0;
  errisstd = 0;
  use_area = (area != NULL);

  if ((status = get_converter(im->dtype, &convert, &size))) return status;
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
      if ((status = get_converter(im->ndtype, &econvert, &esize))) return status;
    }
  }

  xmin = im->w;
  xmax = 0;
  ymin = im->h;
  ymax = 0;
  for (i = 0; i < gcount; i++) {
    int idx = gidx[i];
    int64_t lxmin = sxmin_arr[idx];
    int64_t lxmax = sxmax_arr[idx];
    int64_t lymin = symin_arr[idx];
    int64_t lymax = symax_arr[idx];
    if (trunc_arr && trunc_arr[idx]) flag[idx] |= SEP_APER_TRUNC;
    if (lxmin < 0) lxmin = 0;
    if (lymin < 0) lymin = 0;
    if (lxmax > im->w) lxmax = im->w;
    if (lymax > im->h) lymax = im->h;
    if (lxmin < xmin) xmin = lxmin;
    if (lxmax > xmax) xmax = lxmax;
    if (lymin < ymin) ymin = lymin;
    if (lymax > ymax) ymax = lymax;
  }

  M = (double *)calloc((size_t)gcount * (size_t)gcount, sizeof(double));
  b = (double *)calloc((size_t)gcount, sizeof(double));
  work = (double *)malloc((size_t)gcount * sizeof(double));
  sol = (double *)malloc((size_t)gcount * sizeof(double));
  ai = (double *)malloc((size_t)gcount * sizeof(double));
  overlaps = (double *)malloc((size_t)gcount * sizeof(double));
  if (use_area) {
    totarea = (double *)calloc((size_t)gcount, sizeof(double));
    maskarea = (double *)calloc((size_t)gcount, sizeof(double));
  }
  if (!M || !b || !work || !sol || !ai || !overlaps ||
      (use_area && (!totarea || !maskarea))) {
    status = MEMORY_ALLOC_ERROR;
    goto cleanup;
  }

  if (im->segmap && id) {
    n_group_ids = gcount + fixed_count;
    group_ids = (int *)malloc((size_t)n_group_ids * sizeof(int));
    pos_ids = (int *)malloc((size_t)n_group_ids * sizeof(int));
    neg_ids = (int *)malloc((size_t)n_group_ids * sizeof(int));
    if (!group_ids || !pos_ids || !neg_ids) {
      status = MEMORY_ALLOC_ERROR;
      goto cleanup;
    }
    for (i = 0; i < gcount; i++) group_ids[nsolve_ids++] = id[gidx[i]];
    for (i = 0; i < fixed_count; i++) group_ids[nsolve_ids++] = id[fixed_idx[i]];
    qsort(group_ids, (size_t)nsolve_ids, sizeof(int), opt_int_cmp);
    for (i = 0; i < nsolve_ids; i++) {
      int gid = group_ids[i];
      if (gid > 0) {
        has_pos = 1;
        pos_ids[n_pos++] = gid;
      } else if (gid < 0) {
        has_neg = 1;
        neg_ids[n_neg++] = -gid;
      }
    }
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

  for (iy = ymin; iy < ymax; iy++) {
    for (ix = xmin; ix < xmax; ix++) {
      int64_t pos = iy * im->w + ix;
      double pix_corr;
      double union_overlap = 0.0;

      ismasked = 0;
      if (im->mask &&
          (mconvert(MSVC_VOID_CAST im->mask + pos * msize) > im->maskthresh)) {
        ismasked = 1;
      }

      if (im->segmap) {
        int seg_masked = 0;
        double segval = sconvert(MSVC_VOID_CAST im->segmap + pos * ssize);
        if (has_pos) {
          if (segval > 0.0 && !opt_int_contains(pos_ids, n_pos, (int)segval)) {
            seg_masked = 1;
          }
        } else if (has_neg) {
          if (!opt_int_contains(neg_ids, n_neg, (int)segval)) seg_masked = 1;
        }
        if (seg_masked) ismasked = 1;
      }

      pix = convert(MSVC_VOID_CAST im->data + pos * size);
      if (errisarray) {
        varpix = econvert(MSVC_VOID_CAST im->noise + pos * esize);
        if (errisstd) varpix *= varpix;
      } else if (im->noise_type != SEP_NOISE_NONE) {
        varpix = errisstd ? im->noiseval * im->noiseval : im->noiseval;
      } else {
        varpix = 1.0;
      }

      if (varpix <= 0.0) ismasked = 1;

      for (i = 0; i < gcount; i++) {
        int idx = gidx[i];
        if (ix < sxmin_arr[idx] || ix >= sxmax_arr[idx] || iy < symin_arr[idx] ||
            iy >= symax_arr[idx]) {
          ai[i] = 0.0;
          overlaps[i] = 0.0;
          continue;
        }
        if (iy < symin_arr[idx] || iy >= symax_arr[idx]) {
          ai[i] = 0.0;
          overlaps[i] = 0.0;
          continue;
        }
        dx = ix - x[idx];
        dy = iy - y[idx];
        overlap = optimal_circle_overlap(
            dx, dy, r[idx], r2[idx], r_in2[idx], r_out2[idx], subpix, scale,
            scale2, offset);
        overlaps[i] = overlap;
        if (overlap > 0.0) {
          if (use_area) {
            totarea[i] += overlap;
            if (ismasked) {
              flag[idx] |= SEP_APER_HASMASKED;
              maskarea[i] += overlap;
            }
          }
          if (overlap > union_overlap) union_overlap = overlap;
          ai[i] = gaussian_pixel_integral(dx, dy, sigma_arr[idx]);
        } else {
          ai[i] = 0.0;
        }
      }

      if (ismasked || union_overlap <= 0.0) continue;

      pix_corr = pix;
      if (fixed_idx && fixed_flux) {
        for (i = 0; i < fixed_count; i++) {
          int idx = fixed_idx[i];
          if (fixed_flux[idx] == 0.0) continue;
          if (ix < sxmin_arr[idx] || ix >= sxmax_arr[idx] || iy < symin_arr[idx] ||
              iy >= symax_arr[idx]) {
            continue;
          }
          if (iy < symin_arr[idx] || iy >= symax_arr[idx]) {
            continue;
          }
          pix_corr -= fixed_flux[idx] * gaussian_pixel_integral(
                                          ix - x[idx], iy - y[idx],
                                          sigma_arr[idx]);
        }
      }

      for (i = 0; i < gcount; i++) {
        if (ai[i] <= 0.0) continue;
        b[i] += ai[i] * pix_corr / varpix;
        for (j = 0; j <= i; j++) {
          if (ai[j] > 0.0) M[i * gcount + j] += ai[i] * ai[j] / varpix;
        }
      }
    }
  }

  for (i = 0; i < gcount; i++) {
    for (j = 0; j < i; j++) M[j * gcount + i] = M[i * gcount + j];
  }

  if (use_area) {
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
  }

  tmp = 0.0;
  for (i = 0; i < gcount; i++) tmp += M[i * gcount + i];
  if (!(tmp > 0.0) || cholesky_decomp(M, gcount)) {
    for (i = 0; i < gcount; i++) {
      int idx = gidx[i];
      double area_tmp;
      double *area_ptr = use_area ? &area[idx] : &area_tmp;
      status = sep_sum_circle_optimal(
          im, x[idx], y[idx], r[idx], sigma_arr[idx] * 2.354820045,
          id ? id[idx] : 0, subpix, inflag, &sum[idx], &sumerr[idx], area_ptr,
          &flag[idx]);
      if (status != RETURN_OK) goto cleanup;
    }
    status = RETURN_OK;
    goto cleanup;
  }

  cholesky_solve(M, b, sol, gcount, work);
  for (i = 0; i < gcount; i++) {
    int idx = gidx[i];
    sum[idx] = sol[i];
  }

  cholesky_inverse_diagonal_inplace(M, gcount, work, b);
  for (i = 0; i < gcount; i++) {
    int idx = gidx[i];
    var = b[i];
    if (var < 0.0) var = 0.0;
    if (im->gain > 0.0 && sum[idx] > 0.0) var += sum[idx] / im->gain;
    sumerr[idx] = sqrt(var);
  }

cleanup:
  free(M);
  free(b);
  free(work);
  free(sol);
  free(totarea);
  free(maskarea);
  free(ai);
  free(overlaps);
  free(group_ids);
  free(pos_ids);
  free(neg_ids);
  return status;
}

static int optimal_group_solve_localized(
    const sep_image *im, const double *x, const double *y, const double *r,
    const double *r2, const double *r_in2, const double *r_out2,
    const double *sigma_arr, const int64_t *sxmin_arr, const int64_t *sxmax_arr,
    const int64_t *symin_arr, const int64_t *symax_arr,
    const unsigned char *trunc_arr, const int *id, double halo_factor,
    int subpix, short inflag, int gcount, const int *gidx, double *sum,
    double *sumerr, double *area, short *flag) {
  int status = RETURN_OK;
  int i, sweep, max_idx = -1, max_sweeps, is_large_group;
  double local_radius = 0.0, max_support_radius = 0.0, eff_halo_factor;
  int64_t group_xmin, group_xmax, group_ymin, group_ymax, core_span, core_shift;
  double *work_flux = NULL, *tmp_flux = NULL, *tmp_fluxerr = NULL;
  int *core_idx = NULL, *active_idx = NULL, *fixed_idx = NULL;
  int *split_parent = NULL, *split_rank = NULL, *split_group_id = NULL;
  int *split_counts = NULL, *split_offsets = NULL, *split_fill = NULL;
  int *split_members = NULL, *sub_fixed_idx = NULL;
  int nx_arr[4] = {0}, ny_arr[4] = {0}, ntilings = 0;
  int64_t xbase_arr[4] = {0}, ybase_arr[4] = {0};
  unsigned char *dirty_cur[4] = {NULL}, *dirty_next[4] = {NULL};
  unsigned char *well_centered = NULL;
  short *tmp_flag = NULL;
  opt_group_workspace ws = {0};

  group_xmin = im->w;
  group_xmax = 0;
  group_ymin = im->h;
  group_ymax = 0;
  for (i = 0; i < gcount; i++) {
    int idx = gidx[i];
    if (idx > max_idx) max_idx = idx;
    if (r[idx] > max_support_radius) max_support_radius = r[idx];
    if (sxmin_arr[idx] < group_xmin) group_xmin = sxmin_arr[idx];
    if (sxmax_arr[idx] > group_xmax) group_xmax = sxmax_arr[idx];
    if (symin_arr[idx] < group_ymin) group_ymin = symin_arr[idx];
    if (symax_arr[idx] > group_ymax) group_ymax = symax_arr[idx];
    status = sep_sum_circle_optimal(
        im, x[idx], y[idx], r[idx], sigma_arr[idx] * 2.354820045,
        id ? id[idx] : 0, subpix, inflag, &sum[idx], &sumerr[idx], &area[idx],
        &flag[idx]);
    if (status != RETURN_OK) return status;
  }

  is_large_group = (gcount > OPT_GROUP_EDGE_REFINE_MAX);
  eff_halo_factor = (halo_factor < 1.0) ? 1.0 : halo_factor;
  local_radius = 2.0 * max_support_radius * eff_halo_factor;
  core_span = (int64_t)(4.0 * local_radius + 0.5);
  if (core_span < 1) core_span = 1;
  core_shift = core_span / 2;
  work_flux = (double *)calloc((size_t)max_idx + 1, sizeof(double));
  tmp_flux = (double *)calloc((size_t)max_idx + 1, sizeof(double));
  tmp_fluxerr = (double *)calloc((size_t)max_idx + 1, sizeof(double));
  tmp_flag = (short *)calloc((size_t)max_idx + 1, sizeof(short));
  core_idx = (int *)malloc((size_t)gcount * sizeof(int));
  active_idx = (int *)malloc((size_t)gcount * sizeof(int));
  fixed_idx = (int *)malloc((size_t)gcount * sizeof(int));
  split_parent = (int *)malloc((size_t)gcount * sizeof(int));
  split_rank = (int *)malloc((size_t)gcount * sizeof(int));
  split_group_id = (int *)malloc((size_t)gcount * sizeof(int));
  split_counts = (int *)malloc((size_t)gcount * sizeof(int));
  split_offsets = (int *)malloc((size_t)(gcount + 1) * sizeof(int));
  split_fill = (int *)malloc((size_t)gcount * sizeof(int));
  split_members = (int *)malloc((size_t)gcount * sizeof(int));
  sub_fixed_idx = (int *)malloc((size_t)gcount * sizeof(int));
  well_centered = (unsigned char *)calloc((size_t)max_idx + 1, sizeof(unsigned char));
  if (!work_flux || !tmp_flux || !tmp_fluxerr || !tmp_flag || !core_idx || !active_idx ||
      !fixed_idx || !split_parent || !split_rank || !split_group_id ||
      !split_counts || !split_offsets || !split_fill || !split_members ||
      !sub_fixed_idx || !well_centered) {
    status = MEMORY_ALLOC_ERROR;
    goto cleanup;
  }

  {
    int64_t x_offsets[2] = {0, core_shift};
    int64_t y_offsets[2] = {0, core_shift};
    int nxoff = (!is_large_group && core_shift > 0) ? 2 : 1;
    int nyoff = (!is_large_group && core_shift > 0) ? 2 : 1;
    int xoi, yoi;

    for (xoi = 0; xoi < nxoff; xoi++) {
      for (yoi = 0; yoi < nyoff; yoi++) {
        int t = xoi * nyoff + yoi;
        int ntiles;
        xbase_arr[t] = group_xmin - x_offsets[xoi];
        ybase_arr[t] = group_ymin - y_offsets[yoi];
        nx_arr[t] =
            (int)((group_xmax - xbase_arr[t] + core_span - 1) / core_span);
        ny_arr[t] =
            (int)((group_ymax - ybase_arr[t] + core_span - 1) / core_span);
        ntiles = nx_arr[t] * ny_arr[t];
        dirty_cur[t] = (unsigned char *)malloc((size_t)ntiles);
        dirty_next[t] = (unsigned char *)calloc((size_t)ntiles, sizeof(unsigned char));
        if (!dirty_cur[t] || !dirty_next[t]) {
          status = MEMORY_ALLOC_ERROR;
          goto cleanup;
        }
        memset(dirty_cur[t], 1, (size_t)ntiles);
        ntilings++;
      }
    }
  }

  for (i = 0; i < gcount; i++) {
    int idx = gidx[i];
    work_flux[idx] = sum[idx];
  }

  max_sweeps = is_large_group ? 1 : OPT_GROUP_MAX_SWEEPS;
  for (sweep = 0; sweep < max_sweeps; sweep++) {
    int converged = 1;
    int pass, npasses;

    for (i = 0; i < ntilings; i++) {
      memset(dirty_next[i], 0, (size_t)nx_arr[i] * (size_t)ny_arr[i]);
    }

    npasses = is_large_group ? 1 : 2;
    for (pass = 0; pass < npasses; pass++) {
      int nxoff = (!is_large_group && core_shift > 0) ? 2 : 1;
      int nyoff = (!is_large_group && core_shift > 0) ? 2 : 1;
      int xoi, yoi;

      for (xoi = 0; xoi < nxoff; xoi++) {
        int xsel = pass == 0 ? xoi : (nxoff - 1 - xoi);
        int nx = nx_arr[xsel * nyoff];
        int xi_start = pass == 0 ? 0 : (nx - 1);
        int xi_stop = pass == 0 ? nx : -1;
        int xi_step = pass == 0 ? 1 : -1;

        for (yoi = 0; yoi < nyoff; yoi++) {
          int ysel = pass == 0 ? yoi : (nyoff - 1 - yoi);
          int t = xsel * nyoff + ysel;
          int64_t xbase = xbase_arr[t];
          int64_t ybase = ybase_arr[t];
          int nx = nx_arr[t];
          int ny = ny_arr[t];
          int yi_start = pass == 0 ? 0 : (ny - 1);
          int yi_stop = pass == 0 ? ny : -1;
          int yi_step = pass == 0 ? 1 : -1;

          for (i = xi_start; i != xi_stop; i += xi_step) {
            int yi;
            int64_t core_xmin = xbase + (int64_t)i * core_span;
            int64_t core_xmax = core_xmin + core_span;

            for (yi = yi_start; yi != yi_stop; yi += yi_step) {
              int left, right, core_count, active_count, fixed_count, k;
              int tile_idx = yi * nx + i;
              int64_t core_ymin = ybase + (int64_t)yi * core_span;
              int64_t core_ymax = core_ymin + core_span;
              int64_t ext_xmin = core_xmin - (int64_t)(local_radius + 0.5);
              int64_t ext_xmax = core_xmax + (int64_t)(local_radius + 0.5);
              int64_t ext_ymin = core_ymin - (int64_t)(local_radius + 0.5);
              int64_t ext_ymax = core_ymax + (int64_t)(local_radius + 0.5);

              if (!dirty_cur[t][tile_idx]) continue;

              left = optimal_group_lower_bound(x, gidx, gcount,
                                               (double)ext_xmin - max_support_radius);
              right = optimal_group_upper_bound(x, gidx, gcount,
                                                (double)ext_xmax + max_support_radius);
              core_count = 0;
              active_count = 0;
              fixed_count = 0;

              for (k = left; k < right; k++) {
                int idx = gidx[k];
                int in_core, in_ext;
                in_core = optimal_bbox_overlap(core_xmin, core_xmax, core_ymin,
                                               core_ymax, sxmin_arr[idx],
                                               sxmax_arr[idx], symin_arr[idx],
                                               symax_arr[idx]);
                in_ext = optimal_bbox_overlap(ext_xmin, ext_xmax, ext_ymin,
                                              ext_ymax, sxmin_arr[idx],
                                              sxmax_arr[idx], symin_arr[idx],
                                              symax_arr[idx]);
                if (in_core) {
                  core_idx[core_count++] = idx;
                  active_idx[active_count++] = idx;
                  if ((double)(sxmin_arr[idx] - core_xmin) > local_radius &&
                      (double)(core_xmax - sxmax_arr[idx]) > local_radius &&
                      (double)(symin_arr[idx] - core_ymin) > local_radius &&
                      (double)(core_ymax - symax_arr[idx]) > local_radius) {
                    well_centered[idx] = 1;
                  }
                } else if (in_ext) {
                  fixed_idx[fixed_count++] = idx;
                }
              }

              if (core_count == 0 || active_count == 0) continue;

              if (active_count > 1) {
                int nsub = optimal_group_split_active(
                    x, y, r, active_count, active_idx, split_parent, split_rank,
                    split_group_id, split_counts, split_offsets, split_fill,
                    split_members);
                if (nsub > 1) {
                  int changed_count = 0;

                  for (k = 0; k < nsub; k++) {
                    const int *sub_active = split_members + split_offsets[k];
                    int sub_active_count = split_counts[k];
                    int sub_fixed_count = 0;
                    int si;
                    int64_t sub_xmin = im->w, sub_xmax = 0;
                    int64_t sub_ymin = im->h, sub_ymax = 0;
                    int64_t ext_xmin, ext_xmax, ext_ymin, ext_ymax;

                    for (si = 0; si < sub_active_count; si++) {
                      int idx = sub_active[si];
                      if (sxmin_arr[idx] < sub_xmin) sub_xmin = sxmin_arr[idx];
                      if (sxmax_arr[idx] > sub_xmax) sub_xmax = sxmax_arr[idx];
                      if (symin_arr[idx] < sub_ymin) sub_ymin = symin_arr[idx];
                      if (symax_arr[idx] > sub_ymax) sub_ymax = symax_arr[idx];
                      tmp_flux[idx] = work_flux[idx];
                      tmp_fluxerr[idx] = sumerr[idx];
                      tmp_flag[idx] = flag[idx];
                    }

                    ext_xmin = sub_xmin - (int64_t)(local_radius + 0.5);
                    ext_xmax = sub_xmax + (int64_t)(local_radius + 0.5);
                    ext_ymin = sub_ymin - (int64_t)(local_radius + 0.5);
                    ext_ymax = sub_ymax + (int64_t)(local_radius + 0.5);

                    for (si = 0; si < fixed_count; si++) {
                      int idx = fixed_idx[si];
                      if (!optimal_bbox_overlap(ext_xmin, ext_xmax, ext_ymin,
                                                ext_ymax, sxmin_arr[idx],
                                                sxmax_arr[idx], symin_arr[idx],
                                                symax_arr[idx])) {
                        continue;
                      }
                      sub_fixed_idx[sub_fixed_count++] = idx;
                    }

                    status = optimal_group_solve_compact(
                        im, x, y, r, r2, r_in2, r_out2, sigma_arr, sxmin_arr,
                        sxmax_arr, symin_arr, symax_arr, trunc_arr, id, subpix,
                        inflag, sub_active_count, sub_active, sub_fixed_idx,
                        sub_fixed_count, work_flux, &ws, tmp_flux, tmp_fluxerr,
                        tmp_flag);
                    if (status != RETURN_OK) goto cleanup;

                    for (si = 0; si < sub_active_count; si++) {
                      int idx = sub_active[si];
                      double old_flux = work_flux[idx];
                      double new_flux = tmp_flux[idx];
                      double tol = OPT_GROUP_CONV_ATOL +
                                   OPT_GROUP_CONV_RTOL *
                                       (fabs(old_flux) > fabs(new_flux)
                                            ? fabs(old_flux)
                                            : fabs(new_flux));
                      if (fabs(new_flux - old_flux) > tol) {
                        converged = 0;
                        core_idx[changed_count++] = idx;
                      }
                      work_flux[idx] = tmp_flux[idx];
                      sum[idx] = tmp_flux[idx];
                      sumerr[idx] = tmp_fluxerr[idx];
                      flag[idx] = tmp_flag[idx];
                    }
                  }

                  for (k = 0; k < changed_count; k++) {
                    int idx = core_idx[k];
                    int64_t mark_xmin, mark_xmax, mark_ymin, mark_ymax;
                    mark_xmin = sxmin_arr[idx] - (int64_t)(local_radius + 0.5);
                    mark_xmax = sxmax_arr[idx] + (int64_t)(local_radius + 0.5);
                    mark_ymin = symin_arr[idx] - (int64_t)(local_radius + 0.5);
                    mark_ymax = symax_arr[idx] + (int64_t)(local_radius + 0.5);
                    optimal_mark_tiles_for_bbox(dirty_next[t], nx, ny, xbase, ybase,
                                                core_span, mark_xmin, mark_xmax,
                                                mark_ymin, mark_ymax);
                  }
                  continue;
                }
              }

              for (k = 0; k < active_count; k++) {
                int idx = active_idx[k];
                tmp_flux[idx] = work_flux[idx];
                tmp_fluxerr[idx] = sumerr[idx];
                tmp_flag[idx] = flag[idx];
              }

              status = optimal_group_solve_compact(
                  im, x, y, r, r2, r_in2, r_out2, sigma_arr, sxmin_arr, sxmax_arr,
                  symin_arr, symax_arr, trunc_arr, id, subpix, inflag,
                  active_count, active_idx, fixed_idx, fixed_count, work_flux,
                  &ws, tmp_flux, tmp_fluxerr, tmp_flag);
              if (status != RETURN_OK) goto cleanup;

              fixed_count = 0;
              for (k = 0; k < active_count; k++) {
                int idx = active_idx[k];
                double old_flux = work_flux[idx];
                double new_flux = tmp_flux[idx];
                double tol = OPT_GROUP_CONV_ATOL +
                             OPT_GROUP_CONV_RTOL *
                                 (fabs(old_flux) > fabs(new_flux) ? fabs(old_flux)
                                                                  : fabs(new_flux));
                if (fabs(new_flux - old_flux) > tol) {
                  converged = 0;
                  fixed_idx[fixed_count++] = idx;
                }
              }
              for (k = 0; k < active_count; k++) {
                int idx = active_idx[k];
                work_flux[idx] = tmp_flux[idx];
              }
              for (k = 0; k < core_count; k++) {
                int idx = core_idx[k];
                sum[idx] = tmp_flux[idx];
                sumerr[idx] = tmp_fluxerr[idx];
                flag[idx] = tmp_flag[idx];
              }
              for (k = 0; k < fixed_count; k++) {
                int idx = fixed_idx[k];
                int64_t mark_xmin, mark_xmax, mark_ymin, mark_ymax;
                mark_xmin = sxmin_arr[idx] - (int64_t)(local_radius + 0.5);
                mark_xmax = sxmax_arr[idx] + (int64_t)(local_radius + 0.5);
                mark_ymin = symin_arr[idx] - (int64_t)(local_radius + 0.5);
                mark_ymax = symax_arr[idx] + (int64_t)(local_radius + 0.5);
                optimal_mark_tiles_for_bbox(dirty_next[t], nx, ny, xbase, ybase,
                                            core_span, mark_xmin, mark_xmax,
                                            mark_ymin, mark_ymax);
              }
            }
          }
        }
      }
    }

    if (converged) break;
    for (i = 0; i < ntilings; i++) {
      unsigned char *tmp = dirty_cur[i];
      dirty_cur[i] = dirty_next[i];
      dirty_next[i] = tmp;
    }
  }

  if (gcount <= OPT_GROUP_EDGE_REFINE_MAX) {
    for (i = 0; i < gcount; i++) {
      int idx = gidx[i];
      int left, right, active_count, fixed_count, k;
      int64_t core_xmin, core_xmax, core_ymin, core_ymax;

      if (well_centered[idx]) continue;

      core_xmin = sxmin_arr[idx];
      core_xmax = sxmax_arr[idx];
      core_ymin = symin_arr[idx];
      core_ymax = symax_arr[idx];
      {
        int64_t ext_xmin = core_xmin - (int64_t)(local_radius + 0.5);
        int64_t ext_xmax = core_xmax + (int64_t)(local_radius + 0.5);
        int64_t ext_ymin = core_ymin - (int64_t)(local_radius + 0.5);
        int64_t ext_ymax = core_ymax + (int64_t)(local_radius + 0.5);

        left = optimal_group_lower_bound(x, gidx, gcount,
                                         (double)ext_xmin - max_support_radius);
        right = optimal_group_upper_bound(x, gidx, gcount,
                                          (double)ext_xmax + max_support_radius);

        active_count = 0;
        fixed_count = 0;
        for (k = left; k < right; k++) {
          int jdx = gidx[k];
          if (optimal_bbox_overlap(core_xmin, core_xmax, core_ymin, core_ymax,
                                   sxmin_arr[jdx], sxmax_arr[jdx], symin_arr[jdx],
                                   symax_arr[jdx])) {
            active_idx[active_count++] = jdx;
          } else if (optimal_bbox_overlap(ext_xmin, ext_xmax, ext_ymin, ext_ymax,
                                          sxmin_arr[jdx], sxmax_arr[jdx],
                                          symin_arr[jdx], symax_arr[jdx])) {
            fixed_idx[fixed_count++] = jdx;
          }
        }
      }

      for (k = 0; k < active_count; k++) {
        int jdx = active_idx[k];
        tmp_flux[jdx] = work_flux[jdx];
        tmp_fluxerr[jdx] = sumerr[jdx];
        tmp_flag[jdx] = flag[jdx];
      }

      status = optimal_group_solve_exact(
          im, x, y, r, r2, r_in2, r_out2, sigma_arr, sxmin_arr, sxmax_arr,
          symin_arr, symax_arr, trunc_arr, id, subpix, inflag, active_count,
          active_idx, fixed_idx, fixed_count, work_flux, tmp_flux, tmp_fluxerr,
          area, tmp_flag);
      if (status != RETURN_OK) goto cleanup;

      work_flux[idx] = tmp_flux[idx];
      sum[idx] = tmp_flux[idx];
      sumerr[idx] = tmp_fluxerr[idx];
      flag[idx] = tmp_flag[idx];
    }
  }

cleanup:
  free(work_flux);
  free(tmp_flux);
  free(tmp_fluxerr);
  free(core_idx);
  free(active_idx);
  free(fixed_idx);
  free(split_parent);
  free(split_rank);
  free(split_group_id);
  free(split_counts);
  free(split_offsets);
  free(split_fill);
  free(split_members);
  free(sub_fixed_idx);
  for (i = 0; i < ntilings; i++) {
    free(dirty_cur[i]);
    free(dirty_next[i]);
  }
  free(well_centered);
  free(tmp_flag);
  optimal_group_workspace_free(&ws);
  return status;
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

        if (overlap > 0.0) {
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
              double scale = overlap;
              double pix_eff = pix * scale;
              double var_eff = varpix * scale * scale;
              psf = gaussian_pixel_integral(dx0, dy0, sigma) * scale;
              if (var_eff > 0.0) {
                num += psf * pix_eff / var_eff;
                den += psf * psf / var_eff;
              } else {
                *flag |= SEP_APER_HASMASKED;
                maskarea += overlap;
              }
            } else {
              *flag |= SEP_APER_HASMASKED;
              maskarea += overlap;
            }
          }

          totarea += overlap;
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
    double group_factor,
    double halo_factor,
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
  double dx, dy, dist2, max_r, link_limit, rsum;
  int i, j, g, gi, ngroups;
  int status, use_bkg;
  int *parent = NULL, *rank = NULL, *group_id = NULL, *root_map = NULL;
  int *group_counts = NULL, *group_offsets = NULL, *group_fill = NULL;
  int *members = NULL;
  double *r2 = NULL, *r_in2 = NULL, *r_out2 = NULL, *sigma_arr = NULL;
  int64_t *sxmin_arr = NULL, *sxmax_arr = NULL, *symin_arr = NULL, *symax_arr = NULL;
  unsigned char *trunc_arr = NULL;
  opt_xorder_entry *xorder = NULL;
  opt_group_order_entry *group_order = NULL;

  if (n < 1) return ILLEGAL_APER_PARAMS;
  if (!(group_factor > 0.0)) return ILLEGAL_APER_PARAMS;
  if (!(halo_factor > 0.0)) return ILLEGAL_APER_PARAMS;
  if (subpix < 0) return ILLEGAL_SUBPIX;

  use_bkg = (bkg_mean != NULL);
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
  sxmin_arr = (int64_t *)malloc((size_t)n * sizeof(int64_t));
  sxmax_arr = (int64_t *)malloc((size_t)n * sizeof(int64_t));
  symin_arr = (int64_t *)malloc((size_t)n * sizeof(int64_t));
  symax_arr = (int64_t *)malloc((size_t)n * sizeof(int64_t));
  trunc_arr = (unsigned char *)malloc((size_t)n * sizeof(unsigned char));
  xorder = (opt_xorder_entry *)malloc((size_t)n * sizeof(opt_xorder_entry));

  if (!parent || !rank || !group_id || !root_map || !group_counts ||
      !group_offsets || !group_fill || !members || !r2 || !r_in2 || !r_out2 ||
      !sigma_arr || !sxmin_arr || !sxmax_arr || !symin_arr || !symax_arr ||
      !trunc_arr || !xorder) {
    status = MEMORY_ALLOC_ERROR;
    goto cleanup;
  }

  max_r = 0.0;
  for (i = 0; i < n; i++) {
    parent[i] = i;
    if (r[i] < 0.0 || !(fwhm[i] > 0.0)) {
      status = ILLEGAL_APER_PARAMS;
      goto cleanup;
    }
    r2[i] = r[i] * r[i];
    oversamp_ann_circle(r[i], &r_in2[i], &r_out2[i]);
    sigma_arr[i] = fwhm[i] / 2.354820045;
    if (!(sigma_arr[i] > 0.0)) {
      status = ILLEGAL_APER_PARAMS;
      goto cleanup;
    }
    optimal_source_bbox(x[i], y[i], r[i], &sxmin_arr[i], &sxmax_arr[i],
                        &symin_arr[i], &symax_arr[i]);
    trunc_arr[i] = (sxmin_arr[i] < 0 || sxmax_arr[i] > im->w || symin_arr[i] < 0 ||
                    symax_arr[i] > im->h)
                       ? 1
                       : 0;
    if (r[i] > max_r) max_r = r[i];
    xorder[i].x = x[i];
    xorder[i].idx = i;
    flag[i] = trunc_arr[i] ? SEP_APER_TRUNC : 0;
    sum[i] = 0.0;
    sumerr[i] = 0.0;
    area[i] = 0.0;
  }

  qsort(xorder, (size_t)n, sizeof(*xorder), opt_xorder_cmp);

  for (i = 0; i < n; i++) {
    int ii = xorder[i].idx;
    double xi = xorder[i].x;
    link_limit = group_factor * (r[ii] + max_r);
    for (j = i + 1; j < n; j++) {
      int jj = xorder[j].idx;
      dx = xorder[j].x - xi;
      if (dx > link_limit) break;
      rsum = group_factor * (r[ii] + r[jj]);
      dy = y[ii] - y[jj];
      if (dy > rsum || dy < -rsum) continue;
      dist2 = dx * dx + dy * dy;
      if (dist2 <= rsum * rsum) uf_union(parent, rank, ii, jj);
    }
  }

  for (i = 0; i < n; i++) root_map[i] = -1;
  g = 0;
  for (i = 0; i < n; i++) {
    int root = uf_find(parent, i);
    if (root_map[root] < 0) root_map[root] = g++;
    group_id[i] = root_map[root];
    group_counts[group_id[i]] += 1;
  }
  ngroups = g;

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

  group_order =
      (opt_group_order_entry *)malloc((size_t)ngroups * sizeof(*group_order));
  if (!group_order) {
    status = MEMORY_ALLOC_ERROR;
    goto cleanup;
  }
  for (i = 0; i < ngroups; i++) {
    group_order[i].gid = i;
    group_order[i].count = group_counts[i];
  }
  qsort(group_order, (size_t)ngroups, sizeof(*group_order), opt_group_order_cmp);

  for (gi = 0; gi < ngroups; gi++) {
    int gid = group_order[gi].gid;
    int gcount = group_counts[gid];
    const int *gidx = members + group_offsets[gid];
    double group_mean = 0.0;
    double group_err = 0.0;

    if (use_bkg) {
      double wsum = 0.0;
      double werr2 = 0.0;
      for (i = 0; i < gcount; i++) {
        int idx = gidx[i];
        double w = bkg_weight ? bkg_weight[idx] : 1.0;
        if (!(w > 0.0)) continue;
        if (!(bkg_mean[idx] == bkg_mean[idx])) continue;
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
      if (bkg_mean_err) group_err = sqrt(werr2) / wsum;
    }

    if (gcount == 1) {
      int idx = gidx[0];
      status = sep_sum_circle_optimal(
          im, x[idx], y[idx], r[idx], fwhm[idx], id ? id[idx] : 0, subpix,
          inflag, &sum[idx], &sumerr[idx], &area[idx], &flag[idx]);
    } else if (gcount <= OPT_GROUP_EXACT_MAX) {
      status = optimal_group_solve_exact(
          im, x, y, r, r2, r_in2, r_out2, sigma_arr, sxmin_arr, sxmax_arr,
          symin_arr, symax_arr, trunc_arr, id, subpix, inflag, gcount, gidx,
          NULL, 0, NULL, sum, sumerr, area, flag);
    } else {
      status = optimal_group_solve_localized(
          im, x, y, r, r2, r_in2, r_out2, sigma_arr, sxmin_arr, sxmax_arr,
          symin_arr, symax_arr, trunc_arr, id, halo_factor, subpix, inflag,
          gcount, gidx, sum, sumerr, area, flag);
    }
    if (status != RETURN_OK) goto cleanup;

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
  }

  status = RETURN_OK;

cleanup:
  free(group_order);
  free(xorder);
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
  free(sxmin_arr);
  free(sxmax_arr);
  free(symin_arr);
  free(symax_arr);
  free(trunc_arr);
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
    double group_factor,
    double halo_factor,
    int subpix,
    short inflag,
    double * sum,
    double * sumerr,
    double * area,
    short * flag
) {
  return sep_sum_circle_optimal_multi_impl(
      im, x, y, r, fwhm, n, id, group_factor, halo_factor, subpix, inflag,
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
    double group_factor,
    double halo_factor,
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
      im, x, y, r, fwhm, n, id, group_factor, halo_factor, subpix, inflag,
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
/* elliptical annulus statistics */

int sep_stats_ellipann(
    const sep_image * im,
    double x,
    double y,
    double a,
    double b,
    double theta,
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
  double cxx, cyy, cxy;
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

  if (!(rin >= 0.0 && rout >= rin && b >= 0.0 && a >= b && theta >= -PI / 2.
        && theta <= PI / 2.))
  {
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
  oversamp_ann_ellipse(rin, b, &rin_in2, &rin_out2);
  oversamp_ann_ellipse(rout, b, &rout_in2, &rout_out2);
  sep_ellipse_coeffs(a, b, theta, &cxx, &cyy, &cxy);

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

  if ((status = get_converter(im->dtype, &convert, &size))) {
    return status;
  }
  if (im->mask && (status = get_converter(im->mdtype, &mconvert, &msize))) {
    return status;
  }
  if (im->segmap && (status = get_converter(im->sdtype, &sconvert, &ssize))) {
    return status;
  }

  boxextent_ellipse(x, y, cxx, cyy, cxy, rout, im->w, im->h, &xmin, &xmax, &ymin, &ymax,
                    flag);

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
      rpix2 = cxx * dx * dx + cyy * dy * dy + cxy * dx * dy;
      if ((rpix2 < rout_out2) && (rpix2 > rin_in2)) {
        if ((rpix2 > rout_in2) || (rpix2 < rin_out2)) {
          if (subpix == 0) {
            overlap =
                ellipoverlap(dx - 0.5, dy - 0.5, dx + 0.5, dy + 0.5, a * rout,
                             b * rout, theta)
                - ellipoverlap(dx - 0.5, dy - 0.5, dx + 0.5, dy + 0.5, a * rin,
                               b * rin, theta);
          } else {
            dx += offset;
            dy += offset;
            overlap = 0.0;
            for (sy = subpix; sy--; dy += scale) {
              dx1 = dx;
              dy2 = dy * dy;
              for (sx = subpix; sx--; dx1 += scale) {
                rpix2 = cxx * dx1 * dx1 + cyy * dy2 + cxy * dx1 * dy;
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

  qsort(vw, (size_t)nvals, sizeof(valweight), cmp_valweight);
  *median = weighted_median_sorted(vw, nvals, totw);

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

#ifdef _OPENMP
static void winpos_psf_workspace_nullify(sep_psf *psf) {
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

static void winpos_psf_workspace_free(sep_psf *psf) {
  if (!psf) return;
  free(psf->loc);
  free(psf->resi);
  free(psf->interp_mask);
  free(psf->interp_nmask);
  free(psf->interp_start);
  free(psf->interp_buf);
  winpos_psf_workspace_nullify(psf);
}

static int winpos_psf_workspace_clone(const sep_psf *src, sep_psf *dst) {
  int status = RETURN_OK;
  int npix, rnpix;

  memset(dst, 0, sizeof(*dst));
  *dst = *src;
  winpos_psf_workspace_nullify(dst);

  /* Shared read-only model arrays */
  dst->data = src->data;
  dst->interp_lut = src->interp_lut;

  npix = src->w * src->h;
  rnpix = src->rw * src->rh;

  QMALLOC(dst->loc, float, npix, status);
  QMALLOC(dst->resi, float, rnpix, status);
  QMALLOC(dst->interp_mask, float, src->interp_mask_len, status);
  QMALLOC(dst->interp_nmask, int, src->interp_nmask_len, status);
  QMALLOC(dst->interp_start, int, src->interp_nmask_len, status);
  QMALLOC(dst->interp_buf, float, src->interp_buf_len, status);

  return RETURN_OK;

exit:
  winpos_psf_workspace_free(dst);
  return status;
}
#endif

int sep_windowed_psf(
    const sep_image * im,
    sep_psf * psf,
    double x,
    double y,
    short inflag,
    int id,
    double maxstep,
    double * xout,
    double * yout,
    int * niter,
    short * flag
) {
  PIXTYPE pix;
  double dx, dy, dxpos, dypos, tmp, twv, tv, step, step_scale, weight;
  double maskarea, maskweight, maskdxpos, maskdypos, totarea;
  int64_t imx, imy, pos, size, msize, ssize;
  int i, sx, sy, status, ismasked;
  int ix0, iy0;
  const BYTE *datat, *maskt, *segt;
  converter convert, mconvert, sconvert;

  if (psf == NULL) {
    return ILLEGAL_APER_PARAMS;
  }

  *flag = 0;
  *xout = x;
  *yout = y;
  *niter = 0;
  status = RETURN_OK;
  datat = maskt = segt = NULL;
  size = msize = ssize = 0;

  if ((status = get_converter(im->dtype, &convert, &size))) {
    return status;
  }
  if (im->mask && (status = get_converter(im->mdtype, &mconvert, &msize))) {
    return status;
  }
  if (im->segmap && (status = get_converter(im->sdtype, &sconvert, &ssize))) {
    return status;
  }

  for (i = 0; i < WINPOS_NITERMAX; i++) {
    status = sep_psf_build(psf, x, y);
    if (status != RETURN_OK) {
      return status;
    }

    ix0 = (int)(x + 0.5);
    iy0 = (int)(y + 0.5);
    status = sep_psf_resample(psf, x - ix0, y - iy0);
    if (status != RETURN_OK) {
      return status;
    }

    tv = twv = 0.0;
    dxpos = dypos = 0.0;
    maskarea = maskweight = 0.0;
    maskdxpos = maskdypos = 0.0;
    totarea = 0.0;

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

        weight = psf->resi[sy * psf->rw + sx];
        if (weight == 0.0) {
          continue;
        }

        pos = imy * im->w + imx;
        datat = MSVC_VOID_CAST im->data + pos * size;
        dx = (double)imx - x;
        dy = (double)imy - y;

        ismasked = 0;
        if (im->mask) {
          maskt = MSVC_VOID_CAST im->mask + pos * msize;
          if (mconvert(maskt) > im->maskthresh) {
            ismasked = 1;
          }
        }

        if (im->segmap) {
          segt = MSVC_VOID_CAST im->segmap + pos * ssize;
          if (id > 0) {
            if ((sconvert(segt) > 0.) && (sconvert(segt) != id)) {
              ismasked = 1;
            }
          } else if (id < 0) {
            if (sconvert(segt) != -1 * id) {
              ismasked = 1;
            }
          }
        }

        if (ismasked) {
          *flag |= SEP_APER_HASMASKED;
          maskarea += 1.0;
          maskweight += weight;
          maskdxpos += weight * dx;
          maskdypos += weight * dy;
        } else {
          pix = convert(datat);
          tv += pix;
          twv += pix * weight;
          dxpos += pix * weight * dx;
          dypos += pix * weight * dy;
        }

        totarea += 1.0;
      }
    }

    if (im->mask || im->segmap) {
      if (totarea > 0.0 && maskarea >= totarea) {
        *flag |= SEP_APER_ALLMASKED;
        break;
      }
      if (inflag & SEP_MASK_IGNORE) {
        totarea -= maskarea;
      } else if (totarea > maskarea) {
        tmp = tv / (totarea - maskarea);
        twv += tmp * maskweight;
        dxpos += tmp * maskdxpos;
        dypos += tmp * maskdypos;
      }
    }

    if (twv > 0.0) {
      dxpos /= twv;
      dypos /= twv;
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
      *flag |= SEP_APER_NONPOSITIVE;
      break;
    }

    if (dxpos * dxpos + dypos * dypos < WINPOS_STEPMIN * WINPOS_STEPMIN) {
      break;
    }
  }

  *xout = x;
  *yout = y;
  *niter = i + 1;

  return status;
}

int sep_windowed_psf_array(const sep_image *im, sep_psf *psf, const double *x,
                           const double *y, int64_t n, const int *id,
                           short inflag, const double *maxstep, double *xout,
                           double *yout, int *niter, short *flag) {
#ifdef _OPENMP
  int first_status = RETURN_OK;

#pragma omp parallel
  {
    sep_psf local_psf;
    int ws_status = winpos_psf_workspace_clone(psf, &local_psf);

    if (ws_status != RETURN_OK) {
#pragma omp critical(winpos_psf_status)
      {
        if (first_status == RETURN_OK) first_status = ws_status;
      }
    } else {
#pragma omp for schedule(dynamic, 32)
      for (int64_t i = 0; i < n; i++) {
        int s = sep_windowed_psf(im, &local_psf, x[i], y[i], inflag,
                                 id ? id[i] : 0, maxstep ? maxstep[i] : 0.0,
                                 &xout[i], &yout[i], &niter[i], &flag[i]);
        if (s != RETURN_OK) {
#pragma omp critical(winpos_psf_status)
          {
            if (first_status == RETURN_OK) first_status = s;
          }
        }
      }
    }

    winpos_psf_workspace_free(&local_psf);
  }

  return first_status;
#else
  int status = RETURN_OK;
  int64_t i;

  for (i = 0; i < n; i++) {
    status = sep_windowed_psf(im, psf, x[i], y[i], inflag, id ? id[i] : 0,
                              maxstep ? maxstep[i] : 0.0, &xout[i], &yout[i],
                              &niter[i], &flag[i]);
    if (status != RETURN_OK) return status;
  }

  return RETURN_OK;
#endif
}
