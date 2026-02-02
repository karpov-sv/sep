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

#include "extract.h"
#include "sep.h"
#include "sepcore.h"

#ifndef RAND_MAX
#define RAND_MAX 2147483647
#endif
#define NBRANCH 16 /* starting number per branch */

static _Atomic int nsonmax = 1024; /* max. number sub-objects per level */

/* get and set pixstack */
void sep_set_sub_object_limit(int val) {
  nsonmax = val;
}

int sep_get_sub_object_limit() {
  return nsonmax;
}


int belong(int, objliststruct *, int, objliststruct *);
int64_t *
createsubmap(objliststruct *, int64_t, int64_t *, int64_t *, int64_t *, int64_t *);
int gatherup(objliststruct *, objliststruct *, double);
static int deblend_watershed(
    objliststruct *,
    objliststruct *,
    double,
    double,
    int
);

static _Thread_local const float *wsort_values;
static _Thread_local const float *wsort_peakvals;

static int compare_idx_desc(const void *aptr, const void *bptr) {
  int64_t a = *(const int64_t *)aptr;
  int64_t b = *(const int64_t *)bptr;
  float va = wsort_values[a];
  float vb = wsort_values[b];

  if (va > vb) {
    return -1;
  }
  if (va < vb) {
    return 1;
  }
  if (a < b) {
    return -1;
  }
  if (a > b) {
    return 1;
  }
  return 0;
}

static int compare_peak_desc(const void *aptr, const void *bptr) {
  int a = *(const int *)aptr;
  int b = *(const int *)bptr;
  float va = wsort_peakvals[a];
  float vb = wsort_peakvals[b];

  if (va > vb) {
    return -1;
  }
  if (va < vb) {
    return 1;
  }
  return (a < b) ? -1 : (a > b);
}

/******************************** deblend ************************************/
/*
Divide a list of isophotal detections in several parts (deblending).
NOTE: Even if the object is not deblended, the output objlist threshold is
      recomputed if a variable threshold is used.

This can return two error codes: DEBLEND_OVERFLOW or MEMORY_ALLOC_ERROR
*/
int deblend(
    objliststruct * objlistin,
    objliststruct * objlistout,
    int deblend_nthresh,
    double deblend_mincont,
    double deblend_fwhm,
    int deblend_method,
    int minarea,
    deblendctx * ctx
) {
  objstruct * obj;
  objliststruct debobjlist, debobjlist2;
  double thresh, thresh0, value0;
  int64_t h, i, j, k, l, m, subx, suby, subh, subw, xn, nbm = NBRANCH;
  int64_t * submap;
  int status;

  submap = NULL;
  status = RETURN_OK;
  xn = deblend_nthresh;
  l = 0;

  if (deblend_method == SEP_DEBLEND_WATERSHED) {
    return deblend_watershed(
        objlistin, objlistout, deblend_mincont, deblend_fwhm, minarea
    );
  }

  /* reset global static objlist for deblending */
  objliststruct * const objlist = ctx->objlist;
  memset(objlist, 0, (size_t)xn * sizeof(objliststruct));

  /* initialize local object lists */
  debobjlist.obj = debobjlist2.obj = NULL;
  debobjlist.plist = debobjlist2.plist = NULL;
  debobjlist.nobj = debobjlist2.nobj = 0;
  debobjlist.npix = debobjlist2.npix = 0;

  /* Create the submap for the object.
   * The submap is used in lutz(). We create it here because we may call
   * lutz multiple times below, and we only want to create it once.
   */
  submap = createsubmap(objlistin, l, &subx, &suby, &subw, &subh);
  if (!submap) {
    status = MEMORY_ALLOC_ERROR;
    goto exit;
  }

  for (l = 0; l < objlistin->nobj && status == RETURN_OK; l++) {
    /* set thresholds of object lists based on object threshold */
    thresh0 = objlistin->obj[l].thresh;
    objlistout->thresh = debobjlist2.thresh = thresh0;

    /* add input object to global deblending objlist and one local objlist */
    if ((status = addobjdeep(l, objlistin, &objlist[0])) != RETURN_OK) {
      goto exit;
    }
    if ((status = addobjdeep(l, objlistin, &debobjlist2)) != RETURN_OK) {
      goto exit;
    }

    value0 = objlist[0].obj[0].fdflux * deblend_mincont;
    ctx->ok[0] = (short)1;
    for (k = 1; k < xn; k++) {
      /*------ Calculate threshold */
      thresh = objlistin->obj[l].fdpeak;
      debobjlist.thresh =
          thresh > 0.0 ? thresh0 * pow(thresh / thresh0, (double)k / xn) : thresh0;

      /*--------- Build tree (bottom->up) */
      if (objlist[k - 1].nobj >= nsonmax) {
        status = DEBLEND_OVERFLOW;
        goto exit;
      }

      for (i = 0; i < objlist[k - 1].nobj; i++) {
        status = lutz(
            objlistin->plist,
            submap,
            subx,
            suby,
            subw,
            &objlist[k - 1].obj[i],
            &debobjlist,
            minarea,
            &ctx->lutz
        );
        if (status != RETURN_OK) {
          goto exit;
        }

        for (j = h = 0; j < debobjlist.nobj; j++) {
          if (belong(j, &debobjlist, i, &objlist[k - 1])) {
            debobjlist.obj[j].thresh = debobjlist.thresh;
            if ((status = addobjdeep(j, &debobjlist, &objlist[k])) != RETURN_OK) {
              goto exit;
            }
            m = objlist[k].nobj - 1;
            if (m >= nsonmax) {
              status = DEBLEND_OVERFLOW;
              goto exit;
            }
            if (h >= nbm - 1) {
              if (!(ctx->son = (short *)
                        realloc(ctx->son, xn * nsonmax * (nbm += 16) * sizeof(short))))
              {
                status = MEMORY_ALLOC_ERROR;
                goto exit;
              }
            }
            ctx->son[k - 1 + xn * (i + nsonmax * (h++))] = (short)m;
            ctx->ok[k + xn * m] = (short)1;
          }
        }
        ctx->son[k - 1 + xn * (i + nsonmax * h)] = (short)-1;
      }
    }

    /*------- cut the right branches (top->down) */
    for (k = xn - 2; k >= 0; k--) {
      obj = objlist[k + 1].obj;
      for (i = 0; i < objlist[k].nobj; i++) {
        for (m = h = 0; (j = (int64_t)ctx->son[k + xn * (i + nsonmax * h)]) != -1; h++)
        {
          if (obj[j].fdflux - obj[j].thresh * obj[j].fdnpix > value0) {
            m++;
          }
          ctx->ok[k + xn * i] &= ctx->ok[k + 1 + xn * j];
        }
        if (m > 1) {
          for (h = 0; (j = (int64_t)ctx->son[k + xn * (i + nsonmax * h)]) != -1; h++) {
            if (ctx->ok[k + 1 + xn * j]
                && obj[j].fdflux - obj[j].thresh * obj[j].fdnpix > value0)
            {
              objlist[k + 1].obj[j].flag |= SEP_OBJ_MERGED;
              status = addobjdeep(j, &objlist[k + 1], &debobjlist2);
              if (status != RETURN_OK) {
                goto exit;
              }
            }
          }
          ctx->ok[k + xn * i] = (short)0;
        }
      }
    }

    if (ctx->ok[0]) {
      status = addobjdeep(0, &debobjlist2, objlistout);
    } else {
      status = gatherup(&debobjlist2, objlistout, deblend_fwhm);
    }
  }

exit:
  if (status == DEBLEND_OVERFLOW) {
    put_errdetail(
        "limit of sub-objects reached while deblending. Increase "
        "it with sep.set_sub_object_limit(), decrease number of deblending "
        "thresholds ,or increase the detection threshold."
    );
  }

  free(submap);
  submap = NULL;
  free(debobjlist2.obj);
  free(debobjlist2.plist);

  for (k = 0; k < xn; k++) {
    free(objlist[k].obj);
    free(objlist[k].plist);
  }

  free(debobjlist.obj);
  free(debobjlist.plist);

  return status;
}


/******************************* allocdeblend ******************************/
/*
Allocate the memory allocated by global pointers in refine.c
*/
int allocdeblend(int deblend_nthresh, int64_t w, int64_t h, deblendctx * ctx) {
  int status = RETURN_OK;
  memset(ctx, 0, sizeof(deblendctx));
  QMALLOC(ctx->son, short, deblend_nthresh * nsonmax * NBRANCH, status);
  QMALLOC(ctx->ok, short, deblend_nthresh * nsonmax, status);
  QMALLOC(ctx->objlist, objliststruct, deblend_nthresh, status);
  status = lutzalloc(w, h, &ctx->lutz);
  if (status != RETURN_OK) {
    goto exit;
  }

  return status;
exit:
  freedeblend(ctx);
  return status;
}

/******************************* freedeblend *******************************/
/*
Free the memory allocated by global pointers in refine.c
*/
void freedeblend(deblendctx * ctx) {
  lutzfree(&ctx->lutz);
  free(ctx->son);
  ctx->son = NULL;
  free(ctx->ok);
  ctx->ok = NULL;
  free(ctx->objlist);
  ctx->objlist = NULL;
}

/*************************** deblend_watershed ******************************/
/*
Split objects by detecting local maxima and applying watershed assignment
within the object footprint.
*/
static int deblend_watershed(
    objliststruct * objlistin,
    objliststruct * objlistout,
    double deblend_mincont,
    double deblend_fwhm,
    int minarea
) {
  objstruct * obj;
  pliststruct * pixel, *pixt, *pixt2;
  objliststruct seglist;
  double total_flux, best_d2, dx, dy;
  float *valmap, *peakx, *peaky, *peakval;
  double *flux;
  int64_t *pixpos;
  int64_t *peakidx;
  int64_t subx, suby, subw, subh, nmap, idx, count;
  int64_t i, j, l, write;
  int *label, *keep, *label_to_obj, *keep_labels;
  int64_t *npix;
  int64_t *obj_counts;
  int *obj_labels;
  int status, npeaks, nlabels, nkeep, best_label;
  int deb_minarea;
  double min_peak_sep;
  int *peak_order;

  status = RETURN_OK;
  pixel = objlistin->plist;
  deb_minarea = minarea;
  min_peak_sep = (deblend_fwhm > 0.0 && isfinite(deblend_fwhm))
                     ? 0.5 * deblend_fwhm
                     : 0.0;

  for (l = 0; l < objlistin->nobj && status == RETURN_OK; l++) {
    obj = objlistin->obj + l;
    objlistout->thresh = obj->thresh;

    subx = obj->xmin;
    suby = obj->ymin;
    subw = obj->xmax - subx + 1;
    subh = obj->ymax - suby + 1;
    nmap = subw * subh;

    valmap = NULL;
    label = NULL;
    pixpos = NULL;
    peakidx = NULL;
    peakx = peaky = peakval = NULL;
    flux = NULL;
    npix = NULL;
    keep = NULL;
    label_to_obj = NULL;
    keep_labels = NULL;
    obj_counts = NULL;
    obj_labels = NULL;
    peak_order = NULL;
    seglist.obj = NULL;
    seglist.plist = NULL;

    if (nmap <= 0 || obj->fdnpix <= 0) {
      status = addobjdeep(l, objlistin, objlistout);
      goto obj_cleanup;
    }

    if (!(valmap = (float *)malloc((size_t)nmap * sizeof(float)))) {
      status = MEMORY_ALLOC_ERROR;
      goto obj_cleanup;
    }
    if (!(label = (int *)calloc((size_t)nmap, sizeof(int)))) {
      status = MEMORY_ALLOC_ERROR;
      goto obj_cleanup;
    }
    if (!(pixpos = (int64_t *)malloc((size_t)obj->fdnpix * sizeof(int64_t)))) {
      status = MEMORY_ALLOC_ERROR;
      goto obj_cleanup;
    }

    for (i = 0; i < nmap; i++) {
      valmap[i] = -BIG;
    }

    count = 0;
    for (pixt = pixel + obj->firstpix; pixt >= pixel;
         pixt = pixel + PLIST(pixt, nextpix))
    {
      int64_t x = PLIST(pixt, x);
      int64_t y = PLIST(pixt, y);
      idx = (x - subx) + (y - suby) * subw;
      valmap[idx] = (float)PLISTPIX(pixt, value);
      pixpos[count++] = idx;
    }

    if (count == 0) {
      status = addobjdeep(l, objlistin, objlistout);
      goto obj_cleanup;
    }

    if (!(peakx = (float *)malloc((size_t)(count + 1) * sizeof(float)))) {
      status = MEMORY_ALLOC_ERROR;
      goto obj_cleanup;
    }
    if (!(peaky = (float *)malloc((size_t)(count + 1) * sizeof(float)))) {
      status = MEMORY_ALLOC_ERROR;
      goto obj_cleanup;
    }
    if (!(peakval = (float *)malloc((size_t)(count + 1) * sizeof(float)))) {
      status = MEMORY_ALLOC_ERROR;
      goto obj_cleanup;
    }
    if (!(peakidx = (int64_t *)malloc((size_t)(count + 1) * sizeof(int64_t)))) {
      status = MEMORY_ALLOC_ERROR;
      goto obj_cleanup;
    }

    for (i = 0; i <= count; i++) {
      peakval[i] = -BIG;
    }

    npeaks = 0;
    for (i = 0; i < count; i++) {
      int64_t xx, yy;
      int is_peak = 1;
      idx = pixpos[i];
      if (valmap[idx] <= -BIG / 2) {
        continue;
      }
      xx = idx % subw;
      yy = idx / subw;
      for (int dy_i = -1; dy_i <= 1 && is_peak; dy_i++) {
        for (int dx_i = -1; dx_i <= 1; dx_i++) {
          int64_t nx = xx + dx_i;
          int64_t ny = yy + dy_i;
          int64_t nidx;
          float nval;
          if (dx_i == 0 && dy_i == 0) {
            continue;
          }
          if (nx < 0 || ny < 0 || nx >= subw || ny >= subh) {
            continue;
          }
          nidx = idx + dx_i + dy_i * subw;
          nval = valmap[nidx];
          if (nval <= -BIG / 2) {
            continue;
          }
          if (nval > valmap[idx] || (nval == valmap[idx] && nidx < idx)) {
            is_peak = 0;
            break;
          }
        }
      }
      if (is_peak) {
        npeaks++;
        label[idx] = npeaks;
        peakval[npeaks] = valmap[idx];
        peakx[npeaks] = (float)(subx + xx);
        peaky[npeaks] = (float)(suby + yy);
        peakidx[npeaks] = idx;
      }
    }

    if (min_peak_sep > 0.0 && npeaks > 1) {
      if (!(peak_order = (int *)malloc((size_t)npeaks * sizeof(int)))) {
        status = MEMORY_ALLOC_ERROR;
        goto obj_cleanup;
      }
      for (i = 0; i < npeaks; i++) {
        peak_order[i] = (int)(i + 1);
      }
      wsort_peakvals = peakval;
      qsort(peak_order, (size_t)npeaks, sizeof(int), compare_peak_desc);

      int kept = 0;
      for (i = 0; i < npeaks; i++) {
        int lab = peak_order[i];
        int accept = 1;
        for (j = 0; j < kept; j++) {
          int prev = peak_order[j];
          double dx = peakx[lab] - peakx[prev];
          double dy = peaky[lab] - peaky[prev];
          if (dx * dx + dy * dy < min_peak_sep * min_peak_sep) {
            accept = 0;
            break;
          }
        }
        if (accept) {
          peak_order[kept++] = lab;
        }
      }

      for (i = 1; i <= npeaks; i++) {
        label[peakidx[i]] = 0;
      }
      for (i = 0; i < kept; i++) {
        int lab = peak_order[i];
        int newlab = (int)(i + 1);
        label[peakidx[lab]] = newlab;
        peakx[newlab] = peakx[lab];
        peaky[newlab] = peaky[lab];
        peakval[newlab] = peakval[lab];
        peakidx[newlab] = peakidx[lab];
      }
      npeaks = kept;
    }

    if (npeaks <= 1) {
      status = addobjdeep(l, objlistin, objlistout);
      goto obj_cleanup;
    }

    wsort_values = valmap;
    qsort(pixpos, (size_t)count, sizeof(int64_t), compare_idx_desc);

    nlabels = npeaks;
    for (i = 0; i < count; i++) {
      int64_t xx, yy;
      idx = pixpos[i];
      if (label[idx]) {
        continue;
      }
      xx = idx % subw;
      yy = idx / subw;
      best_label = 0;
      float best_val = -BIG;
      for (int dy_i = -1; dy_i <= 1; dy_i++) {
        for (int dx_i = -1; dx_i <= 1; dx_i++) {
          int64_t nx = xx + dx_i;
          int64_t ny = yy + dy_i;
          int64_t nidx;
          int nlab;
          float nval;
          if (dx_i == 0 && dy_i == 0) {
            continue;
          }
          if (nx < 0 || ny < 0 || nx >= subw || ny >= subh) {
            continue;
          }
          nidx = idx + dx_i + dy_i * subw;
          nlab = label[nidx];
          if (nlab <= 0) {
            continue;
          }
          nval = valmap[nidx];
          if (nval > best_val) {
            best_val = nval;
            best_label = nlab;
          }
        }
      }
      if (best_label == 0) {
        nlabels++;
        label[idx] = nlabels;
        peakval[nlabels] = valmap[idx];
        peakx[nlabels] = (float)(subx + xx);
        peaky[nlabels] = (float)(suby + yy);
      } else {
        label[idx] = best_label;
      }
    }

    flux = (double *)calloc((size_t)(nlabels + 1), sizeof(double));
    npix = (int64_t *)calloc((size_t)(nlabels + 1), sizeof(int64_t));
    keep = (int *)calloc((size_t)(nlabels + 1), sizeof(int));
    if (!flux || !npix || !keep) {
      status = MEMORY_ALLOC_ERROR;
      goto obj_cleanup;
    }

    for (i = 0; i < count; i++) {
      idx = pixpos[i];
      j = label[idx];
      if (j <= 0) {
        continue;
      }
      flux[j] += valmap[idx];
      npix[j] += 1;
      if (valmap[idx] > peakval[j]) {
        peakval[j] = valmap[idx];
        peakx[j] = (float)(subx + (idx % subw));
        peaky[j] = (float)(suby + (idx / subw));
      }
    }

    total_flux = 0.0;
    for (i = 1; i <= nlabels; i++) {
      total_flux += flux[i];
    }

    nkeep = 0;
    for (i = 1; i <= nlabels; i++) {
      if (npix[i] >= deb_minarea && flux[i] >= total_flux * deblend_mincont) {
        keep[i] = 1;
        nkeep++;
      }
    }

    if (nkeep <= 1) {
      status = addobjdeep(l, objlistin, objlistout);
      goto obj_cleanup;
    }

    if (!(keep_labels = (int *)malloc((size_t)nkeep * sizeof(int)))) {
      status = MEMORY_ALLOC_ERROR;
      goto obj_cleanup;
    }
    j = 0;
    for (i = 1; i <= nlabels; i++) {
      if (keep[i]) {
        keep_labels[j++] = (int)i;
      }
    }

    for (i = 0; i < count; i++) {
      idx = pixpos[i];
      j = label[idx];
      if (j <= 0 || keep[j]) {
        continue;
      }
      best_d2 = 1.0e30;
      best_label = keep_labels[0];
      for (int k = 0; k < nkeep; k++) {
        int lab = keep_labels[k];
        dx = (subx + (idx % subw)) - peakx[lab];
        dy = (suby + (idx / subw)) - peaky[lab];
        if ((dx * dx + dy * dy) < best_d2) {
          best_d2 = dx * dx + dy * dy;
          best_label = lab;
        }
      }
      label[idx] = best_label;
    }

    label_to_obj = (int *)malloc((size_t)(nlabels + 1) * sizeof(int));
    obj_labels = (int *)malloc((size_t)nkeep * sizeof(int));
    if (!label_to_obj || !obj_labels) {
      status = MEMORY_ALLOC_ERROR;
      goto obj_cleanup;
    }
    for (i = 0; i <= nlabels; i++) {
      label_to_obj[i] = -1;
    }
    for (i = 1, j = 0; i <= nlabels; i++) {
      if (keep[i]) {
        label_to_obj[i] = (int)j++;
        obj_labels[label_to_obj[i]] = (int)i;
      }
    }

    seglist.nobj = nkeep;
    seglist.npix = 0;
    seglist.thresh = obj->thresh;
    if (!(seglist.obj = (objstruct *)calloc((size_t)nkeep, sizeof(objstruct)))) {
      status = MEMORY_ALLOC_ERROR;
      goto obj_cleanup;
    }
    if (!(seglist.plist = (pliststruct *)malloc((size_t)obj->fdnpix * plistsize))) {
      status = MEMORY_ALLOC_ERROR;
      goto obj_cleanup;
    }
    if (!(obj_counts = (int64_t *)calloc((size_t)nkeep, sizeof(int64_t)))) {
      status = MEMORY_ALLOC_ERROR;
      goto obj_cleanup;
    }

    for (i = 0; i < nkeep; i++) {
      seglist.obj[i].firstpix = -1;
      seglist.obj[i].lastpix = -1;
      seglist.obj[i].flag = obj->flag | SEP_OBJ_MERGED;
      seglist.obj[i].thresh = obj->thresh;
      seglist.obj[i].mx = peakx[obj_labels[i]];
      seglist.obj[i].my = peaky[obj_labels[i]];
    }

    write = 0;
    for (pixt = pixel + obj->firstpix; pixt >= pixel;
         pixt = pixel + PLIST(pixt, nextpix))
    {
      int64_t x = PLIST(pixt, x);
      int64_t y = PLIST(pixt, y);
      idx = (x - subx) + (y - suby) * subw;
      j = label[idx];
      if (j <= 0 || !keep[j]) {
        continue;
      }
      i = label_to_obj[j];
      pixt2 = seglist.plist + (write * plistsize);
      memcpy(pixt2, pixt, (size_t)plistsize);
      PLIST(pixt2, nextpix) = -1;
      if (seglist.obj[i].firstpix < 0) {
        seglist.obj[i].firstpix = write * plistsize;
      } else {
        PLIST(seglist.plist + seglist.obj[i].lastpix, nextpix) =
            write * plistsize;
      }
      seglist.obj[i].lastpix = write * plistsize;
      obj_counts[i] += 1;
      write++;
    }

    seglist.npix = write;
    if (seglist.npix > 0) {
      seglist.plist = realloc(seglist.plist, (size_t)seglist.npix * plistsize);
    }
    for (i = 0; i < seglist.nobj; i++) {
      seglist.obj[i].fdnpix = obj_counts[i];
    }

    for (i = 0; i < seglist.nobj; i++) {
      status = addobjdeep(i, &seglist, objlistout);
      if (status != RETURN_OK) {
        goto obj_cleanup;
      }
    }

  obj_cleanup:
    free(valmap);
    free(label);
    free(pixpos);
    free(peakidx);
    free(peakx);
    free(peaky);
    free(peakval);
    free(flux);
    free(npix);
    free(keep);
    free(label_to_obj);
    free(keep_labels);
    free(obj_counts);
    free(obj_labels);
    free(peak_order);
    free(seglist.obj);
    free(seglist.plist);
  }

  return status;
}

/********************************* gatherup **********************************/
/*
Collect faint remaining pixels and allocate them to their most probable
progenitor.
*/
int gatherup(
    objliststruct * objlistin,
    objliststruct * objlistout,
    double deblend_fwhm
) {
  char * bmp;
  float *amp, *p;
  double dx, dy, drand, dist, distmin, rsq, w, wmax, sigma2, inv_two_sigma2;
  objstruct *objin = objlistin->obj, *objout, *objt;

  pliststruct *pixelin = objlistin->plist, *pixelout, *pixt, *pixt2;

  int64_t i, k, l, *n, iclst, npix, bmwidth, nobj = objlistin->nobj, xs, ys, x, y;
  int status, use_fixed, imax;

  bmp = NULL;
  amp = p = NULL;
  n = NULL;
  status = RETURN_OK;

  objlistout->thresh = objlistin->thresh;

  /* Optional fixed-PSF deblending: use circular Gaussian with given FWHM. */
  use_fixed = 0;
  sigma2 = 0.0;
  inv_two_sigma2 = 0.0;
  if (deblend_fwhm > 0.0 && isfinite(deblend_fwhm)) {
    double sigma = deblend_fwhm / 2.354820045;
    if (sigma > 0.0) {
      sigma2 = sigma * sigma;
      inv_two_sigma2 = 0.5 / sigma2;
      use_fixed = 1;
    }
  }

  QMALLOC(amp, float, nobj, status);
  QMALLOC(p, float, nobj, status);
  QMALLOC(n, int64_t, nobj, status);

  for (i = 1; i < nobj; i++) {
    analyse(i, objlistin, 0, 0.0);
  }

  p[0] = 0.0;
  bmwidth = objin->xmax - (xs = objin->xmin) + 1;
  npix = bmwidth * (objin->ymax - (ys = objin->ymin) + 1);
  if (!(bmp = (char *)calloc(1, npix * sizeof(char)))) {
    bmp = NULL;
    status = MEMORY_ALLOC_ERROR;
    goto exit;
  }

  for (objt = objin + (i = 1); i < nobj; i++, objt++) {
    /*-- Now we have passed the deblending section, reset threshold */
    objt->thresh = objlistin->thresh;

    /* ------------	flag pixels which are already allocated */
    for (pixt = pixelin + objin[i].firstpix; pixt >= pixelin;
         pixt = pixelin + PLIST(pixt, nextpix))
    {
      bmp[(PLIST(pixt, x) - xs) + (PLIST(pixt, y) - ys) * bmwidth] = '\1';
    }

    status = addobjdeep(i, objlistin, objlistout);
    if (status != RETURN_OK) {
      goto exit;
    }
    n[i] = objlistout->nobj - 1;

    if (use_fixed) {
      double amp_est = objt->fdflux / (2.0 * PI * sigma2);
      if (amp_est < 0.0) {
        amp_est = 0.0;
      }
      if (objt->fdpeak > 0.0 && amp_est > 4.0 * objt->fdpeak) {
        amp_est = 4.0 * objt->fdpeak;
      }
      amp[i] = (float)amp_est;
    } else {
      dist = objt->fdnpix / (2 * PI * objt->abcor * objt->a * objt->b);
      amp[i] = dist < 70.0 ? objt->thresh * expf(dist) : 4.0 * objt->fdpeak;

      /* ------------ limitate expansion ! */
      if (amp[i] > 4.0 * objt->fdpeak) {
        amp[i] = 4.0 * objt->fdpeak;
      }
    }
  }

  objout = objlistout->obj; /* DO NOT MOVE !!! */

  if (!(pixelout = realloc(objlistout->plist, (objlistout->npix + npix) * plistsize))) {
    status = MEMORY_ALLOC_ERROR;
    goto exit;
  }

  objlistout->plist = pixelout;
  k = objlistout->npix;
  iclst = 0; /* To avoid gcc -Wall warnings */
  for (pixt = pixelin + objin->firstpix; pixt >= pixelin;
       pixt = pixelin + PLIST(pixt, nextpix))
  {
    x = PLIST(pixt, x);
    y = PLIST(pixt, y);
    if (!bmp[(x - xs) + (y - ys) * bmwidth]) {
      pixt2 = pixelout + (l = (k++ * plistsize));
      memcpy(pixt2, pixt, (size_t)plistsize);
      PLIST(pixt2, nextpix) = -1;
      distmin = 1e+31;
      wmax = -1.0;
      imax = 1;
      for (objt = objin + (i = 1); i < nobj; i++, objt++) {
        dx = x - objt->mx;
        dy = y - objt->my;
        if (use_fixed) {
          rsq = dx * dx + dy * dy;
          w = (amp[i] > 0.0f) ? (double)amp[i] * exp(-rsq * inv_two_sigma2) : 0.0;
          if (w > wmax) {
            wmax = w;
            imax = i;
          }
          if (rsq < distmin) {
            distmin = rsq;
            iclst = i;
          }
        } else {
          dist = 0.5
                 * (objt->cxx * dx * dx + objt->cyy * dy * dy + objt->cxy * dx * dy)
                 / objt->abcor;
          p[i] = p[i - 1] + (dist < 70.0 ? amp[i] * expf(-dist) : 0.0);
          if (dist < distmin) {
            distmin = dist;
            iclst = i;
          }
        }
      }
      if (use_fixed) {
        i = (wmax > 0.0) ? imax : iclst;
      } else if (p[nobj - 1] > 1.0e-31) {
        drand = p[nobj - 1] * rand_r(&randseed) / (double)RAND_MAX;
        for (i = 1; i < nobj && p[i] < drand; i++)
          ;
        if (i == nobj) {
          i = iclst;
        }
      } else {
        i = iclst;
      }
      objout[n[i]].lastpix = PLIST(pixelout + objout[n[i]].lastpix, nextpix) = l;
    }
  }

  objlistout->npix = k;
  if (!(objlistout->plist = realloc(pixelout, objlistout->npix * plistsize))) {
    status = MEMORY_ALLOC_ERROR;
  }

exit:
  free(bmp);
  free(amp);
  free(p);
  free(n);

  return status;
}

/**************** belong (originally in manobjlist.c) ************************/
/*
 * say if an object is "included" in another. Returns 1 if the pixels of the
 * first object are included in the pixels of the second object.
 */

int belong(
    int corenb, objliststruct * coreobjlist, int shellnb, objliststruct * shellobjlist
) {
  objstruct *cobj = &(coreobjlist->obj[corenb]), *sobj = &(shellobjlist->obj[shellnb]);
  pliststruct *cpl = coreobjlist->plist, *spl = shellobjlist->plist, *pixt;

  int64_t xc = PLIST(cpl + cobj->firstpix, x), yc = PLIST(cpl + cobj->firstpix, y);

  for (pixt = spl + sobj->firstpix; pixt >= spl; pixt = spl + PLIST(pixt, nextpix)) {
    if ((PLIST(pixt, x) == xc) && (PLIST(pixt, y) == yc)) {
      return 1;
    }
  }

  return 0;
}


/******************************** createsubmap *******************************/
/*
Create pixel-index submap for deblending.
*/
int64_t * createsubmap(
    objliststruct * objlistin,
    int64_t no,
    int64_t * subx,
    int64_t * suby,
    int64_t * subw,
    int64_t * subh
) {
  objstruct * obj;
  pliststruct *pixel, *pixt;
  int64_t i, n, xmin, ymin, w, *pix, *pt, *submap;

  obj = objlistin->obj + no;
  pixel = objlistin->plist;

  *subx = xmin = obj->xmin;
  *suby = ymin = obj->ymin;
  *subw = w = obj->xmax - xmin + 1;
  *subh = obj->ymax - ymin + 1;

  n = w * *subh;
  if (!(submap = pix = malloc(n * sizeof(int64_t)))) {
    return NULL;
  }
  pt = pix;
  for (i = n; i--;) {
    *(pt++) = -1;
  }

  for (i = obj->firstpix; i != -1; i = PLIST(pixt, nextpix)) {
    pixt = pixel + i;
    *(pix + (PLIST(pixt, x) - xmin) + (PLIST(pixt, y) - ymin) * w) = i;
  }

  return submap;
}
