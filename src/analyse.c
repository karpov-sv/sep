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

#define MTHRESH_STACKSIZE 64

static int fit_fwhm_core(
    const objstruct * obj,
    pliststruct * pixel,
    double mx_fwhm,
    double my_fwhm,
    PIXTYPE thresh0,
    float * fwhm_out
) {
  pliststruct * pixt;
  double s, sx, sxx, sy, sxy;
  double dx, dy, d2, lpix, pix, b, d, bmax, w, raw_fwhm, dscale;

  s = sx = sxx = sy = sxy = 0.0;

  for (pixt = pixel + obj->firstpix; pixt >= pixel; pixt = pixel + PLIST(pixt, nextpix)) {
    pix = (double)PLISTPIX(pixt, value);
    if (pix > thresh0) {
      dx = (double)PLIST(pixt, x) - mx_fwhm;
      dy = (double)PLIST(pixt, y) - my_fwhm;
      lpix = log(pix);
      d2 = dx * dx + dy * dy;
      w = pix * pix;
      s += w;
      sx += d2 * w;
      sxx += d2 * d2 * w;
      sy += lpix * w;
      sxy += lpix * d2 * w;
    }
  }

  d = s * sxx - sx * sx;
  dscale = fabs(s * sxx) + fabs(sx * sx);
  if (!(dscale > 0.0) || fabs(d) <= 1e-12 * dscale) {
    return 0;
  }

  b = -(s * sxy - sx * sy) / d;
  bmax = 1.0 / (13.0 * obj->a * obj->b);
  if (!isfinite(b)) {
    return 0;
  }
  if (b < bmax) {
    b = bmax;
  }

  raw_fwhm = 1.6651 / sqrt(b);
  if (!isfinite(raw_fwhm) || raw_fwhm <= 0.5) {
    return 0;
  }

  *fwhm_out = (float)(raw_fwhm - 1.0 / (4.0 * raw_fwhm));
  return 1;
}

/********************************** cleanprep ********************************/
/*
 * Prepare object for cleaning, by calculating mthresh.
 * This used to be in analyse() / examineiso().
 */

int analysemthresh(int objnb, objliststruct * objlist, int minarea, PIXTYPE thresh) {
  objstruct * obj = objlist->obj + objnb;
  pliststruct * pixel = objlist->plist;
  pliststruct * pixt;
  PIXTYPE tpix;
  float stackheap[MTHRESH_STACKSIZE];
  float *heap, *heap_alloc, *heapt, *heapj, *heapk, swap;
  int j, k, h, status;

  status = RETURN_OK;
  heap = heap_alloc = heapt = heapj = heapk = NULL;
  h = minarea;

  if (obj->fdnpix < minarea) {
    obj->mthresh = 0.0;
    return status;
  }

  if (minarea <= MTHRESH_STACKSIZE) {
    heap = stackheap;
  } else {
    QCALLOC(heap_alloc, float, minarea, status);
    heap = heap_alloc;
  }
  heapt = heap;

  /*-- Find the minareath pixel in decreasing intensity for CLEANing */
  for (pixt = pixel + obj->firstpix; pixt >= pixel; pixt = pixel + PLIST(pixt, nextpix))
  {
    /* amount pixel is above threshold */
    tpix = PLISTPIX(pixt, cdvalue)
           - (PLISTEXIST(thresh) ? PLISTPIX(pixt, thresh) : thresh);
    if (h > 0) {
      *(heapt++) = (float)tpix;
    } else if (h) {
      if ((float)tpix > *heap) {
        *heap = (float)tpix;
        for (j = 0; (k = (j + 1) << 1) <= minarea; j = k) {
          heapk = heap + k;
          heapj = heap + j;
          if (k != minarea && *(heapk - 1) > *heapk) {
            heapk++;
            k++;
          }
          if (*heapj <= *(--heapk)) {
            break;
          }
          swap = *heapk;
          *heapk = *heapj;
          *heapj = swap;
        }
      }
    } else {
      fqmedian(heap, minarea);
    }
    h--;
  }

  obj->mthresh = *heap;

exit:
  free(heap_alloc);
  return status;
}

/************************* preanalyse **************************************/

void preanalyse(int no, objliststruct * objlist) {
  objstruct * obj = &objlist->obj[no];
  pliststruct *pixel = objlist->plist, *pixt;
  PIXTYPE peak, cpeak, val, cval;
  double rv;
  int64_t x, y, xmin, xmax, ymin, ymax, fdnpix;
  int64_t xpeak, ypeak, xcpeak, ycpeak;

  /*-----  initialize stacks and bounds */
  fdnpix = 0;
  rv = 0.0;
  peak = cpeak = -BIG;
  ymin = xmin = 2 * MAXPICSIZE; /* to be really sure!! */
  ymax = xmax = 0;
  xpeak = ypeak = xcpeak = ycpeak = 0; /* avoid -Wall warnings */

  /*-----  integrate results */
  for (pixt = pixel + obj->firstpix; pixt >= pixel; pixt = pixel + PLIST(pixt, nextpix))
  {
    x = PLIST(pixt, x);
    y = PLIST(pixt, y);
    val = PLISTPIX(pixt, value);
    cval = PLISTPIX(pixt, cdvalue);
    if (peak < val) {
      peak = val;
      xpeak = x;
      ypeak = y;
    }
    if (cpeak < cval) {
      cpeak = cval;
      xcpeak = x;
      ycpeak = y;
    }
    rv += cval;
    if (xmin > x) {
      xmin = x;
    }
    if (xmax < x) {
      xmax = x;
    }
    if (ymin > y) {
      ymin = y;
    }
    if (ymax < y) {
      ymax = y;
    }
    fdnpix++;
  }

  obj->fdnpix = fdnpix;
  obj->fdflux = (float)rv;
  obj->fdpeak = cpeak;
  obj->dpeak = peak;
  obj->xpeak = xpeak;
  obj->ypeak = ypeak;
  obj->xcpeak = xcpeak;
  obj->ycpeak = ycpeak;
  obj->xmin = xmin;
  obj->xmax = xmax;
  obj->ymin = ymin;
  obj->ymax = ymax;
}

/******************************** analyse *********************************/
/*
  If robust = 1, you must have run previously with robust=0
*/

void analyse(int no, objliststruct * objlist, int robust, double gain) {
  objstruct * obj = &objlist->obj[no];
  pliststruct *pixel = objlist->plist, *pixt;
  PIXTYPE peak, val, cval;
  double thresh, thresh2, t1t2, darea, mx, my, mx2, my2, mxy, rv, rv2, tv, xm, ym, xm2,
      ym2, xym, temp, temp2, theta, pmx2, pmy2, errx2, erry2, errxy, cvar, cvarsum;
  int64_t x, y, xmin, ymin, area2, dnpix;

  preanalyse(no, objlist);

  dnpix = 0;
  mx = my = tv = 0.0;
  mx2 = my2 = mxy = 0.0;
  cvarsum = errx2 = erry2 = errxy = 0.0;
  thresh = obj->thresh;
  peak = obj->dpeak;
  rv = obj->fdflux;
  rv2 = rv * rv;
  thresh2 = (thresh + peak) / 2.0;
  area2 = 0;

  xmin = obj->xmin;
  ymin = obj->ymin;

  for (pixt = pixel + obj->firstpix; pixt >= pixel; pixt = pixel + PLIST(pixt, nextpix))
  {
    x = PLIST(pixt, x) - xmin; /* avoid roundoff errors on big images */
    y = PLIST(pixt, y) - ymin; /* avoid roundoff errors on big images */
    cval = PLISTPIX(pixt, cdvalue);
    tv += (val = PLISTPIX(pixt, value));
    if (val > thresh) {
      dnpix++;
    }
    if (val > thresh2) {
      area2++;
    }
    mx += cval * x;
    my += cval * y;
    mx2 += cval * x * x;
    my2 += cval * y * y;
    mxy += cval * x * y;
  }

  /* compute object's properties */
  xm = mx / rv; /* mean x */
  ym = my / rv; /* mean y */


  /* In case of blending, use previous barycenters */
  if ((robust) && (obj->flag & SEP_OBJ_MERGED)) {
    double xn, yn;

    xn = obj->mx - xmin;
    yn = obj->my - ymin;
    xm2 = mx2 / rv + xn * xn - 2 * xm * xn;
    ym2 = my2 / rv + yn * yn - 2 * ym * yn;
    xym = mxy / rv + xn * yn - xm * yn - xn * ym;
    xm = xn;
    ym = yn;
  } else {
    xm2 = mx2 / rv - xm * xm; /* variance of x */
    ym2 = my2 / rv - ym * ym; /* variance of y */
    xym = mxy / rv - xm * ym; /* covariance */
  }

  /* Calculate the errors on the variances */
  for (pixt = pixel + obj->firstpix; pixt >= pixel; pixt = pixel + PLIST(pixt, nextpix))
  {
    x = PLIST(pixt, x) - xmin; /* avoid roundoff errors on big images */
    y = PLIST(pixt, y) - ymin; /* avoid roundoff errors on big images */

    cvar = PLISTEXIST(var) ? PLISTPIX(pixt, var) : 0.0;
    if (gain > 0.0) { /* add poisson noise if given */
      cval = PLISTPIX(pixt, cdvalue);
      if (cval > 0.0) {
        cvar += cval / gain;
      }
    }

    /* Note that this works for both blended and non-blended cases
     * because xm is set to xn above for the blended case. */
    cvarsum += cvar;
    errx2 += cvar * (x - xm) * (x - xm);
    erry2 += cvar * (y - ym) * (y - ym);
    errxy += cvar * (x - xm) * (y - ym);
  }
  errx2 /= rv2;
  erry2 /= rv2;
  errxy /= rv2;

  /* Handle fully correlated x/y (which cause a singularity...) */
  if ((temp2 = xm2 * ym2 - xym * xym) < 0.00694) {
    xm2 += 0.0833333;
    ym2 += 0.0833333;
    temp2 = xm2 * ym2 - xym * xym;
    obj->flag |= SEP_OBJ_SINGU;

    /* handle it for the error parameters */
    cvarsum *= 0.08333 / rv2;
    if (errx2 * erry2 - errxy * errxy < cvarsum * cvarsum) {
      errx2 += cvarsum;
      erry2 += cvarsum;
    }
  }

  if ((fabs(temp = xm2 - ym2)) > 0.0) {
    theta = atan2(2.0 * xym, temp) / 2.0;
  } else {
    theta = PI / 4.0;
  }

  temp = sqrt(0.25 * temp * temp + xym * xym);
  pmy2 = pmx2 = 0.5 * (xm2 + ym2);
  pmx2 += temp;
  pmy2 -= temp;

  obj->dnpix = (LONG)dnpix;
  obj->dflux = tv;
  obj->mx = xm + xmin; /* add back xmin */
  obj->my = ym + ymin; /* add back ymin */
  obj->mx2 = xm2;
  obj->errx2 = errx2;
  obj->my2 = ym2;
  obj->erry2 = erry2;
  obj->mxy = xym;
  obj->errxy = errxy;
  obj->a = (float)sqrt(pmx2);
  obj->b = (float)sqrt(pmy2);
  obj->theta = theta;

  obj->cxx = (float)(ym2 / temp2);
  obj->cyy = (float)(xm2 / temp2);
  obj->cxy = (float)(-2 * xym / temp2);

  darea = (double)area2 - dnpix;
  t1t2 = thresh / thresh2;

  /* debugging */
  /*if (t1t2>0.0 && !PLISTEXIST(thresh)) */ /* was: prefs.dweight_flag */
  if (t1t2 > 0.0) {
    obj->abcor = (darea < 0.0 ? darea : -1.0)
                 / (2 * PI * log(t1t2 < 1.0 ? t1t2 : 0.99) * obj->a * obj->b);
    if (obj->abcor > 1.0) {
      obj->abcor = 1.0;
    }
  } else {
    obj->abcor = 1.0;
  }

  /* Compute Gaussian-core FWHM (ported from SExtractor).
   * Assumes background-subtracted data (SEP does not store per-object bkg).
   *
   * For compact sources the highest-core threshold can include too few
   * pixels to constrain the log-profile slope. Retry at lower internal
   * thresholds before giving up, but keep unresolved pathologies at 0
   * instead of switching to a biased moment-based width. */
  {
    PIXTYPE thresh0;
    PIXTYPE tries[4];
    double mx_fwhm = obj->mx;
    double my_fwhm = obj->my;
    int i, ntry;

    thresh0 = obj->dpeak / 5.0;
    if (thresh0 < obj->thresh) {
      thresh0 = obj->thresh;
    }

    if (thresh0 > 0.0 && obj->a > 0.0f && obj->b > 0.0f) {
      ntry = 0;
      tries[ntry++] = thresh0;

      if (obj->dpeak / 10.0 > obj->thresh && obj->dpeak / 10.0 < tries[ntry - 1]) {
        tries[ntry++] = obj->dpeak / 10.0;
      }
      if (obj->dpeak / 20.0 > obj->thresh && obj->dpeak / 20.0 < tries[ntry - 1]) {
        tries[ntry++] = obj->dpeak / 20.0;
      }
      if (obj->thresh > 0.0 && obj->thresh < tries[ntry - 1]) {
        tries[ntry++] = obj->thresh;
      }

      obj->fwhm = 0.0f;
      for (i = 0; i < ntry; i++) {
        if (fit_fwhm_core(obj, pixel, mx_fwhm, my_fwhm, tries[i], &obj->fwhm)) {
          break;
        }
      }
    } else {
      obj->fwhm = 0.0f;
    }
  }
}
