#include <assert.h>
#include <stdio.h>

#include "sep.h"

static void check_catalogs(
    const sep_catalog * original, const sep_catalog * extended, int include_pixels
) {
  int i;
  int64_t j;

  assert(original->nobj == 3);
  assert(extended->nobj == original->nobj);
  assert((original->pix != NULL) == include_pixels);
  assert((extended->pix != NULL) == include_pixels);
  assert((original->objectspix != NULL) == include_pixels);
  assert((extended->objectspix != NULL) == include_pixels);
  for (i = 0; i < original->nobj; i++) {
    assert(original->npix[i] == extended->npix[i]);
    assert(original->flux[i] == extended->flux[i]);
    assert(original->x[i] == extended->x[i]);
    assert(original->y[i] == extended->y[i]);
    assert(original->flag[i] == extended->flag[i]);
    if (include_pixels) {
      for (j = 0; j < original->npix[i]; j++) {
        assert(original->pix[i][j] == extended->pix[i][j]);
      }
    }
  }
}

int main(void) {
  float data[30] = {0};
  float profile[8] = {1, 10, 8.5, 9, 1, 1, 5, 1};
  sep_image image = {0};
  sep_catalog *original, *extended;
  int i, include_pixels, status;

  for (i = 0; i < 8; i++) {
    data[11 + i] = profile[i];
  }
  image.data = data;
  image.dtype = SEP_TFLOAT;
  image.w = 10;
  image.h = 3;
  image.noise_type = SEP_NOISE_NONE;

  /* The original signatures must still compile and disable prominence. */
  status = sep_extract(
      &image, 0.5, SEP_THRESH_ABS, 1, NULL, 0, 0, SEP_FILTER_CONV,
      32, 0.0, 0.0, SEP_DEBLEND_WATERSHED, 0, 1.0, &original
  );
  assert(status == 0);
  status = sep_extract_ex(
      &image, 0.5, SEP_THRESH_ABS, 1, NULL, 0, 0, SEP_FILTER_CONV,
      32, 0.0, 0.0, SEP_DEBLEND_WATERSHED, 0.0, 0, 1.0, &extended
  );
  assert(status == 0);
  check_catalogs(original, extended, 1);
  sep_catalog_free(original);
  sep_catalog_free(extended);

  status = sep_extract_ex(
      &image, 0.5, SEP_THRESH_ABS, 1, NULL, 0, 0, SEP_FILTER_CONV,
      32, 0.0, 0.0, SEP_DEBLEND_WATERSHED, 2.0, 0, 1.0, &extended
  );
  assert(status == 0);
  assert(extended->nobj == 2);
  sep_catalog_free(extended);

  for (include_pixels = 0; include_pixels <= 1; include_pixels++) {
    status = sep_extract_with_pixels(
        &image, 0.5, SEP_THRESH_ABS, 1, NULL, 0, 0, SEP_FILTER_CONV,
        32, 0.0, 0.0, SEP_DEBLEND_WATERSHED, 0, 1.0, include_pixels, &original
    );
    assert(status == 0);
    status = sep_extract_with_pixels_ex(
        &image, 0.5, SEP_THRESH_ABS, 1, NULL, 0, 0, SEP_FILTER_CONV,
        32, 0.0, 0.0, SEP_DEBLEND_WATERSHED, 0.0, 0, 1.0,
        include_pixels, &extended
    );
    assert(status == 0);
    check_catalogs(original, extended, include_pixels);
    sep_catalog_free(original);
    sep_catalog_free(extended);

    status = sep_extract_with_pixels_ex(
        &image, 0.5, SEP_THRESH_ABS, 1, NULL, 0, 0, SEP_FILTER_CONV,
        32, 0.0, 0.0, SEP_DEBLEND_WATERSHED, 2.0, 0, 1.0,
        include_pixels, &extended
    );
    assert(status == 0);
    assert(extended->nobj == 2);
    assert((extended->pix != NULL) == include_pixels);
    sep_catalog_free(extended);
  }

  puts("test_extract passed");
  return 0;
}
