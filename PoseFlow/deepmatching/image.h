#ifndef ___IMAGE_H___
#define ___IMAGE_H___

/********** STRUCTURES *********/

/* structure for 1-channel image */
typedef struct image_s
{
  int width;		/* Width of the image */
  int height;		/* Height of the image */
  int stride;		/* Width of the memory (width + paddind such that it is a multiple of 4) */
  float *data;		/* Image data */
} image_t;

/* structure for 3-channels image stored with one layer per color, it assumes that c2 = c1+width*height and c3 = c2+width*height. */
typedef struct color_image_s
{
    int width;			/* Width of the image */
    int height;			/* Height of the image */
    float *c1;			/* Color 1 */
    float *c2;			/* Color 2 */
    float *c3;			/* Color 3 */
} color_image_t;


/********** Create/Delete **********/

/* allocate a new image of size width x height */
image_t *image_new(int width, int height);

/* allocate a new image and copy the content from src */
image_t *image_cpy(const image_t *src);

/* set all pixels values to zeros */
void image_erase(image_t *image);

/* free memory of an image */
void image_delete(image_t *image);

/* multiply an image by a scalar */
void image_mul_scalar(image_t *image, float scalar);

/* allocate a new color image of size width x height */
color_image_t *color_image_new(int width, int height);

/* allocate a new color image and copy the content from src */
color_image_t *color_image_cpy(const color_image_t *src);

/* set all pixels values to zeros */
void color_image_erase(color_image_t *image);

/* free memory of a color image */
void color_image_delete(color_image_t *image);

/* convert a color image to a gray-scale image */
image_t* image_gray_from_color( color_image_t* img ) ;

/* reallocate the memory of an image to fit the new width height */
void resize_if_needed_newsize(image_t *im, int w, int h);


/************ Resizing *********/

/* resize an image with bilinear interpolation */
image_t *image_resize_bilinear_scale(const image_t *src, float scale);

/* resize an image with bilinear interpolation to fit the new weidht, height ; reallocation is done if necessary */
void image_resize_bilinear_newsize(image_t *dst, const image_t *src, int new_width, int new_height);

/* resize a color image  with bilinear interpolation */
color_image_t *color_image_resize_bilinear(const color_image_t *src, float scale);

/* crop an image (in-place) */
void image_crop(image_t* img, int width, int height);

#endif



























