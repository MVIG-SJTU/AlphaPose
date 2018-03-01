/*
Copyright (C) 2014 Jerome Revaud

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>
*/
#include "pixel_desc.h"
#include "std.h"
#include "image.h"
#include "hog.h"
#include "conv.h"


/* convert a float image to a consecutive array
   a bit stupid but well
*/
UBYTE_image* image_to_arraytype( image_t* img ) {
  UBYTE_image* res = NEW(UBYTE_image);
  *res = empty_image(UBYTE,img->width,img->height);
  
  for(int j=0; j<img->height; j++)
    for(int i=0; i<img->width; i++)
      res->pixels[i+j*res->tx] = (UBYTE)img->data[i+j*img->stride];
  
  return res;
}


// set default params
void set_default_desc_params( desc_params_t* params )
{
  // default = jpg settings,
  // better in almost all cases
  params->presmooth_sigma = 1.0;
  params->mid_smoothing = 1.0;
  params->post_smoothing = 1.0;
  params->hog_sigmoid = 0.2;
  params->ninth_dim = 0.3;
  params->norm_pixels = false;
}


/* extract pixel descriptors (pixel-wise HOG)
*/
float_layers* extract_desc( image_t* _img, const desc_params_t* params, int nt )
{
  // verify parameters
  assert(between(0,params->presmooth_sigma,3));
  assert(between(0,params->mid_smoothing,3));
  assert(between(0,params->post_smoothing,3));
  assert(between(0.05,params->hog_sigmoid,0.8));
  assert(between(0,params->ninth_dim,1));
  assert(between(0,params->norm_pixels,1));
  
  UBYTE_image* img = image_to_arraytype(_img);  // could be optimized but well
  const int npix = img->tx*img->ty;
  //hash_image(img)D(img->tx)D(img->ty)
  
  // pre-smooth image
  assert( params->presmooth_sigma>=0 );
  if( params->presmooth_sigma>0 )
    _smooth_gaussian( img, params->presmooth_sigma, img, nt );
  //hash_image(img)
  
  // extract HOG
  float_layers grad = {NEWA(float,npix*2),img->tx,img->ty,2};
  _compute_grad_101( img, 0, &grad, nt );
  //hash_cube(&grad)
  float_layers* hog = NEW(float_layers);
  *hog = {NEWA(float,9*npix),img->tx,img->ty,8};
  _compute_hog( &grad, 1, hog, nt );
  free(grad.pixels);
  free_image(img);
  //hash_layers(hog)
  
  // mid smoothing
  assert( params->mid_smoothing>=0 );
  if( params->mid_smoothing )
    smooth_hog_gaussian( hog, params->mid_smoothing, nt );
  //hash_layers(hog)
  
  // apply non-linearity
  assert( params->hog_sigmoid>=0 );
  if( params->hog_sigmoid ) {
    float_array hog_ravel = {hog->pixels,npix*hog->tz};
    sigmoid_array( &hog_ravel, params->hog_sigmoid, 0, nt);
  }
  //hash_layers(hog)
  
  // final smoothing
  assert( params->post_smoothing>=0 );
  if( params->post_smoothing )
    smooth_hog_gaussian( hog, params->post_smoothing, nt );
  //hash_layers(hog)
  
  // add ninth dimension and normalize per-pixel
  float* ninth_layer = hog->pixels + hog->tz*npix;
  for(int i=0; i<npix; i++) 
    ninth_layer[i] = params->ninth_dim;
  hog->tz++;
  //hash_layers(hog)
  if( params->norm_pixels )
    norm_layers( hog, 1, nt );
  //hash_layers(hog);D(0)getchar();
  
  return hog;
}

































