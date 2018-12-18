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
#ifndef ___PIXEL_DESC_H___
#define ___PIXEL_DESC_H___
#include "image.h"
#include "array_types.h"


// pixel descriptor params
typedef struct {
  float presmooth_sigma;  // image pre-smoothing
  float mid_smoothing;    // smoothing of oriented gradients (before sigmoid)
  float post_smoothing;   // smoothing of oriented gradients (after  sigmoid)
  float hog_sigmoid;      // sigmoid strength
  float ninth_dim;        // small constant for gradient-less area
  bool norm_pixels;       // 1: normalize pixels separately / 0: normalize atomic patches
  
} desc_params_t;


// set default params
void set_default_desc_params( desc_params_t* params );


/* extract pixel descriptors (pixel-wise HOG)
*/
float_layers* extract_desc( image_t* _img, const desc_params_t* params, int nt );

#endif
