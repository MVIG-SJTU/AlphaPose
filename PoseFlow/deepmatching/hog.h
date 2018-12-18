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
#ifndef ___HOG_H___
#define ___HOG_H___
#include "array_types.h"


/* * * * * * IMAGE SMOOTHING * * * * * * */

/* Smooth an image using a Gaussian filter.
*/
void _smooth_gaussian( UBYTE_image* img, float sigma, UBYTE_image* res, int n_thread );





/* * * * * * GRADIENT COMPUTATIONS * * * * * * */

/* Compute the dx,dy gradient on the image based on a [-1,0,1] mask. 
   method
     =0 :  no prior smoothing 
     =1 :  sobel smoothing 
*/
void _compute_grad_101( UBYTE_image* img, int method, float_layers* grad, int n_thread ); 





/* * * * * * pixel-HOG COMPUTATIONS * * * * * * */

/* Compute the Histogram of oriented gradient for each pixel.
   Number of orientations is determined by hog->tz;
   method determines orientation bining: 
     =0 : atan + linear interpolation
     =1 : fast cos projection
*/
void _compute_hog( float_layers* grad, int method, float_layers* hog, int n_thread );

/* Compute per-pixel HOG of 8 directions using a different pipeline.
   The method uses 4 oriented filters extremely simple ([-1,1])
*/
void _compute_hog_8_direct( UBYTE_image* image, float_layers* hog_out, int n_thread );


/* Post-processing of the HOG: cross-orientation inhibition.
   for one pixel i and orientation o: hog[i,o] = max(0, hog[i,o] - coef*hog[i,:].mean())
   This is useful for HOGs computed from cosinus projection.
*/
void subtract_mean_ori( float_layers* hog, float coef, int n_thread );


/* Pass the gradient image through a sigmoid
   lambda v: 2/(1 + exp(-coef*v + offset)) - 1
*/
void sigmoid_array( float_array* img, float coef, float offset, int n_thread );


/* Compute a spatially smoothed version of the HOG.
*/
void smooth_hog_gaussian( float_layers* hog, float smoothing, int n_thread );


































#endif
