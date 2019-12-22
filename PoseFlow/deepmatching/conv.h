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
#ifndef ___CONV_H___
#define ___CONV_H___
#include "array_types.h"
#include "deep_matching.h"


/* Return the vectorized dimension of a HOG patch
*/
int get_patch_desc_dim( float_layers* hog, int patch_size );


/* Sample a set of patches from a HOG image.
   pos : array of (x,y) position of the patches
   size: size of the patches, ie. [x,x+size[ x [y,y+size[
   res: result array, n_patches x desc_dim
        desc_dim = n_layers * size**2
   norms: result, n_patches x 1, norm of each patch
*/
void _sample_patches( float_layers* hog, float_layers* color, int_image* pos, int size, float norm, 
                      float_image* res, float_array* norms, int n_thread );


/* normalize each pixel of a multi-layers image 
   norm = {0:nothing, 1:L2-normalization, 0-1: normalization by (L2-norm)**<norm> }
*/
void norm_layers( float_layers* res, float norm, int n_thread );


/* Prepare a grid of cell positions in the first image for a given scale. Big cells inherit the cell at the previous scale.
    size = size of cells at current scale
    offset, step = grid generator: (offset + i*step, offset + j*step)
    child_grid = grid of the previous layer (or None if first layer)
    child_norms = image containing the norms of the patch at the previous level
    grid = result center positions of cells in current scale
    children = index of cells in previous scale used to construct big cells
    norms = norms of the cells of this level
*/
void _prepare_big_cells( int size, int offset, int step, 
                         int_cube* child_grid, float_image* child_norms,
                         int_cube* grid, int_cube* children, float_image* norms );



/* Compute the correlation of all patches with the second image (hog).
   In case of ngh_rad>0, the correlation is only computed in a small local neighborhood
   (whose size is parameterized by ngh_rad).
   if extend: width and height of output maps are extended
   if norm: correlation are normalized afterwards.
*/
void fastconv( float_image* patches, float_layers* hog, int patch_size, int ngh_rad, 
                int extend, float norm, int nt, res_scale* res );



/* Compute the (sparse) convolutions specified by <children> on <map> and put the result in <res>.
   A standard order is assumed on the children: 
    a response map #p is built from the children[p] at positions 
      [(gap*dx,gap*dy) for dy in dys for dx in dxs]
      where dxs = [-1,1] or [-1,0,1]
            dys = [-1,1] or [-1,0,1]
   child_assign denote assignement of the children level, while assign is for the next level
   child_norms contain the norms of small patches and norms for big new cells
*/
int _sparse_conv( int_image* children, int_array* child_assign, int gap, float trans_inv,
                   float_layers* child_map, int_image* offsets, float_array* child_norms, float_array* norms, 
                   int_array* assign, float_layers* res, int_image* res_offsets, int n_thread );



/* Compute: arr **= p
*/
void fastipow( float_layers* arr, const float p, int n_thread );

/* Compute: arr = max(0,(arr-p)/(1-p))
*/
void fasthinge( float_layers* arr, const float p, int n_thread );

/* Compute: arr = exp(-arr)
*/
void fastnegexp( float_image* arr );



/* incorporate the color difference between patches into existing patch similarity
    
    formula: new_response = ( color_sim*addw + old_response*(1-addw) ) * ( mulw*color_sim + 1-mulw )
    
    if mulw=1, adddw=0, then:    new_response = old_response * color_sim
    if mulw=0, adddw=0.5,then:   new_response = (old_response + color_sim )/2
*/
void incorporate_color( int_cube* grid, int_array* assign, float_layers* lab0, float_layers* var0, 
                        float_layers* lab1, float_layers* var1, 
                        float_layers* res_maps, float L_std, float ab_std, int sym_dist, int n_opening, 
                        const float addw, const float mulw, int n_thread );

































#endif
