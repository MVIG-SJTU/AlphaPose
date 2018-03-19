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
#ifndef ___MAXFILTER_H___
#define ___MAXFILTER_H___
#include "array_types.h"
#include "deep_matching.h"


/* compute the 3x3 maximum filter on an image and store the result in <res>
*/
void _max_filter_3( float_image* img, float_image* res, int n_thread );


/* Same as above for float_layers* images
*/
void _max_filter_3_layers( float_layers* img, float_layers* res, int n_thread );


/* Subsample an array, equivalent to res = img[:,1::2,1::2]
*/
void _subsample2( float_layers* img, float_layers* res, int n_thread );

/* joint max-pooling and subsampling
*/
void _max_filter_3_and_subsample_layers( float_layers* img, float_layers* res, int n_thread );


/* Subsample an array, equivalent to res = trueimg[:,offset_y::2,offset_x::2]
   except at boundaries, where the rules are a bit more complex (see code)
*/
void _subsample2_offset( float_layers* img, int_image* offsets, float_layers* res, int n_thread );

/* Max-pool in 2x2 px non-overlapping cells
*/
void _maxpool2( float_layers* img, float_layers* res, int n_thread );


/* average-pool in 2x2 px non-overlapping cells
*/
void _avgpool2( float_layers* img, float_layers* res, int n_thread );


/* Return the list of parent cells of all cells of a given scale (parents are from the upper scale)
    children: list of children of the parent cells
    res: result matrix, n_cells_at_current_scale x n_max_parents
    res == -1 when there is no parent
*/
void _get_list_parents( int_cube* children, int_image* res );


/* Return a list of local maxima in the scale-space of scores
*/
void _extract_maxima( res_scale* scales, int n_scales, float_array* sc_factor, float th, int min_scale, float nlpow, 
                      int check_parents, int check_children, int nobordure, int_image* res_out, int n_thread );


/* Return the best assignment (= list of correspondences) for a given maxima
   from a pyramid top, this function returns 
   a list of weigthed correspondences (matches) between
   img0 pixels and img1 pixels
   index = index of the maxima (s,l,x,y), so that it can be linked to the correspondences it generated
*/
void _argmax_correspondences( res_scale* scales, int s, int l, int x, int y, float score, 
                              float_cube* res0, int step0, float_cube* res1, int step1, 
                              int index );

void _argmax_correspondences_v1( res_scale* scales, int s, int l, int x, int y, float score, 
                              float_cube* res0, int step0, float_cube* res1, int step1, 
                              int index );






/* Intersect 2 mappings: erase all correspondences that are not reciprocal 
*/
float* _intersect_corres( const float_cube* map0, const float_cube* map1, int* nres );


/* erase corres in the first array that are not in the second one
*/
void transfer_corres_score( const float_image* ref, float_cube* map0 );




/* merge correspondences from several rotated/scaled version of an image into a single common reference frame
    rot0 = 2x3 rotation matrix:   (pt in rotated img0) = rot0 * (pt in ref frame)
    rot1 = 2x3 rotation matrix:   (pt in rotated img1) = rot1 * (pt in ref frame)
    step0 and step1 are bin size of correspondences histograms
    tol >= 1 is the tolerance to grid rotation (default = 2)
    corres0, corres1:          correspondences histograms of rotated image 
    all_corres0, all_corres1:  correspondences histograms of reference frame (result)
*/
void merge_corres( float rot0[6], float rot1[6], int step0, int step1, 
                   float_cube* corres0, float_cube* corres1, float tol,
                   int all_step0, int all_step1, float_cube* all_corres0, float_cube* all_corres1 );


#endif



























