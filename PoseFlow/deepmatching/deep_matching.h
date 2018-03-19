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
#ifndef ___DEEP_MATCHING_H___
#define ___DEEP_MATCHING_H___
#include "array_types.h"
#include "pixel_desc.h"
#include "image.h"

#include <vector>
using namespace std;


// deep matching parameters
typedef struct {
  desc_params_t desc_params; 
  
  int prior_img_downscale;// downscale the image by 2^(this) prior to matching
  int rot45;              // rotate second img by (45*rot45) prior to matching
  int overlap;            // pyramid level at which patches starts to overlap (999 => no overlap at all)
  bool subsample_ref;     // true if larger patches higher in the pyramid are not densely sampled
  float nlpow;            // non-linear power rectification
  int ngh_rad;            // neighborhood size in pixels => crop res_map (0 == infinite)
  int maxima_mode;        // 1: standard / 0: from all top-level patches
  int min_level;          // minimum pyramid level to retrieve maxima
  int max_psize;          // maximum patch size
  int low_mem;            // use less memory to retrieve the maxima (but approximate result)
  int scoring_mode;       // 0: like ICCV paper / 1: improved scoring mode
  int verbose;            // verbosity
  int n_thread;           // parallelization on several cores, when possible
  
} dm_params_t;

// set default parameters
void set_default_dm_params( dm_params_t* params );

// scale & rotation invariant version
typedef struct {
  bool fast;              // avoid comparing small scaled versions of both images 
  int min_sc0, max_sc0;   // scale range of image0 (expressed as scale=2^(-n/2))
  int min_sc1, max_sc1;   // scale range of image1 (expressed as scale=2^(-n/2))
  int min_rot, max_rot;   // rotation range (expressed as multiples of 45 degrees)
  
} scalerot_params_t;  

// set default parameters
void set_default_scalerot_params( scalerot_params_t* params );


// response maps at a given scale
typedef struct {
  int f;                // subsampling factor with respect to original image size
  int patch_size;       // patch size in original image coordinates in first image
  int_cube grid;        // position (center) of each patch in first image
  float_image norms;    // norm of each patch in first image
  int_array assign;     // mapping between patches and their response maps
  float_layers res_map; // response map of the patches on the second image
  float_layers max_map; // max-filtered response map
  int true_shape[2];    // true res_map shape (width, height) in case of crop (if ngh_rad>0)
  int_image offsets;    // res_map offsets in case of crop (if ngh_rad>0)
  int_cube children;    // index of children patches in the previous level
  float_array passed;  // remember the best score so far at each response when doing argmax
  
} res_scale;  

typedef vector<res_scale> matching_pyramid_t;


// output correspondences
typedef struct {
  float x0, y0;   // position in first image (reference image)
  float x1, y1;   // position in second image (target image)
  float maxima;   // from which maxima it was generated (index)
  float score;    // matching score
} corres_t;

// for scale rot invariant matching
typedef struct {
  float rot[6];
  float_cube corres0;
  float_cube corres1;
} full_corres_t;


// main function. Returns a float_image where each row is <corres_t>
float_image* deep_matching( image_t* img0, image_t* img1, const dm_params_t* params, 
                            full_corres_t* corres_out );  // NULL if you don't use it

// main function for scale & invariant matching. output is same as above.
float_image* deep_matching_scale_rot( image_t* img0, image_t* img1, dm_params_t* params, 
                                      const scalerot_params_t* sr_params );


#endif



































