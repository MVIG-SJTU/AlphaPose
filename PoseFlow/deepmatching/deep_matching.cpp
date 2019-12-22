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
#include "deep_matching.h"
#include "std.h"
#include "conv.h"
#include "maxfilter.h"



// return size of atomic patches
int get_atomic_patch_size( const dm_params_t* params )
{
  int upsize = (1 << params->prior_img_downscale);
  return 4*upsize;
}

// crop dimensions to a multiple of patch_size
void get_source_shape( const int width, const int height, const int patch_size, int* res ) {
    // crop the reference image to a multiple of patch size
    res[0] = patch_size * int(width / patch_size);
    res[1] = patch_size * int(height / patch_size);
}

// extract pixel descriptor for both images
void extract_image_desc( image_t* img0, image_t* img1, const dm_params_t* params,
                         float_layers** desc0, float_layers** desc1 )
{
    // slightly reduce img0 size to fit the patch tiling
    int patch_size = get_atomic_patch_size( params );
    
    int size[2]; // = {width, height}
    get_source_shape( img0->width, img0->height, patch_size, size );
    image_crop(img0, size[0], size[1]);
    
    // extract gradient-based information
    *desc0 = extract_desc( img0, &params->desc_params, params->n_thread );
    *desc1 = extract_desc( img1, &params->desc_params, params->n_thread );
}


void avgpool2( float_layers* hog, const dm_params_t* params )
{
    int niter = params->prior_img_downscale;
    while(niter--) {
      float_layers res = empty_layers(float,hog->tx/2,hog->ty/2,hog->tz);
      _avgpool2(hog,&res,params->n_thread);
      
      // replace hog by res
      free(hog->pixels);
      *hog = res;
    }
}


/* compute the grid of parent cell position, and their connection to children cells
   cells can be half-overlapping if <overlap>=1
   <dense_step> forces the grid spacing if >0
*/
void prepare_big_cells( const int imshape[2], int cell_size, int overlap, int child_overlap, 
                        int_cube* child_grid, float_image* child_norms, int dense_step,
                        int_cube* grid, int_cube* children, float_image* norms )
{
    int offset, step, gtx, gty;
    if( dense_step ) {
      step = dense_step;
      offset = 0;
      // we do not care if the patches are overlapping outside the image
      #define grid_size(imsize) (1+imsize/step)
      gtx = grid_size(imshape[0]);
      gty = grid_size(imshape[1]);
      #undef grid_size
    } else {
      // we want patches fully included in the image
      offset = cell_size/2;
      step = cell_size/(overlap+1);
      #define grid_size(imsize) (1+MAX(0,imsize-2*offset)/step)
      gtx = grid_size(imshape[0]);
      gty = grid_size(imshape[1]);
      #undef grid_size
    }
    
    assert(!grid->pixels);
    *grid = empty_cube(int,gtx,gty,2);
    
    assert(0<=overlap && overlap<=1);
    int nc = pow2(2+child_overlap);  // number of children per cell
    if(child_grid) {
      assert(!norms->pixels);
      *norms = image_like(float,grid);
      assert(!children->pixels);
      *children = empty_cube(int,gtx,gty,nc);
    }
    
    _prepare_big_cells( cell_size, offset, step, child_grid, child_norms, grid, children, norms );
}


void sample_patches( float_layers* hog, int_cube* pos, int patch_size, int f, float norm, int n_thread, 
                     float_image* patches, float_array* norms ) 
{
    assert(norm>0);
    const int npos = pos->tx*pos->ty;
    int_image new_pos = empty_image(int,2,npos);
    for(int i=0; i<2*npos; i++)
      new_pos.pixels[i] = (pos->pixels[i]-patch_size/2)/f;
    
    patch_size /= f;
    const int nh = get_patch_desc_dim(hog,patch_size);
    
    assert(!patches->pixels);
    *patches = empty_image(float,nh,npos);
    assert(norms->tx==npos);
    
    _sample_patches( hog, NULL, &new_pos, patch_size, norm, patches, norms, n_thread );
    
    free(new_pos.pixels);
}


const float trans_inv = 0.9f; 

void convolve_atomic_patches( float_layers* source, float_layers* target, 
                              const dm_params_t* params, res_scale* first_level ) 
{
    const int extend = 1; // slightly spatially extend response maps
    const float norm = 1; // renorm patches
    
    const int f = first_level->f; // scale factor w.r.t. original image
    const int psize = first_level->patch_size; // current patch size
    
    // first, sample patches
    float_image patches = {0};
    assert(!first_level->norms.pixels);
    first_level->norms = image_like(float, &first_level->grid);
    float_array norms_arr = {first_level->norms.pixels, (int)IMG_SIZE(&first_level->norms)};
    sample_patches( source, &first_level->grid, psize, f, norm, params->n_thread, &patches, &norms_arr );
    //hash_image(&patches)
    
    // rectify the norm to a boolean (0 or 1) (useless ?)
    first_level->assign = empty_array(int,norms_arr.tx);
    int n=0, tx = patches.tx;
    for(int i=0; i<norms_arr.tx; i++) {
      norms_arr.pixels[i] = norms_arr.pixels[i]>0;
      
      // eliminate zero-norm patches
      if( norms_arr.pixels[i] ) {
        if( n < i ) // copy
          memcpy( patches.pixels + n*tx, patches.pixels + i*tx, tx*sizeof(float));
        first_level->assign.pixels[i] = n++;
      } else 
        first_level->assign.pixels[i] = -1;
      
      // convolution is not fully invariant to the image border: 
      // blank cells outside the image are a bit disadvantageous
      if( norms_arr.pixels[i] == 0 )
        norms_arr.pixels[i] = 1-trans_inv;
    }
    patches.ty = n; // update new number of valid patches
    
    //hash_image(&first_level->norms)
    //hash_image(&patches)
    
    // compute the first level convolutions
    fastconv( &patches, target, psize/f, params->ngh_rad/f, extend, norm, params->n_thread, first_level );
    
    free(patches.pixels);
}

int_image* maxpool3_and_subsample2( float_layers* hog, int true_shape[2], int_image* offsets, float_layers* res, int nt )
{
  assert(!res->pixels);
  if ( offsets->pixels == NULL )
    assert( hog->tx == true_shape[0] && hog->ty == true_shape[1] );
  
  // set downsampled size
  true_shape[0] = (true_shape[0]+1)/2;
  true_shape[1] = (true_shape[1]+1)/2;
  assert( true_shape[0]>0 && true_shape[1]>0 );
  
  if ( offsets->pixels == NULL ) {
    // joint max-pooling and subsampling
    *res = empty_layers(float, true_shape[0], true_shape[1], hog->tz);
    _max_filter_3_and_subsample_layers( hog, res, nt );
    return NULL;
    
  } else {
    // with offsets
    float_layers maxpooled_hog = layers_like(float, hog);
    _max_filter_3_layers( hog, &maxpooled_hog, nt );
    //CHECK_MAPS(&maxpooled_hog);
    
    // slightly bigger, so that mininum size always >= 2
    int width = (hog->tx+2)/2;
    int height = (hog->ty+2)/2;
    *res = empty_layers(float, width, height, hog->tz);
    _subsample2_offset( &maxpooled_hog, offsets, res, nt );
    free(maxpooled_hog.pixels);
    
    // compute new offsets
    int_image* res_offsets = NEW(int_image);
    *res_offsets = image_like(int, offsets);
    for(long i=0; i<IMG_SIZE(offsets); i++)
      res_offsets->pixels[i] = (int)floor( offsets->pixels[i]/2.f );
    return res_offsets;
  }
}

#define CHECK_MAPS(rmaps) assert(min_array_f((rmaps)->pixels,LAYERS_SIZE(rmaps))>=0 && \
                                 max_array_f((rmaps)->pixels,LAYERS_SIZE(rmaps))<=1.001)

/* aggregate response maps of children patches to form response maps of parent patches */
int sparse_conv( int_cube* children, int_array* children_assign, float_image* child_norms, 
                 int true_patch_size, float_layers* map, int_image* offsets, int nt,
                 res_scale* res )
{
  float_layers ext_map;
  if( MIN(map->tx,map->ty) < 5 ) {
    ext_map = zeros_layers(float,MAX(5,map->tx),MAX(5,map->ty),map->tz);
    for(int l=0; l<map->tz; l++)
      for(int j=0; j<map->ty; j++)
        for(int i=0; i<map->tx; i++)
          ext_map.pixels[(l*ext_map.ty + j)*ext_map.tx + i] = map->pixels[(l*map->ty + j)*map->tx + i];
    map = &ext_map;
    res->true_shape[0] = ext_map.tx;
    res->true_shape[1] = ext_map.ty;
  }
  
  int_image _children = reshape_z_xy(int, &res->children);
  
  if( offsets )
    res->offsets = empty_image(int, 2, _children.ty);
  
  assert(!res->res_map.pixels);
  res->res_map = empty_layers(float, map->tx, map->ty, _children.ty);
  int gap = true_patch_size / 4;
  assert(gap>0);
  float_array _norms = reshape_xy(float, &res->norms);
  float_array _child_norms = reshape_xy(float, child_norms);
  
  // allocate useless assign
  res->assign = empty_array(int, res->res_map.tz);
  for(int i=0; i<res->assign.tx; i++) res->assign.pixels[i] = i;
  
  int_array* _assign = NULL;
  int_array* _ch_assign = children_assign->pixels ? children_assign : NULL;
  int n = _sparse_conv( &_children, _ch_assign, gap, trans_inv, map, offsets, 
                        &_child_norms, &_norms, _assign, &res->res_map, &res->offsets, nt );
  //CHECK_MAPS(res);
  
  if(map==&ext_map) free(ext_map.pixels);
  return n;
}

res_scale new_pyramid_level(int f, int psize) 
{
  res_scale res = {0};          // initialize everything to 0/NULL
  res.f = f;                    // subsampling factor with respect to original image size
  res.patch_size = psize;       // patch size in original image coordinates
  return res;
}

// Compute the multi-scale pyramid response
void compute_matching_pyr( float_layers* source, float_layers* target, const dm_params_t* params,
                              matching_pyramid_t& res_maps )
{
    const int src_shape[2] = {source->tx, source->ty};
    int L = 0;  // current pyramid level
    const int atomic_psize = get_atomic_patch_size( params );
    int psize = atomic_psize; // will grow by a factor 2 at each level
    int f = psize/4;  // initial scaling factor
    
    // subsample if needed
    avgpool2( source, params );
    avgpool2( target, params );
    
    //hash_layers(source)
    //hash_layers(target)
    
    res_maps.clear();
    res_maps.push_back(new_pyramid_level(f,psize));
    res_scale *child, *last = &res_maps[res_maps.size()-1];
    
    // compute the initial patches in source image
    if( params->verbose ) std_printf("layer %d, patch_size = %dx%d\n", L, psize, psize);
    prepare_big_cells( src_shape, psize, params->overlap<L+1, 0, NULL, NULL, 0, &last->grid, NULL, NULL );
    //hash_cube(&last->grid)
    
    //hash_layers(source)
    convolve_atomic_patches( source, target, params, last );
    //hash_layers(&last->res_map)
    if( params->verbose ) 
      std_printf("remaining %ld big cells (actually, %d are unique)\n", IMG_SIZE(&last->grid), last->res_map.tz);
    
    // non-linear correction
    if( params->nlpow>0 ) 
      fastipow( &last->res_map, params->nlpow, params->n_thread );
    
    //hash_layers(&last->res_map)
    
    const int dense_step = params->subsample_ref ? 0 : psize/(1+(params->overlap<1));
    
    // aggregate patches for all subsequent levels
    while( 2*psize <= MIN(params->max_psize, MAX(src_shape[0], src_shape[1])) ) {
        L++;
        f *= 2;
        psize *= 2;
        res_maps.push_back(new_pyramid_level(f,psize));
        child = &res_maps[res_maps.size()-2]; // previous level
        last = &res_maps[res_maps.size()-1];  // current level
        if( params->verbose ) std_printf("layer %d, patch_size = %dx%d\n", L, psize, psize);
        
        // max pooling + subsampling
        //CHECK_MAPS(&child->res_map);
        last->true_shape[0] = child->true_shape[0]; // will be modified in subsampled2()
        last->true_shape[1] = child->true_shape[1];
        float_layers subs_res_map = {0};
        int_image* offsets = maxpool3_and_subsample2( &child->res_map, last->true_shape, &child->offsets, 
                                                      &subs_res_map, params->n_thread );
        //CHECK_MAPS(&subs_res_map);
        
        // build the set of patches at this scale
        prepare_big_cells( src_shape, psize, params->overlap<L+1, params->overlap<L, 
                           &child->grid, &child->norms, dense_step, &last->grid, &last->children, &last->norms );
        //DA(last->true_shape,2)
        //hash_cube(&last->grid)
        //hash_image(&last->norms)
        //hash_cube(&last->children)
        
        // aggregate children response maps to form parent response maps
        sparse_conv( &last->children, &child->assign, &child->norms, psize/f, &subs_res_map, offsets, 
                     params->n_thread, last );
        free(subs_res_map.pixels);
        free_image(offsets);
        //CHECK_MAPS(&last->res_map);
        if( params->verbose ) 
          std_printf("remaining %ld big cells (actually, %d are unique)\n", IMG_SIZE(&last->grid), last->res_map.tz);
        
        // non-linear correction
        if( params->nlpow>0 ) 
          fastipow(&last->res_map, params->nlpow, params->n_thread );
        //hash_layers(&last->res_map)
    }
}


void free_matching_pyramid( matching_pyramid_t& res_maps ) {
  unsigned int i;
  for(i=0; i<res_maps.size(); i++) {
    res_scale& level = res_maps[i];
    
    free(level.grid.pixels);
    free(level.norms.pixels);
    free(level.assign.pixels);
    free(level.res_map.pixels);
    free(level.max_map.pixels);
    free(level.children.pixels);
    free(level.passed.pixels);
  }
}


#ifdef __APPLE__
static int arg_sort_maxima(void* arr, const void* a, const void* b) {
  float diff = ((float*)arr)[5*(*(int*)a)+4] - ((float*)arr)[5*(*(int*)b)+4];
  return (diff<0) - (diff>0); // descending order
}
#else
static int arg_sort_maxima(const void* a, const void* b, void* arr) {
  float diff = ((float*)arr)[5*(*(int*)a)+4] - ((float*)arr)[5*(*(int*)b)+4];
  return (diff<0) - (diff>0); // descending order
}
#endif

void reorder_rows( int_image* img, int_array* order )
{
  assert(order->tx==img->ty);
  const int tx = img->tx;
  int_image res = image_like(int, img);
  
  for(int i=0; i<order->tx; i++)
    memcpy(res.pixels + i*tx, img->pixels+order->pixels[i]*tx, tx*sizeof(int));
  
  free(img->pixels);
  *img = res;
}

// return points corresponding to patch matches 
int_image* find_optimal_matchings( matching_pyramid_t& mp, const dm_params_t* params ) 
{
  const int nobordure = 0;
  int_image* maxima = NEW(int_image);
  int_array order = {0};
  
  if( params->maxima_mode ) { // normal process: maxima detection
    
    float th=0;
    int check_parents=false, check_children=false;
    
    float_array sc_maxima = empty_array(float,int(mp.size()));
    for(unsigned int i=0; i<mp.size(); i++) sc_maxima.pixels[i]=1;  // useless but well
    
    _extract_maxima( mp.data(), mp.size(), &sc_maxima, th, params->min_level, params->nlpow, 
                     check_parents, check_children, nobordure, maxima, params->n_thread );
    free(sc_maxima.pixels);
    
    order = empty_array(int,maxima->ty); 
    for(int i=0; i<maxima->ty; i++) order.pixels[i] = maxima->ty-1-i;  // last first
    
  } else { // we just analyse all cells at the top level
    const float_layers* rmap = &mp[mp.size()-1].res_map;
    const int tx = rmap->tx, txy=tx*rmap->ty;
    *maxima = empty_image(int, 5, (int)LAYERS_SIZE(rmap));
    
    for(int i=0; i<maxima->ty; i++) {
      int* row = maxima->pixels + 5*i;
      row[0] = mp.size()-1; // pyramid level
      row[1] = i/txy;       // layer number
      row[2] = i%tx;        // x position
      row[3] = (i%txy)/tx;  // y position
      ((float*)row)[4] = rmap->pixels[i];
    }
    //hash_image(maxima)
    
    order = empty_array(int,maxima->ty); 
    for(int i=0; i<maxima->ty; i++) order.pixels[i] = i;
    #ifdef __APPLE__
    qsort_r(order.pixels, maxima->ty, sizeof(int), maxima->pixels, arg_sort_maxima);
    #else
    qsort_r(order.pixels, maxima->ty, sizeof(int), arg_sort_maxima, maxima->pixels);
    #endif
  }
  
  if( params->verbose>0 ) 
    std_printf("found %d local matches\n",maxima->ty);
  
  // reorder maxima
  reorder_rows( maxima, &order );
  free(order.pixels);
  return maxima;
}


static inline float ptdot( const float* m, float x, float y ) {
  return x*m[0] + y*m[1] + m[2];
}

void apply_rot( float_cube* corres, float rot[6] ) {
  assert( corres->tz == 6 );
  const int nb = IMG_SIZE(corres);
  float* p = corres->pixels;
  
  for(int i=0; i<nb; i++) {
    // only apply to coordinates of the first image
    float x = p[0], y = p[1];
    p[0] = ptdot(rot+0, x, y);
    p[1] = ptdot(rot+3, x, y);
    p += 6;
  }
}


/* this function gather correspondences from each local maximum in the 
  response maps
*/
float_image* gather_correspondences( int src_shape[2], int target_shape[2], 
                                   matching_pyramid_t& scales, int_image* maxima,
                                   const dm_params_t* params, full_corres_t* corres_out )
{
    const int step = 4*scales[0].f; // bin size
    const int n_scales = (int)scales.size();
    const int tx = maxima->tx;
    const int n_maxima = maxima->ty;
    
    float_cube corres0 = zeros_cube(float, (src_shape[0]+step-1)/step, (src_shape[1]+step-1)/step,6);
    float_cube corres1 = zeros_cube(float, (target_shape[0]+step-1)/step, (target_shape[1]+step-1)/step,6);
    
    int i;
    // allocate temporary optimization maps
    for(i=0; i<n_scales; i++) {
      long size = LAYERS_SIZE(&scales[i].res_map);
      if( params->low_mem && size > 1000003 ) size = 1000003; // big prime
      assert( size <= 2147483647 || !"try using -mem parameter");
      scales[i].passed = zeros_array(float, (int)size);
    }
    
    #if defined(USE_OPENMP)
    #pragma omp parallel for schedule(static,1) num_threads(params->n_thread)
    #endif
    for(i=0; i<n_maxima; i++) {
      if(params->verbose && i%100==0) std_printf("\rgathering correspondences %d%%...",100*i/n_maxima);
      int* m = maxima->pixels + tx*i;
      int level = m[0], num_map = m[1];
      int x = m[2], y = m[3];
      assert(level<n_scales);
      
      if( scales[level].offsets.pixels ) {
        // add offset to form real image coordinates
        x += scales[level].offsets.pixels[2*num_map+0];
        y += scales[level].offsets.pixels[2*num_map+1];
      }
      
      if( params->scoring_mode )  // new mode
        _argmax_correspondences( scales.data(), level, num_map, x, y, ((float*)m)[4], 
                                    &corres0, step, &corres1, step, i );
      else  // old iccv mode
        _argmax_correspondences_v1( scales.data(), level, num_map, x, y, m[0]*((float*)m)[4], 
                                    &corres0, step, &corres1, step, i );
    }
    
    // free optimization maps
    for(i=0; i<n_scales; i++) {
      free( scales[i].passed.pixels );
      scales[i].passed.pixels = NULL;
    }
    
    if(params->verbose) std_printf("\n");
    
    if( params->rot45 ) { // rectify correspondences
      assert( corres_out );
      apply_rot( &corres0, corres_out->rot );
      apply_rot( &corres1, corres_out->rot );
    }
    
    // keep only reciprocal matches
    int nres;
    float* corres = _intersect_corres( &corres0, &corres1, &nres );
    float_image* res = NEW(float_image);
    *res = (float_image){corres, 6, nres};
    
    if( corres_out == NULL ) {
      free(corres0.pixels);
      free(corres1.pixels);
    }
    else {  // save unfiltered correspondences
      corres_out->corres0 = corres0;
      corres_out->corres1 = corres1;
    }
    return res;
}


void eye_rot3x3( float rot[6] ) {
  memset( rot, 0, 6*sizeof(float));
  rot[0] = rot[4] = 1;
}

inline float bilinear_interp(const float* img, const int tx, const int ty, float x, float y ) {
  if( x < 0 || x+1.001 >= tx ) return 0;  // outside
  if( y < 0 || y+1.001 >= ty ) return 0;  // outside
  int ix = int(x);
  int iy = int(y);
  img += ix + iy*tx;  // move pointer
  float rx = x - ix;
  float ry = y - iy;
  return (1-ry)*((1-rx)*img[0] + rx*img[1]) +
            ry *((1-rx)*img[tx]+ rx*img[tx+1]);
}

void scale_rot3x3( float rot[6], float sc ) {
  for(int i=0; i<6; i++)
    rot[i] *= sc;
}

void inv_rot3x3( float rot[6], float res[6] ) {
  assert( fabs((rot[0]*rot[4] - rot[1]*rot[3]) - 1) < 1e-6 );
  // because rot is unitary, invert == transpose
  res[0] = rot[0];
  res[1] = rot[3];
  res[3] = rot[1];
  res[4] = rot[4];
  res[2] = -rot[2]*rot[0] - rot[5]*rot[3];
  res[5] = -rot[2]*rot[1] - rot[5]*rot[4];
}



// rotate a descriptor HOG image by a given angle
float_layers* rotate45( float_layers* hog, const dm_params_t* params, full_corres_t* corres_out ) {
  assert( corres_out ); // we need it to write rot !
  const int patch_size = get_atomic_patch_size( params );
  const int n_rot45 = params->rot45;
  
  if( (n_rot45 % 8) == 0 ) {  // nothing to do
    eye_rot3x3( corres_out->rot );
    return hog;
  }
  const int tx = hog->tx;
  const int ty = hog->ty;
  
  // rotation matrix
  float angle = n_rot45 * M_PI / 4;
  float c = cos(angle), s = sin(angle);
  float rot[6] = {c, -s, 0, s, c, 0};
  // pt_in_original_image = rot * pt_in_rotated_image
  
  // determine center of rotation before
  float cx_before = tx/2.0;
  float cy_before = ty/2.0;
  // determine center of rotation after
  float corners[2][4] = {{0, (float)tx, (float)tx, 0}, {0, 0, (float)ty, (float)ty}};
  for(int i=0; i<4; i++) {  // rotate corners
    float x = corners[0][i], y = corners[1][i];
    corners[0][i] = ptdot(rot+0, x, y);
    corners[1][i] = ptdot(rot+3, x, y);
  }
  int rot_size[2] = {int(0.5 + max_array_f(corners[0], 4) - min_array_f(corners[0], 4)),
                     int(0.5 + max_array_f(corners[1], 4) - min_array_f(corners[1], 4)) };
  get_source_shape( rot_size[0], rot_size[1], patch_size, rot_size );
  float cx_after = rot_size[0]/2.0;
  float cy_after = rot_size[1]/2.0;
  // compute the translation
  rot[2] = cx_before - ptdot(rot+0, cx_after, cy_after);
  rot[5] = cy_before - ptdot(rot+3, cx_after, cy_after);
  
  // create result
  assert( hog->tz == 9 );
  float_layers* rot_hog = NEW(float_layers); 
  *rot_hog = empty_layers(float, rot_size[0], rot_size[1], 9);
  
  for(int c=0; c<hog->tz; c++) {
    const int src_c = (c<8) ? int((c+n_rot45+256)%8) : c; // roll channels except for last one (see hog.h)
    const float* f = hog->pixels + src_c * IMG_SIZE(hog);
    float* p = rot_hog->pixels + c * IMG_SIZE(rot_hog);
    
    for(int y=0; y<rot_size[1]; y++)
      for(int x=0; x<rot_size[0]; x++) {
        float rx = ptdot( rot+0, x, y);
        float ry = ptdot( rot+3, x, y);
        
        *p++ = bilinear_interp(f, tx, ty, rx, ry );
      }
  }
  
  // output inverted rot
  memcpy( corres_out->rot, rot, 6*sizeof(float) );
  
  return rot_hog;
}


// set default parameters
void set_default_dm_params( dm_params_t* params )
{
  // pixel descriptor params
  set_default_desc_params( &params->desc_params );
  
  // general parameters
  params->prior_img_downscale = 1;  // resolution R = 1/2^downscale, default = 1/2
  params->rot45 = 0;  // don't rotate the first image
  params->overlap = 999;  // don't use overlapping patches
  params->subsample_ref = false;  // don't subsample patches in reference image (=first image)
  params->nlpow = 1.4;
  params->ngh_rad = 0; // no limit by default
  params->maxima_mode = 0;  // don't use maxima, just start from all top patches
  params->min_level = 2;  // useless
  params->max_psize = 999; // maximum patch size
  params->low_mem = true; // optimize mem but then results are slightly unstable/non-reproducible
  params->verbose = 0;
  params->scoring_mode = 1; // improved scoring scheme
  params->n_thread = 1; // no multithreading by default
}


// main function
float_image* deep_matching( image_t* img0, image_t* img1, const dm_params_t* params, full_corres_t* corres_out )
{
  // verify parameters
  assert(between(0,params->prior_img_downscale,3));
  assert(between(0,params->overlap,999));
  assert(between(0,params->subsample_ref,1));
  assert(between(0.1,params->nlpow,10));
  assert(between(0,params->ngh_rad,1<<16));
  assert(between(0,params->maxima_mode,1));
  assert(between(0,params->min_level,4));
  assert(between(0,params->low_mem,1));
  assert(between(0,params->scoring_mode,1));
  assert(between(0,params->verbose,10));
  assert(between(1,params->n_thread,128));
  
  // extract pixel descriptors
  float_layers *source, *target;
  extract_image_desc( img0, img1, params, &source, &target );
  if( corres_out )  // the first image is rotated
    source = rotate45( source, params, corres_out );
  int src_shape[2] = {source->tx, source->ty};
  assert( LAYERS_SIZE(source) > 0 );
  int target_shape[2] = {target->tx, target->ty};
  assert( LAYERS_SIZE(target) > 0 );
  
  //hash_layers(source)
  //hash_layers(target)
  
  // compute local matchings
  matching_pyramid_t matching_pyr;
  compute_matching_pyr( source, target, params, matching_pyr );
  free_layers(source);
  free_layers(target);
  
  //hash_layers(&matching_pyr[matching_pyr.size()-1].res_map);
  
  // find optmial matchings (maxima)
  int_image* maxima = find_optimal_matchings(matching_pyr, params);
  
  //hash_image(maxima);
  
  // select the best displacements (maxpool merge)
  float_image* corres = gather_correspondences( src_shape, target_shape, matching_pyr, maxima, params, corres_out );
  
  //hash_image(corres);
  
  // free everything
  free_matching_pyramid(matching_pyr);
  free_layers(maxima);
  
  return corres;
}


void swap_first_second_img( float_cube* corres ) {
  assert( corres->tz == 6 );
  const int nb = IMG_SIZE(corres);
  float* p = corres->pixels;
  
  for(int i = 0; i < nb; i++) {
    float a = p[0];
    float b = p[1];
    float c = p[2];
    float d = p[3];
    *p++ = c;
    *p++ = d;
    *p++ = a;
    *p++ = b;
    p += 2;
  }
}

void rescale_corres( float_cube* corres, float f0, float f1, int code ) {
  assert( corres->tz == 6 );
  const int nb = IMG_SIZE(corres);
  float* p = corres->pixels;
  
  for(int i = 0; i < nb; i++) {
    p[0] *= f0;
    p[1] *= f0;
    p[2] *= f1;
    p[3] *= f1;
    p[5] = code;
    p += 6;
  }
}

// set default parameters
void set_default_scalerot_params( scalerot_params_t* params ) {
  params->fast = true;
  params->min_sc0 = 0;  // scale = 2^(-0/2) = 1
  params->max_sc0 = 5;  // scale = 2^(-5/2) = 0.176
  params->min_sc1 = 0;
  params->max_sc1 = 5;
  params->min_rot = 0;  // rot = 0*45 = 0
  params->max_rot = 8;  // rot = 8*45 = 360
}


// main function for scale/rotation invariant version
float_image* deep_matching_scale_rot( image_t* img0, image_t* img1, dm_params_t* params, 
                                      const scalerot_params_t* sr_params ) {
  // verify parameters
  assert(sr_params->min_sc0 < sr_params->max_sc0);
  assert(sr_params->min_sc1 < sr_params->max_sc1);
  assert(between(0, sr_params->min_sc0, 5));
  assert(between(0, sr_params->max_sc0, 5));
  assert(between(0, sr_params->min_sc1, 5));
  assert(between(0, sr_params->max_sc1, 5));
  assert(sr_params->min_rot >= 0);
  assert(between(1,sr_params->max_rot - sr_params->min_rot, 8));
  
  // init shape
  const int psize = get_atomic_patch_size(params);
  int imshape0[2]; 
  get_source_shape( img0->width, img0->height, psize, imshape0 );
  int imshape1[2] = {img1->width, img1->height};
  
  // check dm params to ensure everything goes fine from now on
  #define mean_dim(shape)  ((shape[0] + shape[1])/2)
  params->max_psize = MIN(mean_dim(imshape0), mean_dim(imshape1));
  const int verbose = params->verbose;
  params->verbose = MAX(0, verbose - 1);  // decrease for inner deepmatchings 
  
  // prepare output
  const int step0 = psize/2;
  const int step1 = psize/2;
  float_cube all_corres0 = zeros_cube(float, (imshape0[0]+step0/2-1)/step0, (imshape0[1]+step0/2-1)/step0, 6);
  float_cube all_corres1 = zeros_cube(float, (imshape1[0]+step1/2-1)/step1, (imshape1[1]+step1/2-1)/step1, 6);
  full_corres_t out;
  
  const int NS = 5;
  image_t *scaled_images1[NS] = {NULL};
  
  // loop over all scale*rot combinations
  for(int sc0 = sr_params->min_sc0; 
          sc0 < sr_params->max_sc0; 
          sc0++) {
    const float scale0 = pow(2, -0.5*sc0 ); // scale factor for img0
    assert( scale0<=1 && sc0<5 );
    image_t* scaled_img0 = ( scale0 >= 1 ) ? img0 : 
                            image_resize_bilinear_scale( img0, scale0 );
    
    for(int sc1 = sr_params->min_sc1; 
            sc1 < sr_params->max_sc1; 
            sc1++) {
      const float scale1 = pow(2, -0.5*sc1 ); // scale factor for img1
      assert( scale1<=1 && sc1<5 );
      // optimization, deactivate only if eg. both images are blurry
      if( sr_params->fast && !(scale0==1 || scale1==1)) continue;
      
      image_t* scaled_img1 = scaled_images1[sc1 - sr_params->min_sc1];
      if( scaled_img1 == NULL ) {
        scaled_img1 = ( scale1 >= 1 ) ? img1 : 
                        image_resize_bilinear_scale( img1, scale1 );
        // remember result
        scaled_images1[sc1 - sr_params->min_sc1] = scaled_img1;
      }
      
      for(int rotation = sr_params->min_rot;  
              rotation < sr_params->max_rot;  
              rotation++) {
        assert( rotation >= 0 );
        const int rot_scale_code = 8*(sc1*5+sc0) + (rotation%8); // cannot be negative, because of bin count
        
        if( verbose ) 
          std_printf( "processing scale = (x%g, x%g) + rotation = %d deg (code %d)...\n",
                           scale0, scale1, 45*rotation, rot_scale_code);
        
        float rot0[6], rot1[6];
        
        // compute correspondences with rotated+scaled image
        #define max_dim(img)  MAX(img->width, img->height)
        if( max_dim(scaled_img0) >= max_dim(scaled_img1) ) { // first image is always the largest
          params->rot45 = rotation;
          
          float_image* corres = deep_matching(scaled_img0, scaled_img1, params, &out );
          free_image( corres ); // we don't care
          
          inv_rot3x3(out.rot, rot0);
          eye_rot3x3(rot1);
          
        } else {  // scaled_img1 is larger
          params->rot45 = -rotation;
          
          float_image* corres = deep_matching(scaled_img1, scaled_img0, params, &out );
          free_image( corres ); // we don't care
          
          // swap first and second image coordinates
          memswap( &out.corres0, &out.corres1, sizeof(float_cube) );
          swap_first_second_img( &out.corres0 );
          swap_first_second_img( &out.corres1 );
          
          inv_rot3x3(out.rot, rot1);
          eye_rot3x3(rot0);
        }
        
        // change scale of correspondences
        rescale_corres( &out.corres0, 1/scale0, 1/scale1, rot_scale_code );
        rescale_corres( &out.corres1, 1/scale0, 1/scale1, rot_scale_code );
        scale_rot3x3(rot0, scale0);
        scale_rot3x3(rot1, scale1);
        
        // merge correspondences in the reference frame
        merge_corres( rot0, rot1, 
                      psize, psize, &out.corres0, &out.corres1, 2, 
                      step0, step1, &all_corres0, &all_corres1 ); // finer grid for merge
        
        free(out.corres0.pixels);
        free(out.corres1.pixels);
      }
    }
    
    // free memory
    if( img0 != scaled_img0 )
      image_delete( scaled_img0 );
  }
  
  // final intersection
  int nres;
  float* corres = _intersect_corres( &all_corres0, &all_corres1, &nres );
  float_image* res = NEW(float_image);
  *res = (float_image){corres, 6, nres};
  
  // free memory
  for(int i=0; i<NS; i++)
    if( scaled_images1[i] != img1 )
      image_delete( scaled_images1[i] );
  free(all_corres0.pixels);
  free(all_corres1.pixels);
  
  return res;
}

























