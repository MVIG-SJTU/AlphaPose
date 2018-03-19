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
#include "conv.h"
#include "std.h"
#include "omp.h"
#include "maxfilter.h"

extern "C" {
#include <immintrin.h>
#define integer int
#define real float
extern int saxpy_(integer *n, real *sa, real *sx, integer *incx, real *sy, integer *incy);
extern int sscal_(integer *n, real *sa, real *sx, integer *incx);
}


static inline void fast_set_val( float * __restrict__ a, long d, const float val) {
  if(val) {
    int j;
    for(j=0; j<d; j++)
      a[j] = val;
  } else
    memset(a,0,d*sizeof(float));
}
static inline void fast_add_val( float * __restrict__ a, long d, const float val) {
  int j;
  for(j=0; j<d; j++)
    a[j] += val;
}
static inline void fast_set_vec( float * __restrict__ dest,
                                 const float * __restrict__ src, int d, const float mul) {
  if( mul==1) 
    memcpy(dest,src,d*sizeof(float));
  else {
    int j;
    for(j=0; j<d; j++)
      dest[j] = mul*src[j];
  }
}
static inline void fast_add_vec( float * __restrict__ dest,
                                 const float * __restrict__ add, int d, float mul) {
  if(d<=4) {
    int j;
    for(j=0; j<d; j++)
      dest[j] += mul*add[j];
  } else {
    int inc = 1;
    saxpy_( &d, &mul, (float*)add, &inc, (float*)dest, &inc );
  }
}
static inline void fast_div( float * __restrict__ a, long d, const float div) {
  const float divi = 1/div;
//  assert( ((long)a & 15) == 0 && (d & 3) == 0 );
//  const float _divi4[] = {divi,divi,divi,divi};
//  __v4sf *a4 = (__v4sf*)a;
//  __v4sf *divi4 = (__v4sf*)_divi4;
//  int e = d>>2;
//  while(e--) *a4++ *= (*divi4);
  int j;
  for(j=0; j<d; j++)
    a[j] *= divi;
}

static inline float* fast_set_trans( float * dest, const float * src, const float mul,
                                     int dx, int dy, const int tx, const int ty, const int ex, const float def ) {
  if(mul==0)  {
    memset(dest,0,sizeof(float)*(tx+ex)*(ty+ex));
    return dest+(tx+ex)*(ty+ex);
  }
  if(dx>tx) dx=tx;  // after those alues, nothing happens anyway
  if(dy>ty) dy=ty;
  if(-dx>tx) dx=-tx;
  if(-dy>ty) dy=-ty;
  
  #define add_default(n)  {fast_set_val(dest,(n),mul*def); dest+=(n);}
  float* _dest = dest;
  
  // paste -v zeros rows
  if(dy<0) add_default(-dy*(tx+ex));
  
  src += MAX(0,dx);
  const int row_len = MIN(tx,tx+dx+ex) - MAX(0,dx);
  int j;
  for(j=MAX(0,dy); j<MIN(ty,ty+dy+ex); j++) {
    
    // paste -u zeros cols
    if(dx<0) add_default(-dx);
    
    // past image
    fast_set_vec(dest,src+j*tx,row_len,mul);
    dest += row_len;
    
    // paste +u zeros cols
    if(dx>=0) {add_default(dx)
    if(ex) add_default(ex)}
  }
  
  // paste +v zeros rows
  if(dy>=0){add_default(dy*(tx+ex))
  if(ex) add_default(ex*(tx+ex))}
  
  #undef add_default
  assert( dest-_dest == (tx+ex)*(ty+ex) );
  return dest;
}

static inline float* fast_add_trans( float * dest, const float * src, const float mul,
                                     int dx, int dy, const int tx, const int ty, const int ex, const float def ) {
  if(mul==0)  return dest+(tx+ex)*(ty+ex);
  if(dx>tx) dx=tx;  // after those alues, nothing happens anyway
  if(dy>ty) dy=ty;
  if(-dx>tx) dx=-tx;
  if(-dy>ty) dy=-ty;
  #define add_default(n)  {fast_add_val(dest,n,def*mul); dest+=n;}
  float* _dest = dest;
  
  // paste -v zeros rows
  if(dy<0) add_default(-dy*(tx+ex));
  
  src += MAX(0,dx);
  const int row_len = MIN(tx,tx+dx+ex) - MAX(0,dx);
  int j;
  for(j=MAX(0,dy); j<MIN(ty,ty+dy+ex); j++) {
    
    // paste -u zeros cols
    if(dx<0) add_default(-dx);
    
    // past image
    fast_add_vec(dest,src+j*tx,row_len,mul);
    dest += row_len;
    
    // paste +u zeros cols
    if(dx>=0) {add_default(dx)
    if(ex) add_default(ex)}
  }
  
  // paste +v zeros rows
  if(dy>=0){add_default(dy*(tx+ex))
  if(ex) add_default(ex*(tx+ex))}
  
  #undef add_default
  assert( dest-_dest == (tx+ex)*(ty+ex) );
  return dest;
}


static inline void norm_norm( float* norms, int nb, float mode ) {
    int i;
    if( mode < 0 ) 
      assert(!"error: unknown norm mode");
    else if( mode == 0.5 ) {
      for(i=0; i<nb; i++)
        norms[i] = sqrt(sqrt(norms[i]));
    } else if( mode < 1 ) {
      mode *= 0.5;  // cumulate with initial 1/sqrt(.)
      for(i=0; i<nb; i++)
        norms[i] = pow(norms[i], mode);
    } else if( mode == 1 ) {
      for(i=0; i<nb; i++)
        norms[i] = sqrt(norms[i]);
    } else if( mode > 1 )
      assert(!"error: unknown norm mode");
}


/* normalize each pixel of a multi-layers image 
   norm = {0:nothing, 1:L2-normalization, 0-1: normalization by (L2-norm)**<norm> }
*/
void norm_layers( float_layers* res, float norm, int n_thread ) {
  if(norm==0) return;
  
  const int layer_size = res->tx*res->ty;
  const int n_layers = res->tz;
  float* norms = NEWAC(float,layer_size);
  long l;
  
  for(l=0; l<n_layers; l++) {
    float* r = res->pixels + l*layer_size; 
    int i;
    #if defined(USE_OPENMP)
    #pragma omp parallel for num_threads(n_thread)
    #endif
    for(i=0; i<layer_size; i++)
      norms[i] += r[i]*r[i];
  }
  norm_norm( norms, layer_size, norm );
  
  for(l=0; l<n_layers; l++) {
    float* r = res->pixels + l*layer_size; 
    int i;
    #if defined(USE_OPENMP)
    #pragma omp parallel for num_threads(n_thread)
    #endif
    for(i=0; i<layer_size; i++)
      r[i] /= norms[i]+1e-8;
  }
  
  free(norms);
}


/* Return the vectorized dimension of a HOG patch
*/
int get_patch_desc_dim( float_layers* hog, int patch_size ) 
{
    return patch_size*patch_size * hog->tz; // number of dimensions of an atomic patch descriptor
}


/* Sample a set of patches from a HOG image.
   grid : array of (x,y) position of the patches
   size: size of the patches, ie. [x,x+size[ x [y,y+size[
   res: result array, n_patches x desc_dim
        desc_dim = n_layers * size**2
   norms: result, n_patches x 1, norm of each patch
*/
void _sample_patches( float_layers* hog, float_layers* color, int_image* grid, int size, float norm, 
                      float_image* res, float_array* norms, int n_thread ) {
  const int tx = hog->tx;
  const long npix = tx*hog->ty;
  assert( grid->tx == 2 );
  const int n_patches = grid->ty;
  assert( res->ty == n_patches );
  const int n_layers = hog->tz;
  const int n_colors = (color? color->tz: 0);
  const int color_npix = (color? color->tx*color->ty: 0);
  const int desc_size = size*size*n_layers + (color? color->tz: 0);
  assert(res->tx == desc_size );
  
  int n;
  
  #if defined(USE_OPENMP)
  #pragma omp parallel for num_threads(n_thread)
  #endif
  for(n=0; n<n_patches; n++) {
    float *r = res->pixels + desc_size*n;
    int *p = grid->pixels + 2*n;
    // copy hog
    int x=p[0],y=p[1];
    assert(0<=x && x+size<=tx);
    assert(0<=y && y+size<=hog->ty);
    int l,j;
    for(l=0; l<n_layers; l++) {
      float* h = hog->pixels + l*npix + y*tx + x;
      for(j=0; j<size; j++) {
        memcpy(r, h, size*sizeof(float));
        h += tx;
        r += size;
      }
    }
    if(!color)  continue;
    // copy color
    float* c = color->pixels + (y+size/2)*color->ty + (x+size/2);
    for(l=0; l<n_colors; l++) 
      *r++ = c[l*color_npix];
  }
  
  if(norm) {
    float* normp = norms ? norms->pixels : NEWAC(float, n_patches);
    if(norms) {
      assert(norms->tx==n_patches);
      memset(normp,0,n_patches*sizeof(float));
    }
    
    #if defined(USE_OPENMP)
    #pragma omp parallel for num_threads(n_thread)
    #endif
    for(n=0; n<n_patches; n++) {
      float *r = res->pixels + desc_size*n;
      int l;
      for(l=0; l<desc_size; l++)
        normp[n] += r[l]*r[l];
    }
    norm_norm( normp, n_patches, norm );
    
    #if defined(USE_OPENMP)
    #pragma omp parallel for num_threads(n_thread)
    #endif
    for(n=0; n<n_patches; n++) {
      float *r = res->pixels + desc_size*n;
      int l;
      float nn = normp[n]+1e-8;
      for(l=0; l<desc_size; l++)
        r[l] /= nn;
    }
    
    if(!norms)  free(normp);
  }
}




static inline int retrieve_children( const int x, const int y, const int_cube* child_grid ) {
  const int size0_div2 = child_grid->pixels[0];
  const int step0 = child_grid->tx==1 && child_grid->ty==1 ? 1 : 
                                        MAX( child_grid->pixels[2]-child_grid->pixels[0], 
                                             child_grid->pixels[1+2*child_grid->tx]-child_grid->pixels[1] );
  int i = (x-size0_div2)/step0;
  int j = (y-size0_div2)/step0;
  assert( x==(i*step0+size0_div2) || !"error: child_grid does not match current grid" );
  assert( y==(j*step0+size0_div2) || !"error: child_grid does not match current grid" );
  if( i<0 || i>=child_grid->tx )  return -1;
  if( j<0 || j>=child_grid->ty )  return -1;
  return i+j*child_grid->tx;
}

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
                         int_cube* grid, int_cube* children, float_image* norms ) {
  assert(grid->tz==2);
  const int ntx = grid->tx; // should be == 1+(tx-size)/step so that patches do not pass the border
  const int nty = grid->ty; // should be == 1+(ty-size)/step so that patches do not pass the border
  
  /* grid[i,j] = ( offset + i*step, offset + j*step )
    
    connection between two scales:
    x cell position in lower scale == x position of children in upper scale
    child_offset + child_i*child_step = offset + i*step + (2*u/(nc-1)-1)*size/4
  */
  
  int i,j,u,v;
  int* r = grid->pixels;
  
  if( !child_grid ) {
    // this is the first scale: 
    // we just return a grid of step size*(1-overlap/2) in [0, tx[ x [0, ty[
    
    for(j=0; j<nty; j++)
      for(i=0; i<ntx; i++) {
        *r++ = offset + i*step;
        *r++ = offset + j*step;
      }
  } else {
    assert(child_grid->tz==2);
    ASSERT_SAME_SIZE( child_grid, child_norms );
    assert( children );
    const int nc = sqrt(children->tz); // number of children per row or col
    assert( children->tz==pow2(nc) );
    ASSERT_SAME_SIZE( grid, children );
    ASSERT_SAME_SIZE( grid, norms );
    // this is at least second scale
    // we return a grid of step size*(1-overlap/2) in [0, tx[ x [0, ty[
    
    const int quarter = size/4;
    assert(4*quarter==size);
    int* c = children->pixels; 
    float *n = norms->pixels;
    memset(n,0,ntx*nty*sizeof(float));
    for(j=0; j<nty; j++)
      for(i=0; i<ntx; i++) {
        int x = offset + i*step;
        int y = offset + j*step;
        *r++ = x;
        *r++ = y;
        
        // accumulate norms from 2x2 or 3x3 neighbors        
        for(v=0; v<nc; v++)
          for(u=0; u<nc; u++,c++) {
            // we want to index the children at position:
            // ( center_x + (2*u/(nc-1)-1)*size/4, center_y + (2*v/(nc-1)-1)*size/4 )
            *c = retrieve_children( x+(2*u/(nc-1)-1)*quarter, y+(2*v/(nc-1)-1)*quarter, child_grid );
            if(*c>=0) *n += child_norms->pixels[*c];
          }
        n++;
      }
  }
}


/* Prepare image for dotprod : dot(patches, res)
   where patches is n_patches x patch_dim
   set outside of the image to be equal to (0,...,ninth_val)
*/
void _prepare_dotprod_convolution( float_layers* img, int patch_size, float ninth_val, int extend, 
                                   float_layers* res, int n_thread ) {
  assert( img->tx+extend == res->tx );
  assert( img->ty+extend == res->ty );
  const int n_layers = img->tz;
  const int tx = img->tx;
  const int ty = img->ty;
  const int npix = tx*ty;
  const int npixex = (tx+extend)*(ty+extend);
  assert( res->tz==patch_size*patch_size*img->tz );
  
  long l;
  const int first_half = patch_size/2;  // half-size
  const int second_half = patch_size - first_half;
  const int layer_size = patch_size*patch_size*npixex;
  
  #if defined(USE_OPENMP)
  #pragma omp parallel for num_threads(n_thread)
  #endif
  for(l=0; l<n_layers; l++) {
    float* img_pix = img->pixels + l*npix;
    float* r = res->pixels + l*layer_size; 
    int u,v;
    // copy translated version of the image into res
    for(v=-first_half; v<second_half; v++)
      for(u=-first_half; u<second_half; u++)
        r = fast_set_trans( r, img_pix, 1, u, v, tx, ty, extend, l+1<n_layers? 0 : ninth_val );
  }
}


float_layers* prepare_dotprod_convolution( float_layers* hog, int patch_size, int extend, float norm, int nt )
{
  assert(0<=extend and extend<=1);
  const int nh = get_patch_desc_dim(hog,patch_size);
  const int etx = hog->tx+extend; // extend a bit the image
  const int ety = hog->ty+extend;
  
  float_layers* res = NEW(float_layers);
  *res = empty_layers(float,etx,ety,nh);
  
  float ninth_val = 0;
  _prepare_dotprod_convolution( hog, patch_size, ninth_val, extend, res, nt );
  
  if( norm )  norm_layers( res, norm, nt );
  return res;
}


inline float sum_array_f(const float* a, int n) {
  int i=n;
  double res = 0;
  while(i--)  res+=a[i];
  return (float)res;
}


extern "C" {
int sgemm_(char *transa, char *transb, integer *m, integer *
           n, integer *k, float *alpha, float *a, integer *lda, float *b, integer *
           ldb, float *beta, float *c, integer *ldc);
}

/* matrix-matrix multiplication with several SGEMM (each is single-threaded)
   res = dot(patches, convolved_hog)
   P*npix    P * nh     nh * npix
*/
void _dotprod( float_image* patches, float_layers* convolved_hog, float_layers* res, int n_thread ) {
  int nh = patches->tx;
  assert( nh == convolved_hog->tz );
  ASSERT_SAME_IMG_SIZE( convolved_hog, res );
  int P = patches->ty;
  assert( res->tz == P );
  int threadP = 1 + (P-1) / n_thread; // how many patches per thread
  int npix = (int)IMG_SIZE(convolved_hog);
  
  int l;
  #if (defined(USE_OPENMP) && !defined(MULTITHREADED_BLAS))
  #pragma omp parallel for num_threads(n_thread)
  #else
  n_thread = 1; // BLAS is already multithreaded
  threadP = P;
  #endif
  for(l=0; l<n_thread; l++) {
    // we do dotprod( patches[l*threadP : (l+1)*threadP], convolved_hog )
    long start = l*threadP;
    long end   = MIN(P,(l+1)*threadP);
    int np = int(end - start);
    float* p = patches->pixels + nh*start;
    float* r = res->pixels + npix*start;
    
    // blas fast matrix-matrix product
    char T='n'; float alpha = 1, beta = 0;
    sgemm_( &T, &T, &npix, &np, &nh, &alpha, 
            convolved_hog->pixels, &npix, 
            p, &nh, &beta, r, &npix);
  }
}

inline void transpose_scalar_block(const float *A, float *B, const int lda, const int ldb, 
                                   const int block_row, const int block_col) {
    for(int i=0; i<block_row; i++) 
        for(int j=0; j<block_col; j++)
            B[j*ldb + i] = A[i*lda +j];
}

// Transpose A (N rows by M cols) into B (M by N)
void transpose_matrix(const float_image* A, float_image* B, int nt) {
    const int n = A->ty, m = A->tx;
    assert( n==B->tx && m==B->ty );
    const int block_size = 16;
    const float* pA = A->pixels;
    float* pB = B->pixels;
    
    #ifdef USE_OPENMP
    #pragma omp parallel for num_threads(nt)
    #endif
    for(int i=0; i<n; i+=block_size) 
        for(int j=0; j<m; j+=block_size) 
            transpose_scalar_block(&pA[i*m +j], &pB[j*n + i], m, n, MIN(block_size, n-i), MIN(block_size, m-j));
}

extern "C" {
int sgemv_(char *transa, integer *m, integer * n, 
           float *alpha, float *a, integer *lda, 
           float *b, integer * ldb, float *beta, 
           float *c, integer * ldc);
}

/* convolution of each patch within a local neighborhood
    ngh_rad = max translation 
                neighborhood has size 2*ngh_rad
    patch at (x,y) is compared to patches in [y-ngh_rad : y+ngh_rad,
                                              x-ngh_rad : y+ngh_rad]
*/
void _dotprod_ngh_rad_T( int_cube* grid, float_image* patches, int ngh_rad,
                           float_cube* convolved_hog, float_layers* res_out, 
                           int_image* offsets, int n_thread ) {
  int nh = patches->tx;
  assert( nh == convolved_hog->tz );
  const int P = patches->ty;
  assert( IMG_SIZE(grid)==P && grid->tz==2 );
  const int tx = convolved_hog->tx;
  const int ty = convolved_hog->ty;
  
  // neighborhood size
  int res_tx = MIN(tx,2*ngh_rad);
  int res_ty = MIN(ty,2*ngh_rad);
  assert(res_tx<tx-1 || res_ty<ty-1 || !"ngh_rad is too large and results in loss of perf. Set ngh_rad=0 instead.");
  int res_npix = res_tx * res_ty;
  // allocate result
  *res_out = empty_layers(float, res_tx, res_ty, P);
  assert(res_out->pixels || !"error: ran out of memory before sgemm");
  *offsets = empty_image(int, 2, P);
  
  char T='t'; float alpha=1, beta=0; int one=1;
  
  #if defined(USE_OPENMP)
  #pragma omp parallel for num_threads(n_thread)
  #endif
  for(int j=0; j<res_ty; ++j) {
    // By organizing loops this way, 
    // we exploit overlap between patches.
    
    for(int l=0; l<P; l++) {
      float* p = patches->pixels + l*nh;
      float* r = res_out->pixels + l*res_npix;
      int left = MAX(0, MIN(grid->pixels[2*l+0] - ngh_rad, tx-2*ngh_rad));
      int top  = MAX(0, MIN(grid->pixels[2*l+1] - ngh_rad, ty-2*ngh_rad));
      if(j==0) {
        offsets->pixels[2*l+0] = left;
        offsets->pixels[2*l+1] = top;
      }
      float* c = convolved_hog->pixels + (left + top*tx)*nh;
      
      // blas fast matrix-vector product
      sgemv_( &T, &nh, &res_tx, &alpha, c + j*tx*nh, &nh, 
              p, &one, &beta, r + j*res_tx, &one);
    }
  }
}

/* correct the convolution on the boundaries of the image
   ttx, tty: true shape of the res_map (in case of using offsets)
*/
void rectify_conv( int patch_size, int nori, float_image* patches, int_image* offsets, 
                   const int ttx, const int tty, int extend, float_layers* res, int n_thread ) {
  const int n_patches = patches->ty;
  assert( n_patches == res->tz );
  //const int nori = patches->tx/pow2(patch_size);
  assert( patches->tx >= nori*pow2(patch_size) );
  const int tx = res->tx;  // real true shape because it has been extended
  const int ty = res->ty;
  const int first_half = patch_size/2;
  const int second_half = patch_size - first_half;  // in case patch_size is odd
  assert( offsets || (ttx==tx && tty==ty) );
  assert( !offsets || (ttx>=tx && tty>=ty) );
  assert( !offsets || (offsets->ty==res->tz && offsets->tx==2) );
  const long npix = IMG_SIZE(res);
  
  int l;
  #if defined(USE_OPENMP)
  #pragma omp parallel for num_threads(n_thread)
  #endif
  for(l=0; l<n_patches; l++) {
    // load offsets
    const int offi = offsets ? offsets->pixels[2*l+0] : 0;
    const int offj = offsets ? offsets->pixels[2*l+1] : 0;
    
    float sums[8]; // temporary norm of columns or rows
    assert( patch_size <= (int)(sizeof(sums)/sizeof(sums[0])) );
    int o,i,j;
    
    // horizontal boundaries
    memset(sums,0,sizeof(sums));
    float* p = patches->pixels + l*patches->tx;
    for(o=0; o<nori; o++)
      for(j=0; j<patch_size; j++)
        for(i=0; i<patch_size; i++)
          sums[j] += pow2(*p++);
    
    float old_norm = sqrt(sum_array_f(sums,patch_size));
    if( old_norm==0 ) continue;
    
    // upper boundary
    for(j=offj; j<first_half; j++) {
      float new_norm = sqrt(sum_array_f(sums+(first_half-j),second_half+j)); // sums to patch_size
      float mul = old_norm / (new_norm + 1e-8);
      float* r = res->pixels + l*npix + (j-offj)*tx;
      for(i=0; i<tx; i++) {
        r[i] *= mul;
        //assert(r[i]<1.1);
      }
    }
    // lower boundary
    for(j=tty-extend+1-second_half; j<offj+ty; j++) {
      float new_norm = sqrt(sum_array_f(sums,first_half+tty-extend-j)); // sums to patch_size
      float mul = old_norm / (new_norm + 1e-8);
      float* r = res->pixels + l*npix + (j-offj)*tx;
      for(i=0; i<tx; i++) {
        r[i] *= mul;
        //assert(r[i]<1.1);
      }
    }
    
    // vertical boundaries
    memset(sums,0,sizeof(sums));
    p = patches->pixels + l*patches->tx;
    for(o=0; o<nori; o++)
      for(j=0; j<patch_size; j++)
        for(i=0; i<patch_size; i++)
          sums[i] += pow2(*p++);
    
    // left boundary
    for(i=offi; i<first_half; i++) {
      float new_norm = sqrt(sum_array_f(sums+(first_half-i),second_half+i));
      float mul = old_norm / (new_norm + 1e-8);
      float* r = res->pixels + l*npix + (i-offi);
      for(j=0; j<ty; j++) {
        r[j*tx] *= mul;
        //assert(r[j*tx]<1.1);
      }
    }
    // right boundary
    for(i=ttx-extend+1-second_half; i<offi+tx; i++) {
      float new_norm = sqrt(sum_array_f(sums,first_half+ttx-extend-i));
      float mul = old_norm / (new_norm + 1e-8);
      float* r = res->pixels + l*npix + (i-offi);
      for(j=0; j<ty; j++) {
        r[j*tx] *= mul;
        //assert(r[j*tx]<1.1);
      }
    }
    
    // because we over-estimated the rectification for the corners, check that they do not overpass old_norm
    float* r = res->pixels + l*npix;
    for(j=offj; j<first_half; j++) {
      for(i=offi; i<first_half; i++)
        r[(j-offj)*tx+(i-offi)] = MIN(r[(j-offj)*tx+(i-offi)], old_norm);
      for(i=ttx-extend+1-second_half; i<offi+tx; i++)
        r[(j-offj)*tx+(i-offi)] = MIN(r[(j-offj)*tx+(i-offi)], old_norm);
    }
    for(j=tty-extend+1-second_half; j<offj+ty; j++) {
      for(i=offi; i<first_half; i++)
        r[(j-offj)*tx+(i-offi)] = MIN(r[(j-offj)*tx+(i-offi)], old_norm);
      for(i=ttx-extend+1-second_half; i<offi+tx; i++)
        r[(j-offj)*tx+(i-offi)] = MIN(r[(j-offj)*tx+(i-offi)], old_norm);
    }
  }
}


/* Compute the correlation of all patches with the second image (hog).
   In case of ngh_rad>0, the correlation is only computed in a small local neighborhood
   (whose size is parameterized by ngh_rad).
   if extend: width and height of output maps are extended
   if norm: correlation are normalized afterwards.
*/
void fastconv( float_image* patches, float_layers* hog, int patch_size, int ngh_rad, 
                int extend, float norm, int nt, res_scale* res ) {
  
  assert(0<=extend and extend<=1);
  float_layers* convolved_hog = prepare_dotprod_convolution( hog, patch_size, extend, norm, nt );
  assert( patches->tx==convolved_hog->tz);
  res->true_shape[0] = convolved_hog->tx;
  res->true_shape[1] = convolved_hog->ty;
  //hash_layers(convolved_hog)
  
  int_image* offsets = NULL;
  if( ngh_rad == 0 ) {  // no limit on translation
    // allocate result
    res->res_map = empty_layers(float, convolved_hog->tx, convolved_hog->ty, patches->ty);
    assert(res->res_map.pixels || !"error: ran out of memory before sgemm");
    
    // multi-threaded fast matrix product
    _dotprod( patches, convolved_hog, &res->res_map, nt );
    
  } else {  // ngh_rad>0: cropping res_map
    offsets = &res->offsets;
    
    // transpose hog: _dotprod is much faster this way
    float_cube convolved_hog_T = empty_cube(float, convolved_hog->tx, convolved_hog->ty, convolved_hog->tz);
    { float_image A = reshape_xy_z(float,  convolved_hog); // cast to 2D matrix without copy
      float_image B = reshape_z_xy(float, &convolved_hog_T);
      transpose_matrix( &A, &B, nt);
    }
    //hash_cube(&convolved_hog_T)
    
    // resized grid
    int_cube fgrid = cube_like(int, &res->grid);
    for(int i=0; i<CUBE_SIZE(&fgrid); i++) 
      fgrid.pixels[i] = res->grid.pixels[i]/res->f;
    //hash_cube(&fgrid)
    
    // multi-threaded fast matrix product
    _dotprod_ngh_rad_T( &fgrid, patches, ngh_rad, &convolved_hog_T, &res->res_map, offsets, nt );
    
    free(fgrid.pixels);
    free(convolved_hog_T.pixels);
    //hash_image(offsets)
  }
  free_layers(convolved_hog);
  
  // correct border effects on the correlation maps
  rectify_conv( patch_size, hog->tz, patches, offsets, res->true_shape[0], res->true_shape[1], 
                extend, &res->res_map, nt ); 
}



/* Compute: arr **= p
*/
void fastipow( float_layers* arr, const float p, int n_thread ) {
  const int n_layers = arr->tz;
  const long npix = arr->tx*arr->ty;
  int l;
  
  // optimization: precompute some values of pow(x,p)
  const int npc = 64;
  float precom[npc+1];
  for(l=0; l<=npc; l++) precom[l]= pow(l/(float)npc,p);
  const float maxindex = npc - 0.001;
  
  #if defined(USE_OPENMP)
  #pragma omp parallel for num_threads(n_thread)
  #endif
  for(l=0; l<n_layers; l++) {
    float* a = arr->pixels + l*npix;
    int i;
    for(i=0; i<npix; i++) {
//      arr[i] = pow(arr[i],p);
      float v = a[i]*npc;
      assert( v>=0 && v<npc+1 );
      if(v>maxindex)  v=maxindex;
      int n = int(v);
      float w = v-n;
      a[i] = (1-w)*precom[n] + w*precom[n+1];
    }
  }
}

/* Compute: arr = max(0,(arr-p)/(1-p))
*/
void fasthinge( float_layers* arr, const float p, int n_thread ) {
  const int n_layers = arr->tz;
  const long npix = arr->tx*arr->ty;
  int l;
  const float f = 1/(1-p);
  
  #if defined(USE_OPENMP)
  #pragma omp parallel for num_threads(n_thread)
  #endif
  for(l=0; l<n_layers; l++) {
    float* a = arr->pixels + l*npix;
    int i;
    for(i=0; i<npix; i++) {
      float v = a[i];
      a[i] = MAX(0,f*(v-p));
    }
  }
}

inline int max_array_i(const int* a, int n)  {
  int i=n;
  int res = INT_MIN;
  while(i--)  if(a[i]>res)  res=a[i];
  return res;
}

/* Normalize weights in border areas of width <gap>. 
   There are 9 areas: top-left, top-middle, top-right, ..., bottom-right.
   sum_divf indicates the current weight in those areas, i.e. values in the area 
   should be divided by the weight. But trans_inv allow to control the amount of
   normalization: 0=no normalization, 1=normal
*/
static inline void normalize_trans(const int tx, const int ty, const int gap, float* rmap, 
                                   const float trans_inv, float sum_divf[9] ) {
    if( trans_inv == 0 )  return;
    int i,j;
    for(i=0; i<9; i++) {
      if( sum_divf[i]>0 )
        sum_divf[i] = 1/pow(sum_divf[i], trans_inv);  // if trans_inv==1, no effect
    }
    for(j=0; j<gap; j++) {
      if(sum_divf[0])
        for(i=0; i<gap; i++)
          rmap[j*tx+i] *= sum_divf[0];
      if(sum_divf[1])
        for(i=gap; i<tx-gap; i++)
          rmap[j*tx+i] *= sum_divf[1];
      if(sum_divf[2])
        for(i=tx-gap; i<tx; i++)
          rmap[j*tx+i] *= sum_divf[2];
    }
    for(; j<ty-gap; j++) {
      if(sum_divf[3])
        for(i=0; i<gap; i++)
          rmap[j*tx+i] *= sum_divf[3];
      if(sum_divf[5])
        for(i=tx-gap; i<tx; i++)
          rmap[j*tx+i] *= sum_divf[5];
    }
    for(; j<ty; j++) {
      if(sum_divf[6])
        for(i=0; i<gap; i++)
          rmap[j*tx+i] *= sum_divf[6];
      if(sum_divf[7])
        for(i=gap; i<tx-gap; i++)
          rmap[j*tx+i] *= sum_divf[7];
      if(sum_divf[8])
        for(i=tx-gap; i<tx; i++)
          rmap[j*tx+i] *= sum_divf[8];
    }
}



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
                   int_array* assign, float_layers* res, int_image* res_offsets, int n_thread ) {
  const int nconv = children->ty; // number of convolutions to perform
  const int nc2 = children->tx;
  const int nc = sqrt(nc2);
  assert( nc*nc == nc2 );
  assert( res->tz == nconv );
  const int tx = child_map->tx;
  const int ty = child_map->ty;
  const long npix = tx*ty;
  ASSERT_SAME_SIZE( child_map, res );
  const int n_lower_conv = max_array_i(children->pixels,nconv*nc2)+1;
  int* cass = child_assign ? child_assign->pixels : NEWA(int,n_lower_conv);
  if(!child_assign) {for(int i=0; i<n_lower_conv; i++) cass[i]=i;}
  assert( !offsets || (offsets->pixels && offsets->tx==2 && offsets->ty==n_lower_conv &&
                       res_offsets && res_offsets->tx==2 && res_offsets->ty==nconv) );
  
  if(assign) {
    assert(0); // not supposed to happen
  } else {
    // normal case: no redundancy to exploit in response maps
    
    int l;
    #if defined(USE_OPENMP)
    #pragma omp parallel for num_threads(n_thread)
    #endif
    for(l=0; l<nconv; l++) {
      float *rmap = res->pixels + l*npix;
      
      int u,v,c,ncall=0;  // children number
      const int* const child = children->pixels + l*nc2;
      
      float sum_divf[9];
      memset(sum_divf,0,sizeof(sum_divf));
      int i,j;
      
      // first, choose an offset for the result rmap from the child offsets
      int offx=0, offy=0;
      if( offsets ) {
        int sum_ox=0, sum_oy=0, w=0;
        for(c=v=0; v<nc; v++) {
          int dy = (2*v/(nc-1)-1);
          for(u=0; u<nc; u++,c++) {
            int dx = (2*u/(nc-1)-1);
            
            if(child[c]<0 || cass[child[c]]<0) continue;
            
            sum_ox += offsets->pixels[2*child[c]+0] - dx*gap;
            sum_oy += offsets->pixels[2*child[c]+1] - dy*gap;
            w++;
          }
        }
        if(w==0)  w++;  // just in case
        offx = (int)floor(0.5 + sum_ox/float(w));
        offy = (int)floor(0.5 + sum_oy/float(w));
        
        // store result for later
        res_offsets->pixels[2*l+0] = offx;
        res_offsets->pixels[2*l+1] = offy;
      }
      
      for(c=v=0; v<nc; v++) {
        int dy = (2*v/(nc-1)-1);
        for(u=0; u<nc; u++,c++) {
          int dx = (2*u/(nc-1)-1);
          
          if(child[c]<0 || cass[child[c]]<0) continue;
          float divf = child_norms->pixels[child[c]]/norms->pixels[l];
          
          // difference with rmap's offset
          const int trans_x = dx*gap + (offsets? offx - offsets->pixels[2*child[c]+0] : 0);
          const int trans_y = dy*gap + (offsets? offy - offsets->pixels[2*child[c]+1] : 0);
          
          // count the sum of weights in every image area
          for(i=-1; i<=1; i++)for(j=-1; j<=1; j++)
            if(i*trans_x<=0 && j*trans_y<=0)
              sum_divf[4+j*3+i] += divf;  
          
          // add a translated version of map[children[c]] by (ox-dx,oy-dy)
          if(ncall++==0)  // first call
            fast_set_trans( rmap, child_map->pixels + cass[child[c]]*npix, divf, trans_x,trans_y, tx,ty, 0, 0 );
          else
            fast_add_trans( rmap, child_map->pixels + cass[child[c]]*npix, divf, trans_x,trans_y, tx,ty, 0, 0 );
        }
      }
      
      if( ncall == 0) // default = zeros
        memset(rmap, 0, npix*sizeof(float));
      
      // now we are supposed to rectify the boundaries (to perfect convolution)
      normalize_trans(tx, ty, gap, rmap, trans_inv, sum_divf );
      
      //assert(min_array_f(rmap,npix)>=0 && max_array_f(rmap,npix)<=1.001);
    }
  }
  if(!child_assign) free(cass);
  
  #define CHECK_MAPS(rmaps) assert(min_array_f((rmaps)->pixels,LAYERS_SIZE(rmaps))>=0 && \
                                 max_array_f((rmaps)->pixels,LAYERS_SIZE(rmaps))<=1.001)
  //CHECK_MAPS(res);
  
  return nconv;
}





















