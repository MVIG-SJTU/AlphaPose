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
#include "hog.h"
#include "std.h"


/* compute horizontal gradient centered with [-1,0,1] mask 
*/
void _diff_horiz(int tx, int ty, UBYTE* pixels, float* res) {
  int x,y,pos=0;
  float* r=res;
  for(y=0; y<ty; y++,pos+=tx) {
    *r++ = pixels[1+pos] - pixels[0+pos];
    for(x=1; x<tx-1; x++)
      *r++ = pixels[x+1+pos] - pixels[x-1+pos];
    *r++ = pixels[x+pos] - pixels[x-1+pos];
  }
}

/* compute vertical gradient centered with [-1,0,1] mask 
*/
void _diff_vert(int tx, int ty, UBYTE* pixels, float* res) {
  int x,y,pos=0;
  for(x=0; x<tx; x++,pos++)
    res[pos] = pixels[pos+tx] - pixels[pos];
  for(y=1; y<ty-1; y++) {
    pos = y*tx;
    for(x=0; x<tx; x++,pos++)
      res[pos] = pixels[pos+tx] - pixels[pos-tx];
  }
  for(x=0; x<tx; x++,pos++)
    res[pos] = pixels[pos] - pixels[pos-tx];
}

/* compute original, unsmoothed, gradient
*/
void _compute_pure_gradient( UBYTE_image* img, float_layers* grad ) {
  ASSERT_SAME_SIZE(img,grad);
  assert(grad->tz==2);
  int tx = img->tx;
  int ty = img->ty;
  
  // compute horizontal gradient
  _diff_vert(tx,ty,img->pixels,grad->pixels);
  
  // compute vertical gradient
  _diff_horiz(tx,ty,img->pixels,grad->pixels+tx*ty);
}

/* compute horizontal smoothing with 3-sized mask
*/
template<typename TData>
void _smooth_3_horiz(int tx, int ty, const int w_center, const int w_side, TData* pixels, TData* _res, int n_thread) {
  int y;
  const int sum_w = 2*w_side + w_center;
  #if defined(USE_OPENMP)
  #pragma omp parallel for num_threads(n_thread)
  #endif
  for(y=0; y<ty; y++) {
    int x,pos = y*tx;
    TData* res = _res + pos;
    *res++ = ( (w_center+w_side)*pixels[0+pos] + w_side*pixels[1+pos])/sum_w;
    for(x=1; x<tx-1; x++)
      *res++ = (w_side*pixels[x+1+pos] + w_center*pixels[x+pos] + w_side*pixels[x-1+pos])/sum_w;
    *res++ = ( (w_center+w_side)*pixels[x+pos] + w_side*pixels[x-1+pos])/sum_w;
  }
}
void _smooth_121_horiz(int tx, int ty, UBYTE* pixels, UBYTE* res, int n_thread) {
  _smooth_3_horiz( tx, ty, 2, 1, pixels, res, n_thread );
}
template<typename TData>
void _smooth_5_horiz( int tx, int ty, const int w_center, const int w_side1, const int w_side2, 
                      TData* pixels, TData* _res, int n_thread) {
  int y;
  const int sum_w = 2*(w_side1 + w_side2) + w_center;
  #if defined(USE_OPENMP)
  #pragma omp parallel for num_threads(n_thread)
  #endif
  for(y=0; y<ty; y++) {
    int x,pos = y*tx;
    TData* res = _res + pos;
      x=0;
      *res++ = ( 
                w_side2 * pixels[x  +pos] +
                w_side1 * pixels[x  +pos] +
                w_center* pixels[x  +pos] +
                w_side1 * pixels[x+1+pos] +
                w_side2 * pixels[x+2+pos] ) / sum_w;
      x++;
      *res++ = ( 
                w_side2 * pixels[x-1+pos] +
                w_side1 * pixels[x-1+pos] +
                w_center* pixels[x  +pos] +
                w_side1 * pixels[x+1+pos] +
                w_side2 * pixels[x+2+pos] ) / sum_w;
    
    for(x=2; x<tx-2; x++)
      *res++ = ( 
                w_side2 * pixels[x-2+pos] +
                w_side1 * pixels[x-1+pos] +
                w_center* pixels[x  +pos] +
                w_side1 * pixels[x+1+pos] +
                w_side2 * pixels[x+2+pos] ) / sum_w;
    
      *res++ = ( 
                w_side2 * pixels[x-2+pos] +
                w_side1 * pixels[x-1+pos] +
                w_center* pixels[x  +pos] +
                w_side1 * pixels[x+1+pos] +
                w_side2 * pixels[x+1+pos] ) / sum_w;
      x++;
      *res++ = ( 
                w_side2 * pixels[x-2+pos] +
                w_side1 * pixels[x-1+pos] +
                w_center* pixels[x  +pos] +
                w_side1 * pixels[x  +pos] +
                w_side2 * pixels[x  +pos] ) / sum_w;
  }
}
template<typename TData>
void _smooth_7_horiz(int tx, int ty, const int w_center, const int w_side1, const int w_side2, const int w_side3, 
                    TData* pixels, TData* _res, int n_thread) {
  int y;
  const int sum_w = 2*(w_side1 + w_side2 + w_side3) + w_center;
  #if defined(USE_OPENMP)
  #pragma omp parallel for num_threads(n_thread)
  #endif
  for(y=0; y<ty; y++) {
    int x,pos = y*tx;
    TData* res = _res + pos;
      x=0;
      *res++ = ( 
                w_side3 * pixels[x  +pos] +
                w_side2 * pixels[x  +pos] +
                w_side1 * pixels[x  +pos] +
                w_center* pixels[x  +pos] +
                w_side1 * pixels[x+1+pos] +
                w_side2 * pixels[x+2+pos] +
                w_side3 * pixels[x+3+pos] ) / sum_w;
      x++;
      *res++ = ( 
                w_side3 * pixels[x-1+pos] +
                w_side2 * pixels[x-1+pos] +
                w_side1 * pixels[x-1+pos] +
                w_center* pixels[x  +pos] +
                w_side1 * pixels[x+1+pos] +
                w_side2 * pixels[x+2+pos] +
                w_side3 * pixels[x+3+pos] ) / sum_w;
      x++;
      *res++ = ( 
                w_side3 * pixels[x-2+pos] +
                w_side2 * pixels[x-2+pos] +
                w_side1 * pixels[x-1+pos] +
                w_center* pixels[x  +pos] +
                w_side1 * pixels[x+1+pos] +
                w_side2 * pixels[x+2+pos] +
                w_side3 * pixels[x+3+pos] ) / sum_w;
    
    for(x=3; x<tx-3; x++)
      *res++ = ( 
                w_side3 * pixels[x-3+pos] +
                w_side2 * pixels[x-2+pos] +
                w_side1 * pixels[x-1+pos] +
                w_center* pixels[x  +pos] +
                w_side1 * pixels[x+1+pos] +
                w_side2 * pixels[x+2+pos] +
                w_side3 * pixels[x+3+pos] ) / sum_w;
    
      *res++ = ( 
                w_side3 * pixels[x-3+pos] +
                w_side2 * pixels[x-2+pos] +
                w_side1 * pixels[x-1+pos] +
                w_center* pixels[x  +pos] +
                w_side1 * pixels[x+1+pos] +
                w_side2 * pixels[x+2+pos] +
                w_side3 * pixels[x+2+pos] ) / sum_w;
      x++;
      *res++ = ( 
                w_side3 * pixels[x-3+pos] +
                w_side2 * pixels[x-2+pos] +
                w_side1 * pixels[x-1+pos] +
                w_center* pixels[x  +pos] +
                w_side1 * pixels[x+1+pos] +
                w_side2 * pixels[x+1+pos] +
                w_side3 * pixels[x+1+pos] ) / sum_w;
      x++;
      *res++ = ( 
                w_side3 * pixels[x-3+pos] +
                w_side2 * pixels[x-2+pos] +
                w_side1 * pixels[x-1+pos] +
                w_center* pixels[x  +pos] +
                w_side1 * pixels[x  +pos] +
                w_side2 * pixels[x  +pos] +
                w_side3 * pixels[x  +pos] ) / sum_w;
  }
}


/* compute vertical smoothing with 3-sized mask
*/
template<typename TData>
void _smooth_3_vert(int tx, int ty, const int w_center, const int w_side, TData* pixels, TData* res, int n_thread) {
  int x,y,pos=0;
  const int sum_w = 2*w_side + w_center;
  for(x=0; x<tx; x++,pos++)
    res[pos] = (  (w_center+w_side)*pixels[pos] + w_side*pixels[pos+tx])/sum_w;
  #if defined(USE_OPENMP)
  #pragma omp parallel for num_threads(n_thread)
  #endif
  for(y=1; y<ty-1; y++) {
    int x,pos = y*tx;
    for(x=0; x<tx; x++,pos++)
      res[pos] = ( w_side*pixels[pos+tx] + w_center*pixels[pos] + w_side*pixels[pos-tx])/sum_w;
  }
  pos = (ty-1)*tx;
  for(x=0; x<tx; x++,pos++)
    res[pos] = ( (w_center+w_side)*pixels[pos] + w_side*pixels[pos-tx])/sum_w;
}
void _smooth_121_vert(int tx, int ty, UBYTE* pixels, UBYTE* res, int n_thread) {
  _smooth_3_vert( tx, ty, 2, 1, pixels, res, n_thread );
}
template<typename TData>
void _smooth_5_vert(int tx, int ty, const int w_center, const int w_side1, const int w_side2, 
                    TData* pixels, TData* res, int n_thread) {
  int x,y,pos=0;
  const int sum_w = 2*(w_side1 + w_side2) + w_center;
  const int tx1=tx,tx2=2*tx;
  for(x=0; x<tx; x++,pos++)
      res[pos] = ( 
                  w_side2 * pixels[pos] + 
                  w_side1 * pixels[pos] + 
                  w_center* pixels[pos] + 
                  w_side1 * pixels[pos+tx1] + 
                  w_side2 * pixels[pos+tx2] 
                  )/sum_w;
  for(x=0; x<tx; x++,pos++)
      res[pos] = ( 
                  w_side2 * pixels[pos-tx1] + 
                  w_side1 * pixels[pos-tx1] + 
                  w_center* pixels[pos] + 
                  w_side1 * pixels[pos+tx1] + 
                  w_side2 * pixels[pos+tx2] 
                  )/sum_w;
  
  #if defined(USE_OPENMP)
  #pragma omp parallel for num_threads(n_thread)
  #endif
  for(y=2; y<ty-2; y++) {
    int x,pos = y*tx;
    for(x=0; x<tx; x++,pos++)
      res[pos] = ( 
                  w_side2 * pixels[pos-tx2] + 
                  w_side1 * pixels[pos-tx1] + 
                  w_center* pixels[pos] + 
                  w_side1 * pixels[pos+tx1] + 
                  w_side2 * pixels[pos+tx2] 
                  )/sum_w;
  }
  pos = (ty-2)*tx;
  for(x=0; x<tx; x++,pos++)
      res[pos] = ( 
                  w_side2 * pixels[pos-tx2] + 
                  w_side1 * pixels[pos-tx1] + 
                  w_center* pixels[pos] + 
                  w_side1 * pixels[pos+tx1] + 
                  w_side2 * pixels[pos+tx1] 
                  )/sum_w;
  for(x=0; x<tx; x++,pos++)
      res[pos] = ( 
                  w_side2 * pixels[pos-tx2] + 
                  w_side1 * pixels[pos-tx1] + 
                  w_center* pixels[pos] + 
                  w_side1 * pixels[pos] + 
                  w_side2 * pixels[pos] 
                  )/sum_w;
}
template<typename TData>
void _smooth_7_vert(int tx, int ty, const int w_center, const int w_side1, const int w_side2, const int w_side3, 
                    TData* pixels, TData* res, int n_thread) {
  int x,y,pos=0;
  const int sum_w = 2*(w_side1 + w_side2 + w_side3) + w_center;
  const int tx1=tx,tx2=2*tx,tx3=3*tx;
  for(x=0; x<tx; x++,pos++)
      res[pos] = ( 
                  w_side3 * pixels[pos] + 
                  w_side2 * pixels[pos] + 
                  w_side1 * pixels[pos] + 
                  w_center* pixels[pos] + 
                  w_side1 * pixels[pos+tx1] + 
                  w_side2 * pixels[pos+tx2] + 
                  w_side3 * pixels[pos+tx3] 
                  )/sum_w;
  for(x=0; x<tx; x++,pos++)
      res[pos] = ( 
                  w_side3 * pixels[pos-tx1] + 
                  w_side2 * pixels[pos-tx1] + 
                  w_side1 * pixels[pos-tx1] + 
                  w_center* pixels[pos] + 
                  w_side1 * pixels[pos+tx1] + 
                  w_side2 * pixels[pos+tx2] + 
                  w_side3 * pixels[pos+tx3] 
                  )/sum_w;
  for(x=0; x<tx; x++,pos++)
      res[pos] = ( 
                  w_side3 * pixels[pos-tx2] + 
                  w_side2 * pixels[pos-tx2] + 
                  w_side1 * pixels[pos-tx1] + 
                  w_center* pixels[pos] + 
                  w_side1 * pixels[pos+tx1] + 
                  w_side2 * pixels[pos+tx2] + 
                  w_side3 * pixels[pos+tx3] 
                  )/sum_w;
  
  #if defined(USE_OPENMP)
  #pragma omp parallel for num_threads(n_thread)
  #endif
  for(y=3; y<ty-3; y++) {
    int x,pos = y*tx;
    for(x=0; x<tx; x++,pos++)
      res[pos] = ( 
                  w_side3 * pixels[pos-tx3] + 
                  w_side2 * pixels[pos-tx2] + 
                  w_side1 * pixels[pos-tx1] + 
                  w_center* pixels[pos] + 
                  w_side1 * pixels[pos+tx1] + 
                  w_side2 * pixels[pos+tx2] + 
                  w_side3 * pixels[pos+tx3] 
                  )/sum_w;
  }
  pos = (ty-3)*tx;
  for(x=0; x<tx; x++,pos++)
      res[pos] = ( 
                  w_side3 * pixels[pos-tx3] + 
                  w_side2 * pixels[pos-tx2] + 
                  w_side1 * pixels[pos-tx1] + 
                  w_center* pixels[pos] + 
                  w_side1 * pixels[pos+tx1] + 
                  w_side2 * pixels[pos+tx2] + 
                  w_side3 * pixels[pos+tx2] 
                  )/sum_w;
  for(x=0; x<tx; x++,pos++)
      res[pos] = ( 
                  w_side3 * pixels[pos-tx3] + 
                  w_side2 * pixels[pos-tx2] + 
                  w_side1 * pixels[pos-tx1] + 
                  w_center* pixels[pos] + 
                  w_side1 * pixels[pos+tx1] + 
                  w_side2 * pixels[pos+tx1] + 
                  w_side3 * pixels[pos+tx1] 
                  )/sum_w;
  for(x=0; x<tx; x++,pos++)
      res[pos] = ( 
                  w_side3 * pixels[pos-tx3] + 
                  w_side2 * pixels[pos-tx2] + 
                  w_side1 * pixels[pos-tx1] + 
                  w_center* pixels[pos] + 
                  w_side1 * pixels[pos] + 
                  w_side2 * pixels[pos] + 
                  w_side3 * pixels[pos] 
                  )/sum_w;
}

/* Smooth an image using a Gaussian filter.
*/
template<typename TData>
void _smooth_gaussian_alltype( const int tx, const int ty, TData* img, float _sigma, TData* res, int n_thread ) {
  const float MAX_SIGMA = 1.86f;
  
  TData* img2 = img;
  if(_sigma>MAX_SIGMA) {  // reallocate if more than one smoothing pass is required
    img2 = NEWA(TData,tx*ty);
    memcpy(img2,img,tx*ty*sizeof(TData));
  }
  TData* tmp = NEWA(TData,tx*ty);
  TData* old_res = res;
  
  float remaining = _sigma*_sigma;
  while( 1 ) {
    float sigma = MIN(MAX_SIGMA,sqrt(remaining));
    remaining -= sigma*sigma;
    
    // compute gaussian filter coefficients
    const int wcenter = 1000;
    const int wside1 = int(0.5 + wcenter*exp( -pow2(1./sigma)/2 ));
    const int wside2 = int(0.5 + wcenter*exp( -pow2(2./sigma)/2 ));
    const int wside3 = int(0.5 + wcenter*exp( -pow2(3./sigma)/2 ));
    const int wside4 = int(0.5 + wcenter*exp( -pow2(4./sigma)/2 ));
    assert( wside4 < wcenter/10 || !"error: smoothing is too large" );
    
    if ( wside2 < wcenter/10 ) {
      _smooth_3_horiz( tx, ty, wcenter, wside1, img2, tmp, n_thread );
      _smooth_3_vert(  tx, ty, wcenter, wside1, tmp, res, n_thread );
    } else if( wside3 < wcenter/10 ) {
      _smooth_5_horiz( tx, ty, wcenter, wside1, wside2, img2, tmp, n_thread );
      _smooth_5_vert(  tx, ty, wcenter, wside1, wside2, tmp, res, n_thread );
    } else {
      _smooth_7_horiz( tx, ty, wcenter, wside1, wside2, wside3, img2, tmp, n_thread );
      _smooth_7_vert(  tx, ty, wcenter, wside1, wside2, wside3, tmp, res, n_thread );
    }
    
    if(remaining < 0.001)
      break;
    else {
      TData* tmp3;
      tmp3 = img2;
      img2 = res;
      res = tmp3;
    }
  }
  
  if(res!=old_res) { // copy to true res
    memcpy(old_res,res,tx*ty*sizeof(TData));
    img2 = res;
  }
  if(_sigma>MAX_SIGMA) 
    free(img2);
  free(tmp);
}

void _smooth_gaussian( UBYTE_image* img, float _sigma, UBYTE_image* res, int n_thread ) {
  ASSERT_SAME_SIZE(img,res);
  _smooth_gaussian_alltype(img->tx,img->ty,img->pixels,_sigma,res->pixels,n_thread);
}


/* compute gradient smoothed with Sobel mask
*/
void _compute_sobel_gradient( UBYTE_image* img, float_layers* grad, int n_thread ) {
  ASSERT_SAME_SIZE(img,grad);
  assert(grad->tz==2);
  int tx = img->tx;
  int ty = img->ty;
  UBYTE* tmp = NEWA(UBYTE,tx*ty);
  
  // compute horizontal gradient
  _smooth_121_horiz(tx,ty,img->pixels,tmp, n_thread);
  _diff_vert(tx,ty,tmp,grad->pixels);
  
  // compute vertical gradient
  _smooth_121_vert(tx,ty,img->pixels,tmp, n_thread);
  _diff_horiz(tx,ty,tmp,grad->pixels+tx*ty);
  
  // free everything
  free(tmp);
}

/* Compute the dx,dy gradient on the image based on a [-1,0,1] mask. 
     =0 :  no prior smoothing 
     =1 :  sobel smoothing 
*/
void _compute_grad_101( UBYTE_image* img, int method, float_layers* grad, int n_thread ) {
  ASSERT_SAME_SIZE(img,grad);
  assert(grad->tz==2);
  
  // compute gradient
  if( method == 0 )
    _compute_pure_gradient(img, grad);
  else if( method == 1 )
    _compute_sobel_gradient(img, grad, n_thread);
  else
    assert(!"error: unknown method for compute_grad_101");
}


/* Compute the Histogram of oriented gradient for each pixel.
   Number of orientations is determined by hog->tz;
   method determines orientation bining: 
     =0 : atan + linear interpolation
     =1 : fast cos projection
*/
void _compute_hog( float_layers* grad, int method, float_layers* hog, int n_thread ) {
  ASSERT_SAME_SIZE(grad,hog);
  const int n_ori = hog->tz;
  const int npix = hog->tx*hog->ty;
  
  const float* dx = grad->pixels; 
  const float* dy = grad->pixels + npix; 
  
  if( method == 0 ) {
    // use atan
    memset(hog->pixels,0,n_ori*npix*sizeof(float));
    int i;
    for(i=0; i<npix; i++) {
      float norm = sqrt(dy[i]*dy[i] + dx[i]*dx[i]);
      float angle = atan2(dy[i],dx[i]);  // angle in [-pi,pi]
      
      float b_angle = (angle + M_PI)/n_ori;
      int q_angle = int(0.5 + b_angle); // first bin
      float coef = b_angle-q_angle;
      q_angle = (q_angle + 3*n_ori/2) % n_ori;
      
      hog->pixels[ ((q_angle  )      )*npix + i ] += (1-coef)*norm;
      hog->pixels[ ((q_angle+1)%n_ori)*npix + i ] += (  coef)*norm;
    }
  } else if (method == 1 ) {
    int l;
    #if defined(USE_OPENMP)
    #pragma omp parallel for num_threads(n_thread)
    #endif
    for(l=0; l<n_ori; l++) {
      float angle = -2*(l-2)*M_PI/n_ori;
      float kos = cos( angle );
      float zin = sin( angle );
      float* layer_l = hog->pixels + l*npix;
      int i;
      for(i=0; i<npix; i++) {
        float value = kos*dx[i] + zin*dy[i];
        layer_l[i] = (value > 0 ) ? value : 0;
      }
    }
  } else 
    assert(!"error: unknown method for compute_hog");
}


/* compute 8 directions of gradient per pixels
   using 4 oriented filters extremely simple like [-1,1]
*/
void _compute_hog_8_direct( UBYTE_image* image, float_layers* hog_out, int n_thread ) {
  ASSERT_SAME_SIZE(image,hog_out);
  assert(hog_out->tz==8);
  int j,tx=image->tx, ty=image->ty;
  int npix=tx*image->ty;
  
  // init output
  memset(hog_out->pixels,0,8*npix*sizeof(float));
  
  // compute horizontal filter
  #if defined(USE_OPENMP)
  #pragma omp parallel for num_threads(n_thread)
  #endif
  for(j=0; j<ty; j++) {
    UBYTE* img = image->pixels + j*tx;
    UBYTE* lastimg = img + tx-1;
    float* hog0f = hog_out->pixels + 0*npix + j*tx; // first
    float* hog0l  = hog0f+1;  // last
    float* hog1f = hog_out->pixels + 4*npix + j*tx; // first
    float* hog1l = hog1f+1; // last
    
    for(; img<lastimg; img++) {
      float diff = img[1] - img[0];
      float pos,neg;
      if( diff < 0 ) {
        pos = 0;
        neg = -diff/2.f;
      } else {
        neg = 0;
        pos = diff/2.f;
      }
      *hog0f++ += neg;
      *hog1f++ += pos;
      *hog0l++ += neg;
      *hog1l++ += pos;
    }
  }
  
  // compute veritical filter
  #if defined(USE_OPENMP)
  #pragma omp parallel for num_threads(n_thread)
  #endif
  for(j=0; j<ty-1; j++) {
    UBYTE* img = image->pixels + j*tx;
    UBYTE* lastimg = img + tx;
    const int offset = tx;
    UBYTE* img2 = img + offset;
    float* hog0f = hog_out->pixels + 2*npix + j*tx; // first
    float* hog0l  = hog0f + offset;  // last
    float* hog1f = hog_out->pixels + 6*npix + j*tx; // first
    float* hog1l = hog1f + offset;  // last
    
    while(img<lastimg) {
      float diff = (*img2++) - (*img++);
      float pos,neg;
      if( diff < 0 ) {
        pos = 0;
        neg = -diff/2.f;
      } else {
        neg = 0;
        pos = diff/2.f;
      }
      *hog0f++ += neg;
      *hog1f++ += pos;
      *hog0l++ += neg;
      *hog1l++ += pos;
    }
  }
  
  const float div_diag = 2*1.2666f; // learned
  
  // compute diagonal filter 1
  #if defined(USE_OPENMP)
  #pragma omp parallel for num_threads(n_thread)
  #endif
  for(j=0; j<ty-1; j++) {
    UBYTE* img = image->pixels + j*tx;
    UBYTE* lastimg = img + tx-1;
    const int offset = 1+tx;
    UBYTE* img2 = img + offset;
    float* hog0f = hog_out->pixels + 1*npix + j*tx; // first
    float* hog0l  = hog0f + offset;  // last
    float* hog1f = hog_out->pixels + 5*npix + j*tx; // first
    float* hog1l = hog1f + offset;  // last
    
    while(img<lastimg) {
      float diff = (*img2++) - (*img++);
      float pos,neg;
      if( diff < 0 ) {
        pos = 0;
        neg = -diff/div_diag;
      } else {
        neg = 0;
        pos = diff/div_diag;
      }
      *hog0f++ += neg;
      *hog1f++ += pos;
      *hog0l++ += neg;
      *hog1l++ += pos;
    }
  }
  
  // compute diagonal filter 2
  #if defined(USE_OPENMP)
  #pragma omp parallel for num_threads(n_thread)
  #endif
  for(j=1; j<ty; j++) {
    UBYTE* img = image->pixels + j*tx;
    UBYTE* lastimg = img + tx-1;
    const int offset = 1-tx;
    UBYTE* img2 = img + offset;
    float* hog0f = hog_out->pixels + 7*npix + j*tx; // first
    float* hog0l  = hog0f + offset;  // last
    float* hog1f = hog_out->pixels + 3*npix + j*tx; // first
    float* hog1l = hog1f + offset;  // last
    
    while(img<lastimg) {
      float diff = (*img2++) - (*img++);
      float pos,neg;
      if( diff < 0 ) {
        pos = 0;
        neg = -diff/div_diag;
      } else {
        neg = 0;
        pos = diff/div_diag;
      }
      *hog0f++ += neg;
      *hog1f++ += pos;
      *hog0l++ += neg;
      *hog1l++ += pos;
    }
  }
}


/* Post-processing of the HOG: cross-orientation inhibition.
   for one pixel i and orientation o: hog[i,o] = max(0, hog[i,o] - coef*hog[i,:].mean())
*/
void subtract_mean_ori( float_layers* hog, float coef, int n_thread ) {
  const int npix = hog->tx*hog->ty;
  int l;
  float* sum = NEWAC(float, npix);
  float* max = NEWAC(float, npix);
  
  // compute mean per pixel
  #if defined(USE_OPENMP)
  #pragma omp parallel for num_threads(n_thread)
  #endif
  for(l=0; l<hog->tz; l++) {
    float* p = sum;
    float* m = max;
    float* hog_pix = hog->pixels + l*npix;
    int i;
    for(i=0; i<npix; i++,m++) {
      float v = *hog_pix++;
      *p++ += v;
      float max = *m;
      if(v>max)  *m=v;
    }
  }
  
  // subtract coef*mean
  coef /= hog->tz;
  #if defined(USE_OPENMP)
  #pragma omp parallel for num_threads(n_thread)
  #endif
  for(l=0; l<hog->tz; l++) {
    float* p = sum;
    float* m = max;
    float* hog_pix = hog->pixels + l*npix;
    int i;
    for(i=0; i<npix; i++) {
      float Max = *m++; // max
      float mean =  coef * (*p++);  // == mean * coef
      if( mean >= Max ) 
        *hog_pix = 0;
      else {
        *hog_pix = Max*(1 - (Max - (*hog_pix))/(Max - mean + 1e-8f));
        if(*hog_pix<0)  *hog_pix = 0;
      }
      hog_pix++;
    }
  }
  
  free(sum);
  free(max);
}


/* Pass the gradient image through a sigmoid
*/
void sigmoid_array( float_array* img, float coef, float offset, int n_thread ) {
  assert(coef>0);
  const int npix=img->tx;
//  float* p = img->pixels;
//  for(i=0; i<npix; i++) {
//    float v = *p;
//    *p++ = 2.f/(1.f + exp(-coef*v + offset)) - 1.f;
//  }
  int l;
  
  // optimization: precompute some values of sigmoid
  //  2/(1 + exp(-arange(0,8,0.5)+offset)) -1
  const int npc = 64;
  float precom[npc+1];
  for(l=0; l<=npc; l++) precom[l]= 1.f/(1.f + exp(-l/8.f + offset));
  for(l=1; l<=npc; l++) precom[l] = (precom[l]-precom[0]) / (1 - precom[0]);  // renorm between 0 and 1
  precom[0] = 0;
  const float maxindex = npc - 0.001;
  
  #define NSUB 32
  #if defined(USE_OPENMP)
  #pragma omp parallel for num_threads(n_thread)
  #endif
  for(l=0; l<NSUB; l++) {
    int start = (l*npix)/NSUB;
    int end = (l+1)*npix/NSUB;
    int npixsub = end-start;
    float* p = img->pixels + start;
    int i;
    for(i=0; i<npixsub; i++) {
      float v = 8*(coef*(*p));
      if(v>maxindex)  v=maxindex;
      int n = int(v);
      float w = v-n;
      *p++ = (1-w)*precom[n] + w*precom[n+1];
    }
  }
}


/* Compute a spatially smoothed version of the HOG.
*/
void smooth_hog_gaussian( float_layers* hog, float smoothing, int n_thread ) {
  int l;
  const int npix = hog->tx*hog->ty;
  for(l=0; l<hog->tz; l++)
    _smooth_gaussian_alltype(hog->tx,hog->ty,hog->pixels+l*npix,smoothing,hog->pixels+l*npix, n_thread);
}
































