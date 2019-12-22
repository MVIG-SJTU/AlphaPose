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
#include <mex.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

#include <stdio.h>
#include <stdarg.h>

void std_printf(const char* format, ... ) {
  va_list arglist;
  va_start( arglist, format );
  char buffer[1024];
  vsprintf( buffer, format, arglist );
  va_end(arglist);
  
  mexPrintf(buffer);
}

void err_printf(const char* format, ... ) {
  va_list arglist;
  va_start( arglist, format );
  char buffer[1024];
  vsprintf( buffer, format, arglist );
  va_end(arglist);
  
  mexErrMsgTxt(buffer);
}


#include "image.h"
#include "deep_matching.h"
#include "io.h"
#include "main.h"


static inline bool ispowerof2( long n ) {
  return (n & (n-1))==0;
}

color_image_t *input3darray_to_color_image(const mxArray *p){
    const int *dims = mxGetDimensions(p);
    const int h = dims[0], w = dims[1];
    assert( dims[2]==3 );
    float *in = (float*) mxGetData(p);
    color_image_t *out = color_image_new(w, h);
    for(int c=0 ; c<3 ; c++){
        float *inptr = in + c*w*h;
        float *outptr = out->c1 + c*w*h;
        for( int j=0 ; j<h ; j++){
            for( int i=0 ; i<w ; i++){
                outptr[j*w+i] = inptr[i*h+j];
            }
        }
    }
    return out;
}

void corres_to_output(float_image *corres, mxArray *p){
    const int h = corres->ty, w = corres->tx;
    float *data = (float*) mxGetData(p);
    for( int j=0 ; j<h ; j++) {
        for( int i=0 ; i<w ; i++) {
            data[i*h+j] = corres->pixels[j*w+i];
        }
    }    
}

void mexFunction( int nl, mxArray *pl[], int nr, const mxArray *pr[] ) {
    
    if( nr==0 ) {
        usage(MATLAB_OPTIONS);
        return;
    }
    
    if ( nl != 1){
        usage(MATLAB_OPTIONS);
        mexErrMsgTxt("error: returns one output");
    }
    if( nr < 2 || nr > 3){
        usage(MATLAB_OPTIONS);
        mexErrMsgTxt("error: takes two to four inputs");
    }
    
    // The code is originally written for C-order arrays.
    // We thus transpose all arrays in this mex-function which is not efficient...
    
    const int *pDims;
    if(mxGetNumberOfDimensions(pr[0]) != 3) mexErrMsgTxt("input images must have 3 dimensions");
    if(!mxIsClass(pr[0], "single")) mexErrMsgTxt("input images must be single");
    pDims = mxGetDimensions(pr[0]);
    if( pDims[2]!=3 ) mexErrMsgTxt("input images must have 3 channels");
    const int h = pDims[0], w = pDims[1];
    color_image_t *cim1 = input3darray_to_color_image( pr[0] );
    
    if(mxGetNumberOfDimensions(pr[1]) != 3) mexErrMsgTxt("input images must have 3 dimensions");
    if(!mxIsClass(pr[1], "single")) mexErrMsgTxt("input images must be single");
    pDims = mxGetDimensions(pr[1]);
    if( pDims[2]!=3) mexErrMsgTxt("input images must have 3 channels");
    color_image_t *cim2 = input3darray_to_color_image( pr[1] );
    
    // convert images to gray
    image_t *im1=image_gray_from_color(cim1), *im2=image_gray_from_color(cim2);;
    color_image_delete(cim1);
    color_image_delete(cim2);
    
    // set params to default
    dm_params_t params;
    set_default_dm_params(&params);
    scalerot_params_t sr_params;
    set_default_scalerot_params(&sr_params);
    bool use_scalerot = false;
    float fx=1, fy=1;

    // read options
    if( nr == 3 ){
      char *options = mxArrayToString(pr[2]);
      if( !options )  mexErrMsgTxt("Third parameter must be a string");
      int argc=0;
      const char* argv[256];
      argv[argc] = strtok(options," ");
      while(argv[argc]!=NULL)
        argv[++argc] = strtok(NULL," ");
      
      parse_options(&params, &sr_params, &use_scalerot, &fx, &fy, argc, argv, MATLAB_OPTIONS, &im1, &im2);
    }
    
    if( use_scalerot )
        assert( params.ngh_rad == 0 || !"max trans cannot be used in full scale and rotation mode");
    else
        if( params.subsample_ref && (!ispowerof2(im1->width) || !ispowerof2(im1->height)) ) {
            std_printf("WARNING: first image has dimension which are not power-of-2\n");
            std_printf("For improved results, you should consider resizing the images with '-resize <w> <h>'\n");
        }    

    // compute deep matching
    float_image* corres = use_scalerot ? 
             deep_matching_scale_rot( im1, im2, &params, &sr_params ) : 
             deep_matching          ( im1, im2, &params, NULL );  // standard call
   
    // output
    pl[0] = mxCreateNumericMatrix(corres->ty, corres->tx, mxSINGLE_CLASS, mxREAL);
    corres_to_output(corres, pl[0]);

    image_delete(im1); 
    image_delete(im2);
    free_image(corres);
    return;
}
