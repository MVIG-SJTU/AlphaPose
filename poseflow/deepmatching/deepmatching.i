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
%module(docstring="Module to compute DeepMatching") deepmatching

%{
    #define SWIG_FILE_WITH_INIT

    #include <numpy/arrayobject.h>

    
    #define CHECK_NUMPY_ARRAY(a, expected_npy)                                 \
      if(!a) {                                                              \
        fprintf(stderr,"error in %s(): NULL input\n",__PRETTY_FUNCTION__);             \
        return NULL;                                                        \
      }                                                                     \
      if(!PyArray_Check(a)) {                                               \
        fprintf(stderr,"error in %s(): input not numpy array\n",__PRETTY_FUNCTION__);  \
        return NULL;                                                        \
      }                                                                     \
      if(!PyArray_ISCONTIGUOUS(a)) {                                        \
        fprintf(stderr,"error in %s(): array is not C-contiguous\n",__PRETTY_FUNCTION__);  \
        return NULL;                                                        \
      }                                                                     \
      if(PyArray_TYPE(a)!=expected_npy) {                                   \
        fprintf(stderr,"error in %s(): input has bad type (type id %d != " #expected_npy " %d)\n",__PRETTY_FUNCTION__, \
                PyArray_TYPE(a),expected_npy);                              \
        return NULL;                                                        \
      }

%}

%init %{
    import_array();
%}


%{
    #include "image.h"
    #include "array_types.h"
%}

%typemap(in) 
  (color_image_t* cimg) 
  (color_image_t cimage) {
  
  PyObject* a = $input;
  if(a==Py_None) {
    $1 = NULL;
  } else {
    CHECK_NUMPY_ARRAY(a, NPY_FLOAT)
    cimage.c1 = (float*) PyArray_DATA(a);
    a = PyObject_GetAttrString($input,"shape");
    assert(PyTuple_Size(a)==3);
    assert( PyInt_AsLong(PyTuple_GetItem(a,0)) == 3);
    cimage.height =  PyInt_AsLong(PyTuple_GetItem(a,1));
    cimage.width =  PyInt_AsLong(PyTuple_GetItem(a,2));
    cimage.c2 = cimage.c1 + cimage.width*cimage.height; 
    cimage.c3 = cimage.c2 + cimage.width*cimage.height; 
    $1=&cimage;
  }
}
%apply (color_image_t* cimg) {(color_image_t* )};

%typemap(out) float_image* corres {
  PyObject *o;
  npy_intp n_elem[2] = {$1->ty, $1->tx};
  o = PyArray_SimpleNewFromData(2,n_elem,NPY_FLOAT,$1->pixels);
  PyArray_FLAGS(o) |= NPY_OWNDATA;
  
  // append to current function result as a tuple
  $result = o;

}
%apply (float_image* corres) {(float_image* )};

float_image* deepmatching_numpy( color_image_t* cim1, color_image_t* cim2, char *options);

void usage_python();

%{
    #include "deep_matching.h"
    #include "io.h"
    #include "main.h"
    #include <string.h>

    static inline bool ispowerof2( long n ) {
      return (n & (n-1))==0;
    }
    
    float_image* deepmatching_numpy( color_image_t* cim1, color_image_t* cim2, char *options){
        // convert images to gray
        image_t *im1=image_gray_from_color(cim1), *im2=image_gray_from_color(cim2);
        
        // set params to default
        dm_params_t params;
        set_default_dm_params(&params);
        scalerot_params_t sr_params;
        set_default_scalerot_params(&sr_params);
        bool use_scalerot = false;
        float fx=1, fy=1;
        
        // read options
        if( options!=NULL ){
          int argc=0;
          const char* argv[256];
          argv[argc] = strtok(options," ");
          while(argv[argc]!=NULL)
            argv[++argc] = strtok(NULL," ");
          
          parse_options(&params, &sr_params, &use_scalerot, &fx, &fy, argc, argv, PYTHON_OPTIONS, &im1, &im2);
        }
        

        if( use_scalerot )
            assert( params.ngh_rad == 0 || !"max trans cannot be used in full scale and rotation mode");
        else
            if( params.subsample_ref && (!ispowerof2(im1->width) || !ispowerof2(im1->height)) ) {
                fprintf(stderr, "WARNING: first image has dimension which are not power-of-2\n");
                fprintf(stderr, "For improved results, you should consider resizing the images with '-resize <w> <h>'\n");
            }    

        // compute deep matching
        float_image* corres = use_scalerot ? 
                 deep_matching_scale_rot( im1, im2, &params, &sr_params ) : 
                 deep_matching          ( im1, im2, &params, NULL );  // standard call
                 
        image_delete(im1); image_delete(im2);
        return corres;
    }
    
    void usage_python() {
      usage(PYTHON_OPTIONS);
    }

%}


%pythoncode %{  
    from numpy import float32, rollaxis, ascontiguousarray
    def deepmatching( im1=None, im2=None, options=""):
        """
        matches = deepmatching.deepmatching(image1, image2, options='')
        Compute the 'DeepMatching' between two images.
        Images must be HxWx3 numpy arrays (converted to float32).
        Options is an optional string argument ('' by default), to set the options.
        The function returns a numpy array with 6 columns, each row being x1 y1 x2 y2 score index.
         (index refers to the local maximum from which the match was retrieved)
        Version 1.2"""
        if None in (im1,im2):
          usage_python()
          return
        
        # convert images
        if im1.dtype != float32:
            im1 = im1.astype(float32)
        if im2.dtype != float32:
            im2 = im2.astype(float32)
        assert len(im1.shape)==3 and len(im2.shape)==3, "images must have 3 dimensions"
        h, w, nchannels = im1.shape
        assert nchannels==3, "images must have 3 channels"
        im1 = ascontiguousarray(rollaxis(im1,2))
        im2 = ascontiguousarray(rollaxis(im2,2))
        corres = deepmatching_numpy( im1, im2, options)
        return corres
%} 







