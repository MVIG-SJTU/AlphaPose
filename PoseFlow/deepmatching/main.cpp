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
#include "std.h"
#include "image.h"
#include "io.h"
#include "deep_matching.h"
#include "main.h"
#include <thread>

void usage(const int language)
{
  #define p(msg)  std_printf(msg "\n");
  p("usage:");
  switch(language){
    case EXE_OPTIONS:
      p("./deepmatching image1 image2 [options]");
      p("Compute the 'DeepMatching' between two images and print a list of")
      p("pair-wise point correspondences:")
      p("  x1 y1 x2 y2 score index ...")
      p("(index refers to the local maximum from which the match was retrieved)")
      p("Images must be in PPM, PNG or JPG format. Version 1.2.2")
      break;
    case MATLAB_OPTIONS:
      p("matches = deepmatching(image1, image2 [, options])")
      p("Compute the 'DeepMatching' between two images.")
      p("Images must be HxWx3 single matrices.")
      p("Options is an optional string argument ('' by default).")
      p("The function returns a matrix with 6 columns, each row being x1 y1 x2 y2 score index.")
      p("(index refers to the local maximum from which the match was retrieved)")
      p("Version 1.2.2")
      break;
    case PYTHON_OPTIONS:
      p("matches = deepmatching.deepmatching(image1, image2, options='')")
      p("Compute the 'DeepMatching' between two images.")
      p("Images must be HxWx3 numpy arrays (converted to float32).")
      p("Options is an optional string argument ('' by default).")
      p("The function returns a numpy array with 6 columns, each row being x1 y1 x2 y2 score index.")
      p("(index refers to the local maximum from which the match was retrieved)")
      p("Version 1.2.2")
      break;
  }
  p("")
  p("Options:")
  p("    -h, --help                 print this message")
//p("  HOG parameters (low-level pixel descriptor):")
//p("    -png_settings              (auto) recommended for uncompressed images")
//p("    -jpg_settings              (auto) recommended for compressed images")
//p("   in more details: (for fine-tuning)")
//p("    -hog.presm <f=1.0>         prior image smoothing")
//p("    -hog.midsm <f=1.0>         intermediate HOG smoothing")
//p("    -hog.sig <f=0.2>           sigmoid strength")
//p("    -hog.postsm <f=1.0>        final HOG-smoothing")
//p("    -hog.ninth <f=0.3>         robustness to pixel noise (eg. JPEG artifacts)")
  p("")
  p("  Matching parameters:")
//p("    -iccv_settings             settings used for the ICCV paper")
//p("    -improved_settings         (default) supposedly improved settings")
//p("   in more details: (for fine-tuning)")
  p("    -downscale/-R <n=1>        downsize the input images by a factor 2^n")
//p("    -overlap <n=999>           use overlapping patches in image1 from level n")
//p("    -subref <n=0>              0: denser sampling or 1: not of image1 patches")
  p("    -ngh_rad <n=0>             if n>0: restrict matching to n pxl neighborhood")
  p("    -nlpow <f=1.4>             non-linear rectification x := x^f")
//p("    -maxima_mode <n=0>         0: from all top cells / 1: from local maxima")
//p("    -min_level <n=2>           skip maxima in levels [0, 1, ..., n-1]")
  p("    -mem <n=1>                 if n>0: optimize memory footprint (bit unstable)")
//p("    -scoring_mode <n=1>        type of correspondence scoring mode (0/1)")
  p("")
  p("  Fully scale & rotation invariant DeepMatching:")
  p("    if either one of these options is used, then this mode is activated:")
  p("    -max_scale <factor=5>         max scaling factor")
  p("    -rot_range <from=0> <to=360>  rotation range")
  p("")
  p("  Other parameters:")
  p("    -resize <width> <height>   to resize input images beforehand")
  p("    -v                         increase verbosity")
  p("    -nt <n>                    multi-threading with <n> threads")
  if(language==EXE_OPTIONS) {
  p("    -out <file_name>           output correspondences in a file")
  exit(1);}
}

bool endswith(const char *str, const char *suffix)
{
    if(!str || !suffix)  return false;
    size_t lenstr = strlen(str);
    size_t lensuffix = strlen(suffix);
    if(lensuffix >  lenstr)  return false;
    return strncmp(str + lenstr - lensuffix, suffix, lensuffix) == 0;
}

image_t* rescale_image( image_t* im, int width, int height ) 
{
  image_t* res = image_new(width,height);
  image_resize_bilinear_newsize(res, im, width, height);
  image_delete(im);
  return res;
}


const char *parse_options(dm_params_t *params, scalerot_params_t *sr_params, bool *use_scalerot, float *fx, float *fy, const int argc, const char **argv, const int language, image_t **im1, image_t **im2) {
  int current_arg = 0;
  const char* out_filename = NULL;
  
  // parse options
  while(current_arg < argc)
  {
    const char* a = argv[current_arg++];
    #define isarg(key)  !strcmp(a,key)
    
    if(isarg("-h") || isarg("--help") )    usage(language);
  // HOG and patch parameters
  //else if(isarg("-hog.presm"))
  //  params->desc_params.presmooth_sigma = atof(argv[current_arg++]);
  //else if(isarg("-hog.sig"))
  //  params->desc_params.hog_sigmoid = atof(argv[current_arg++]);
  //else if(isarg("-hog.midsm"))
  //  params->desc_params.mid_smoothing = atof(argv[current_arg++]);
  //else if(isarg("-hog.postsm"))
  //  params->desc_params.post_smoothing = atof(argv[current_arg++]);
  //else if(isarg("-hog.ninth"))
  //  params->desc_params.ninth_dim = atof(argv[current_arg++]);
  //else if(isarg("-hog.nrmpix"))
  //  params->desc_params.norm_pixels = atof(argv[current_arg++]);
    else if(isarg("-png_settings")) { 
      params->desc_params.presmooth_sigma = 0; // no image smoothing since the image is uncompressed
      params->desc_params.hog_sigmoid = 0.2;
      params->desc_params.mid_smoothing = 1.5;
      params->desc_params.post_smoothing = 1;
      params->desc_params.ninth_dim = 0.1; } // low ninth_dim since image PSNR is high
    else if(isarg("-jpg_settings")) {
      params->desc_params.presmooth_sigma = 1; // smooth the image to remove jpg artifacts
      params->desc_params.hog_sigmoid = 0.2;
      params->desc_params.mid_smoothing = 1.5;
      params->desc_params.post_smoothing = 1;
      params->desc_params.ninth_dim = 0.3; } // higher ninth_dim because of pixel noise
  // matching parameters
    else if(isarg("-R") || isarg("-downscale"))
      params->prior_img_downscale = atoi(argv[current_arg++]);
  //else if(isarg("-overlap"))
  //  params->overlap = atoi(argv[current_arg++]);
  //else if(isarg("-subref"))
  //  params->subsample_ref = atoi(argv[current_arg++]);
    else if(isarg("-nlpow"))
      params->nlpow = atof(argv[current_arg++]);
    else if(isarg("-ngh_rad"))
      params->ngh_rad = atoi(argv[current_arg++]);
  // maxima parameters
  //else if(isarg("-maxima_mode"))
  //  params->maxima_mode = atoi(argv[current_arg++]);
    else if(isarg("-mem")) {
      params->low_mem = atoi(argv[current_arg++]); }
    //else if(isarg("-min_level"))
    //  params->min_level = atoi(argv[current_arg++]);
  //else if(isarg("-scoring_mode"))
  //  params->scoring_mode = atoi(argv[current_arg++]);
    //else if(isarg("-iccv_settings")) {
    //  params->prior_img_downscale = 2;
    //  params->overlap = 0; // overlap from level 0
    //  params->subsample_ref = 1;
    //  params->nlpow = 1.6;
    //  params->maxima_mode = 1;
    //  params->low_mem = 0;
    //  params->min_level = 2;
    //  params->scoring_mode = 0; }
    //else if(isarg("-improved_settings")) {
    //  params->prior_img_downscale = 1; // less down-scale
    //  params->overlap = 999; // no overlap
    //  params->subsample_ref = 0; // dense patch sampling at every level in first image
    //  params->nlpow = 1.4;
    //  params->maxima_mode = 0;
    //  params->low_mem = 1;
    //  params->min_level = 2;
    //  params->scoring_mode = 1; } // improved scoring
    //else if(isarg("-max_psize")) {
    //  params->max_psize = atoi(argv[current_arg++]); }
  // scale & rot invariant version
    else if(isarg("-scale") || isarg("-max_scale")) {
      *use_scalerot = true;
      float scale = atof(argv[current_arg++]);
      sr_params->max_sc0 = sr_params->max_sc1 = int(1 + 2*log2(scale)); }
    else if(isarg("-rot") || isarg("-rot_range")) {
      *use_scalerot = true;
      int min_rot = atoi(argv[current_arg++]);
      int max_rot = atoi(argv[current_arg++]);
      while( min_rot < 0 ) {
        min_rot += 360;
        max_rot += 360;
      }
      sr_params->min_rot = int(floor(0.5 + min_rot/45.));
      sr_params->max_rot = int(floor(1.5 + max_rot/45.));
      while( sr_params->max_rot - sr_params->min_rot > 8 )  
        sr_params->max_rot--;
      assert( sr_params->min_rot < sr_params->max_rot ); }
  // other parameters
    else if(isarg("-resize")) {
      assert((*im1)->width==(*im2)->width && (*im1)->height==(*im2)->height);
      int width = atoi(argv[current_arg++]);
      int height = atoi(argv[current_arg++]);
      *fx *= (*im1)->width / float(width);
      *fy *= (*im1)->height / float(height);
      *im1 = rescale_image(*im1, width, height);
      *im2 = rescale_image(*im2, width, height); }
    else if(isarg("-v"))
      params->verbose++;
    else if(isarg("-nt")) {
      params->n_thread = atoi(argv[current_arg++]);
      if (params->n_thread==0)
        params->n_thread = std::thread::hardware_concurrency(); }
    else if(language == EXE_OPTIONS && isarg("-out"))
      out_filename = argv[current_arg++];
    else {
      err_printf("error: unexpected parameter '%s'", a);
      exit(-1);
    }
  }
  
  if( *use_scalerot )
    assert( params->ngh_rad == 0 || !"max trans cannot be used in full scale and rotation mode");
  else
    if( params->subsample_ref && (!ispowerof2((*im1)->width) || !ispowerof2((*im1)->height)) ) {
      err_printf("WARNING: first image has dimension which are not power-of-2\n");
      err_printf("For improved results, you should consider resizing the images with '-resize <w> <h>'\n");
    }
  
  return out_filename;
}


int main(int argc, const char ** argv)
{
  if( argc<=2 || !strcmp(argv[1],"-h") || !strcmp(argv[1],"--help") )  usage(EXE_OPTIONS); 
  
  int current_arg = 3;
  image_t *im1=NULL, *im2=NULL;
  {
    color_image_t *cim1 = color_image_load(argv[1]);
    color_image_t *cim2 = color_image_load(argv[2]);
    
    // Following deactivated because quite useless/dangerous in practice
    // default behavior == always using -jpg_settings
    
    //if( endswith(argv[1],"png") || endswith(argv[1],"PNG") )
    //  argv[--current_arg] = "-png_settings";  // set default
    //if( endswith(argv[1],"ppm") || endswith(argv[1],"PPM") )
    //  argv[--current_arg] = "-png_settings";  // set default
    //if( endswith(argv[1],"jpg") || endswith(argv[1],"JPG") )
    //  argv[--current_arg] = "-jpg_settings";  // set default
    //if( endswith(argv[1],"jpeg") || endswith(argv[1],"JPEG") )
    //  argv[--current_arg] = "-jpg_settings";  // set default
    
    im1 = image_gray_from_color(cim1);
    im2 = image_gray_from_color(cim2);
    color_image_delete(cim1);
    color_image_delete(cim2);
  }
  
  // set params to default
  dm_params_t params;
  set_default_dm_params(&params);
  scalerot_params_t sr_params;
  set_default_scalerot_params(&sr_params);
  bool use_scalerot = false;
  float fx=1, fy=1;
  
  // parse options
  const char* out_filename = parse_options(&params, &sr_params, &use_scalerot, &fx, &fy, argc-current_arg, 
                                           &argv[current_arg], EXE_OPTIONS, &im1, &im2);
  
  // compute deep matching
  float_image* corres = use_scalerot ? 
         deep_matching_scale_rot( im1, im2, &params, &sr_params ) : 
         deep_matching          ( im1, im2, &params, NULL );  // standard call
  
  // save result
  output_correspondences( out_filename, (corres_t*)corres->pixels, corres->ty, fx, fy );
  
  free_image(corres);
  image_delete(im1);
  image_delete(im2);
  return 0;
}































