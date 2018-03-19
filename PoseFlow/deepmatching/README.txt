Implementation of the Deep Matching algorithm, published at ICCV 2013 in
"DeepFlow: Large displacement optical flow with deep matching" by Philippe 
Weinzaepfel, Jerome Revaud, Zaid Harchaoui and Cordelia Schmid.
Code and idea by Jerome Revaud, INRIA. The code is only for scientific 
or personnal use. Please contact me/INRIA for commercial use.
Email: jerome.revaud@inria.fr

Copyright (C) 2015 Jerome Revaud

Version 1.2.2

License:

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


Installation:
  
  make clean all
  
  This program has been built on a fedora18 x64 machine and tested on Mac OS X. 
  *No assistance* will be given to compile the code on other OS. However, if 
  you are able to sucessfully adapt the code for other platforms (Windows), 
  please notify me so that I can release these versions on the webpage:
  
    http://lear.inrialpes.fr/src/deepmatching/
  
  
  Matlab wrapper:
    [Prerequisite: to have compiled the executable, see above.]
    
    1) Launch matlab by preloading the same 'libatlas' than the one used to compile ./deepmatching:
      LD_PRELOAD=/usr/lib64/atlas/libtatlas.so.3   matlab
    
    2) Compile the MEX file:
      mex deepmatching_matlab.cpp deep_matching.o conv.o hog.o image.o io.o main.o maxfilter.o pixel_desc.o -output deepmatching '-DUSEOMP' CFLAGS="-fPIC -Wall -g -std=c++11 -O3 -fopenmp" LDFLAGS="-fopenmp" -lpng -ljpeg -lm /usr/local/atlas/lib/libsatlas.so
    
    3) Try executing the code:
      >> help deepmatching
      >> deepmatching() % show some help about options
      >> img1 = single(imread('liberty1.png'));
      >> img2 = single(imread('liberty2.png'));
      >> matches = deepmatching( img1, img2, '-downscale 2 -v' );
      >> matches % print matches, should be as the listing shown below
  
  Python wrapper:
    1) Compile the python module:
      make python
    
    2) Try executing the code:
      >> import deepmatching as dm
      >> help(dm.deepmatching)
      >> dm.deepmatching() # show some help about options
      >> from PIL import Image
      >> import numpy as np
      >> img1 = np.array(Image.open('liberty1.png'))
      >> img2 = np.array(Image.open('liberty2.png'))
      >> matches = dm.deepmatching( img1, img2, '-downscale 2 -v' )
      >> matches % print matches, should be as the listing shown below






Example usages and explanations:
  
  To get detailed information on parameters:
    ./deepmatching -h
    ./deepmatching --help
  
  
  * Build verification:
      ./deepmatching liberty1.png liberty2.png -downscale 2 -v
      
    should produce the following output:
      layer 0, patch_size = 16x16
      remaining 16 big cells (actually, 16 are unique)
      layer 1, patch_size = 32x32
      remaining 25 big cells (actually, 25 are unique)
      layer 2, patch_size = 64x64
      remaining 25 big cells (actually, 25 are unique)
      found 625 local matches
      gathering correspondences 96%...
      8 8 0 12 2.6554 10
      8 40 4 48 2.65679 11
      8 24 8 32 2.5486 11
      40 40 40 32 2.64178 0
      40 56 44 52 2.58631 0
      40 24 40 12 2.65065 0
      56 40 56 28 2.64225 0
      56 24 56 12 2.68497 0
      24 40 24 32 2.62045 3
      24 56 28 60 2.5849 12
  
  * To visualize the output correspondences:
    Use the "viz.py" python script provided.
      ./deepmatching climb1.png climb2.png -nt 0 | python viz.py climb1.png climb2.png
  
  * To restrict matching to local neighborhood:
    The "-ngh_rad <D>" option restricts the matching to a radius of <D> pixels.
    It uses less memory and is faster. For instance, This should produce about 
    the same output as before but consumes 2 times less memory and cpu:
    
      ./deepmatching climb1.png climb2.png -nt 0 -ngh_rad 192 | python viz.py climb1.png climb2.png
  
 * To rescore matches prior to calling deepflow / epicflow:
    simply pipe the output correspondences in 'rescore.py'
      ./deepmatching img1 img2 [args] | python rescore.py img1 img2
  
  
 * Scale and invariant version: (see the --help)
      ./deepmatching dino1.jpg dino2.jpg -nt 0 -downscale 1 -max_scale 2 -rot_range -45 +45 -v | python viz.py dino1.jpg dino2.jpg
    
    param -max_scale: maximum scale factor (here x2, default = x5)
    param -rot_range: rotation range in degrees (default = from 0 to 360)


For details about the options, please refer to the help, the papers or the code.


Important tip:
  If the program stops with "segmentation fault", then it means that your machine 
  does not have enough memory. In this case, you should consider increasing the 
  "-downscale" parameter.


Version history:

  version 1.0.2:
    Many thanks to Bowen Zhang from Tongji University for reporting an issue with the makefile

  version 1.1:
  - New mode added for "fully scale & rotation invariant DeepMatching".
  - Improved visualisation (viz.py) 
  - Removed useless/suboptimal options (-iccv_settings)
  - Fixed a bug related to memory allocation for large images

  version 1.2:
  - Added a new option "-ngh_rad" to restrict the matching to a local neighborhood, which allows
    much reduced memory usage and computations.
  - static-compiled version is now fully multhi-threaded with BLAS
  - few minor bugfix, code cleaning and updates.

  version 1.2.1:
  - Now performing the maxpooling and subsampling steps jointly,
    which results in 2/3 of memory usage compared to before. Also, it is now a bit faster.
  - Removed some useless/confusing options in the executable.
  
  version 1.2.2:
  - Now include a Matlab and a Python wrapper!
  






















