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
#ifndef ___IO_H___
#define ___IO_H___
#include <stdlib.h>

#include "image.h"
#include "deep_matching.h"

// output correspondences to a file or on the stdout
void output_correspondences( const char* out_filename, const corres_t* corres, int nb, float fx, float fy );

/* load a color image from a file */
color_image_t *color_image_load(const char *fname);

#endif
