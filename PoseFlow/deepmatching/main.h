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
#ifndef ___MAIN_H___
#define ___MAIN_H___

#define EXE_OPTIONS 0
#define MATLAB_OPTIONS 1
#define PYTHON_OPTIONS 2

#include "deep_matching.h"

void usage(const int language);

const char* parse_options(dm_params_t *params, scalerot_params_t *sr_params, bool *use_scalerot, float *fx, float *fy, const int argc, const char **argv, const int language, image_t **im1, image_t **im2);

#endif
