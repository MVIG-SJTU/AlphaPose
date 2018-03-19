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
#ifndef ___ARRAY_TYPES_H___
#define ___ARRAY_TYPES_H___

typedef unsigned char UBYTE;
typedef unsigned int UINT;


/************************
* 1D Array
  
  Equivalences:
  
    C/Python/numpy:  array.shape = (tx,)
                     array[x] := array->pixels[x]
    
    Matlab/Fortran:  [1, tx] = size(array)
                     array(x, 1) := array->pixels[x-1]
*/

#define DEFINE_ARRAY(type)  \
    typedef struct {  \
      type* pixels; \
      int tx; \
    } type##_array;

DEFINE_ARRAY(UBYTE)
DEFINE_ARRAY(int)
DEFINE_ARRAY(UINT)
DEFINE_ARRAY(float)

#define ASSERT_ARRAY_ZEROS(arr) {int size=arr->tx; assert((arr->pixels[0]==0 && arr->pixels[size/2]==0 && arr->pixels[size-1]==0) || !"error: matrix " #arr "is supposed to be zeros");}


/************************
* 2D Image
  
  Equivalences:
  
    C/Python/numpy:  array.shape = (ty, tx)
                     array[y, x] := array->pixels[x + y*tx]
    
    Matlab/Fortran:  [tx, ty] = size(array)
                     array(x, y) := array->pixels[(x-1) + (y-1)*tx]
*/

#define DEFINE_IMG(type)    \
    typedef struct { \
      type* pixels;\
      int tx,ty;\
    } type##_image;

DEFINE_IMG(UBYTE)
DEFINE_IMG(int)
DEFINE_IMG(UINT)
DEFINE_IMG(float)

#define ASSERT_SAME_SIZE  ASSERT_SAME_IMG_SIZE
#define ASSERT_IMG_SIZE  ASSERT_SAME_IMG_SIZE
#define ASSERT_SAME_IMG_SIZE(im1,im2)  if(im1 && im2)  assert(im1->tx==im2->tx && im1->ty==im2->ty);

#define ASSERT_IMAGE_ZEROS
#define ASSERT_IMG_ZEROS(img) {int size=img->tx*img->ty; assert((img->pixels[0]==0 && img->pixels[size/2]==0 && img->pixels[size-1]==0) || !"error: matrix " #img "is supposed to be zeros");}
#define IMG_SIZE(img) (long((img)->tx)*(img)->ty)



/************************
* 3D Image = Cube (Z coordinates are contiguous)
  
  Equivalences:
  
    C/Python/numpy:  array.shape = (ty, tx, tz)
                     array[y, x, z] := array->pixels[z + x*tz + y*tx*tz]
    
    Matlab/Fortran:  [tz, tx, ty] = size(array)
                     array(z, x, y) := array->pixels[(z-1) + (x-1)*tz + (y-1)*tx*tz]
*/

#define DEFINE_CUBE(type) \
    typedef struct {  \
      type* pixels;  \
      int tx,ty,tz;  \
    } type##_cube;

DEFINE_CUBE(UBYTE)
DEFINE_CUBE(short)
DEFINE_CUBE(int)
DEFINE_CUBE(UINT)
DEFINE_CUBE(float)

#define ASSERT_SAME_CUBE_SIZE(im1, im2)   \
  if((im1) && (im2))  assert((im1)->tx==(im2)->tx && (im1)->ty==(im2)->ty && (im1)->tz==(im2)->tz);

#define ASSERT_CUBE_ZEROS(img) {int size=img->tx*img->ty*img->tz; assert((img->pixels[0]==0 && img->pixels[size/2]==0 && img->pixels[size-1]==0) || !"error: matrix " #img "is supposed to be zeros");}
#define CUBE_SIZE(cube) (long((cube)->tx)*(cube)->ty*(cube)->tz)



/************************
* 3D Image = concatenation of XY layers
  
  Equivalences:
  
    C/Python/numpy:  array.shape = (tz, ty, tx)
                     array[z, y, x] := array->pixels[x + y*tx + z*tx*ty]
    
    Matlab/Fortran:  [tx, ty, tz] = size(array)
                     array(x, y, z) := array->pixels[(x-1) + (y-1)*tx + (z-1)*tx*ty]
*/

#define DEFINE_LAYERS(type) \
    typedef struct {  \
      type* pixels;  \
      int tx,ty,tz;  \
    } type##_layers;  \

DEFINE_LAYERS(UBYTE)
DEFINE_LAYERS(int)
DEFINE_LAYERS(UINT)
DEFINE_LAYERS(float)


#define ASSERT_SAME_LAYERS_SIZE(im1,im2)  ASSERT_SAME_CUBE_SIZE(im1,im2)
#define ASSERT_LAYERS_ZEROS ASSERT_CUBE_ZEROS
#define LAYERS_SIZE(layers)   CUBE_SIZE(layers)



/*****************
  creation, reshaping macros
*/

// Because there was a random bug happening because of uninitialized memory
// and the bug was difficult to locate, I have just transformed all malloc(...)
// into calloc(...) ( = malloc + memset(0) ), which is not really consuming more time anyways
// and seems to solve the issue. This is kind of stupid technique but it works well.

#define empty_array(type,tx)        ((type##_array){NEWAC(type,long(tx)),tx})
#define empty_image(type,tx,ty)     ((type##_image){NEWAC(type,long(tx)*(ty)),tx,ty})
#define empty_cube(type,tx,ty,tz)   ((type##_cube  ){NEWAC(type,long(tx)*(ty)*long(tz)),tx,ty,tz})
#define empty_layers(type,tx,ty,tz) ((type##_layers){NEWAC(type,long(tx)*(ty)*(tz)),tx,ty,tz})

#define zeros_array(type,tx)        ((type##_array){NEWAC(type,long(tx)),tx})
#define zeros_image(type,tx,ty)     ((type##_image){NEWAC(type,long(tx)*(ty)),tx,ty})
#define zeros_cube(type,tx,ty,tz)   ((type##_cube  ){NEWAC(type,long(tx)*(ty)*(tz)),tx,ty,tz})
#define zeros_layers(type,tx,ty,tz) ((type##_layers){NEWAC(type,long(tx)*(ty)*(tz)),tx,ty,tz})

#define array_like(type,l)     ((type##_array){NEWAC(type,long((l)->tx)),(l)->tx})
#define image_like(type,l)     ((type##_image){NEWAC(type,long((l)->tx)*(l)->ty),(l)->tx,(l)->ty})
#define cube_like(type,l)      ((type##_cube  ){NEWAC(type,long((l)->tx)*(l)->ty*(l)->tz),(l)->tx,(l)->ty,(l)->tz})
#define layers_like(type,l)    ((type##_layers){NEWAC(type,long((l)->tx)*(l)->ty*(l)->tz),(l)->tx,(l)->ty,(l)->tz})


#define reshape_xy(type, arr)   ((type##_array){(arr)->pixels, (arr)->tx*(arr)->ty})
#define reshape_xyz(type, arr)  ((type##_array){(arr)->pixels, (arr)->tx*(arr)->ty*(arr)->tz})
#define reshape_xy_z(type, arr) ((type##_image){(arr)->pixels, (arr)->tx*(arr)->ty, (arr)->tz})
#define reshape_z_xy(type, arr) ((type##_image){(arr)->pixels, (arr)->tz, (arr)->tx*(arr)->ty})
#define reshape_x_yz(type, arr) ((type##_image){(arr)->pixels, (arr)->tx, (arr)->ty*(arr)->tz})


#define free_image(img) if(img){free(img->pixels); free(img); img=NULL;}
#define free_cube(cube) free_image(cube)
#define free_layers(cube) free_cube(cube)


// debugging only
//#include <stdio.h>  
//inline long hash_arr(char* ptr, int nb, bool show) {
//  long res = 0;
//  if(show) printf("hashing [");
//  for(int i=0; i<nb; i++) {
//    res = 1000003*res + ((UBYTE*)ptr)[i];
//    if(show)  printf("%d, ",((UBYTE*)ptr)[i]);
//    res = (res>>17) | (res<<47);
//  }
//  if(show)  printf("]\n");
//  return res;
//}
//#define H(arr,val)       printf("hash(" #arr ") = %ld\n",val);
//#define hash_array(arr)  H(arr,hash_arr((char*)(arr)->pixels,(arr)->tx*sizeof(*(arr)->pixels),0))
//#define hash_image(arr)  H(arr,hash_arr((char*)(arr)->pixels,(arr)->tx*(arr)->ty*sizeof(*(arr)->pixels),0))
//#define hash_cube(arr)   H(arr,hash_arr((char*)(arr)->pixels,(arr)->tx*(arr)->ty*(arr)->tz*sizeof(*(arr)->pixels),0))
//#define hash_layers(arr)  hash_cube(arr)

//inline void save_raw(const char* fname, int* shape, int ndim, char* ptr, int size) {
//  FILE* f = fopen(fname, "w");
//  fwrite( &ndim, sizeof(int), 1, f);
//  fwrite( shape, sizeof(int), ndim, f);
//  fwrite( ptr, sizeof(*ptr), size, f);
//  fclose(f);
//}
//#define save_cube(fname,cube) {int sh[3] = {(cube)->ty, (cube)->tx, (cube)->tz}; save_raw(fname, sh, 3, (char*)(cube)->pixels, sizeof(*(cube)->pixels)*CUBE_SIZE(cube));}
//#define save_layers(fname,layers) {int sh[3] = {(layers)->tz, (layers)->ty, (layers)->tx}; save_raw(fname, sh, 3, (char*)(layers)->pixels, sizeof(*(layers)->pixels)*LAYERS_SIZE(layers));}

#endif



































