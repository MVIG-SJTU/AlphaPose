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
#ifndef ___STD_H___
#define ___STD_H___

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
//#include <time.h>

#define MIN(a,b)  (((a)<(b)) ? (a) : (b))
#define MAX(a,b)  (((a)>(b)) ? (a) : (b))
#define SWAP(a,b,type)  {type _t = a; a = b; b = _t;}
#define between(min,val,max)  (min<=val && val<=max)

#define NEWA(type,n) (type*)malloc(sizeof(type)*long(n))
#define NEWAC(type,n) (type*)calloc(sizeof(type),(n))
#define NEW(type) NEWA(type,1)
#define REALLOC(ptr,type,n) ptr = (type*)realloc(ptr, sizeof(type)*long(n))

/* debugging macros */
#define P(x)  printf(#x " = %g\n",(double)(x));
#define D(x)  P(x)
#define DA(x,nb)  {int _iter; printf(#x " = {"); for(_iter=0; _iter<nb; _iter++) printf("%g,",(double)((x)[_iter])); puts("}");}
#define ASSERT(test,msg,p1)  if(!(test)){fprintf(stderr," ---\n  " msg "\n ---\n",p1); assert(0);}
#define EXIT(msg,p1)         ASSERT(1,msg,p1)

static inline void memswap( void* a, void* b, unsigned int nbytes ) {
  while(nbytes>=sizeof(double)) {
    double tmp = *(double*)a;
    *((double*&)a)++ = *(double*)b;
    *((double*&)b)++ = tmp;
    nbytes -= sizeof(double);
  }
  while(nbytes) {
    char tmp = *(char*)a;
    *((char*&)a)++ = *(char*)b;
    *((char*&)b)++ = tmp;
    nbytes--;
  }
}

static inline float pow2( float f ) {
  return f*f;
}
static inline bool ispowerof2( long n ) {
  return (n & (n-1))==0;
}

const double INF = 1.0/0.0;
const double NaN = 0.0/0.0;
const int INT_MIN = 0x80000000;
const int INT_MAX = 0x7FFFFFFF;
const float FLOAT_MIN = -1e39; // converted to -inf
const float FLOAT_MAX = +1e39; // converted to +inf


inline float min_array_f(const float* a, int n)  {
  int i=n;
  float res = FLOAT_MAX;
  while(i--)  if(a[i]<res)  res=a[i];
  return res;
}

inline float max_array_f(const float* a, int n)  {
  int i=n;
  float res = FLOAT_MIN;
  while(i--)  if(a[i]>res)  res=a[i];
  return res;
}

// override printf because matlab can't use it as such
void std_printf(const char* fmt, ... );
void err_printf(const char* fmt, ... );


//#include <sys/time.h>
//inline double now() 
//{
//  struct timeval tv;
//  gettimeofday (&tv,NULL);
//  return (tv.tv_sec*1e3 +tv.tv_usec*1e-3)/1000;
//}
//#define tic {double t = now();
//#define toc t=now()-t; printf("elapsed time = %g ms\n",1000*t);}

#endif





























