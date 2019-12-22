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
#include "maxfilter.h"
#include "omp.h"

void _max_filter_3_horiz( float_image* img, float_image* res, int n_thread ) {
  ASSERT_SAME_SIZE(img,res);
  int j;
  const int tx = img->tx;
  const int ty = img->ty;
  
  #if defined(USE_OPENMP)
  #pragma omp parallel for num_threads(n_thread)
  #endif
  for(j=0; j<ty; j++) {
    int i;
    float *p = img->pixels + j*tx;
    float *r = res->pixels + j*tx;
    
    float m = MAX(p[0],p[1]);
    *r++ = m;
    
    for(i=1; i<tx-1; i++) {
      float m2 = MAX(p[i],p[i+1]);
      *r++ = MAX(m,m2);
      m=m2;
    }
    
    *r++ = m;
  }
}

void _max_filter_3_vert( float_image* img, float_image* res ) {
  ASSERT_SAME_SIZE(img,res);
  const int tx = img->tx;
  const int ty = img->ty;
  int j;
  
  for(j=0; j<ty-1; j++) {
    int i;
    float *p = img->pixels + j*tx;
    float *r = res->pixels + j*tx;
    
    for(i=0; i<tx; i++) {
      *r++ = MAX(p[i],p[i+tx]);
    }
  }
  memcpy(res->pixels+(ty-1)*tx,res->pixels+(ty-2)*tx,tx*sizeof(float)); // copy last row
  
  for(j=ty-2; j>0; j--) {
    int i;
    float *p = res->pixels + (j-1)*tx;
    float *r = res->pixels + j*tx;
    
    for(i=0; i<tx; i++) {
      float r0 = *r;
      *r++ = MAX(r0,p[i]);
    }
  }
}

void _max_filter_3( float_image* img, float_image* res, int n_thread ) {
  _max_filter_3_vert(img,res);
  _max_filter_3_horiz(res,res, res->ty>128? n_thread : 1);
}

void _max_filter_3_layers( float_layers* img, float_layers* res, int n_thread ) {
  ASSERT_SAME_LAYERS_SIZE(img,res);
  const long npix = img->tx*img->ty;
  
  int l;
  #if defined(USE_OPENMP)
  omp_set_nested(0);
  omp_set_dynamic(0);
  #pragma omp parallel for num_threads(n_thread)
  #endif
  for(l=0; l<img->tz; l++) {
    float_image img2 = {img->pixels + l*npix,img->tx,img->ty};
    float_image res2 = {res->pixels + l*npix,res->tx,res->ty};
    _max_filter_3( &img2, &res2, n_thread );
  }
}


/* Subsample an array, equivalent to res = img[:,::2,::2]
*/
void _subsample2( float_layers* img, float_layers* res, int n_thread ) {
  const int n_layers = res->tz;
  assert( img->tz==n_layers );
  const int tx = res->tx;
  const int ty = res->ty;
  assert( (img->tx+1)/2 == tx );
  assert( (img->ty+1)/2 == ty );
  
  long l;
  #if defined(USE_OPENMP)
  #pragma omp parallel for num_threads(n_thread)
  #endif
  for(l=0; l<n_layers; l++) {
    int x,y;
    for(y=0; y<ty; y++) {
      float* i = img->pixels + (l*img->ty + (2*y))*img->tx ;
      float* r = res->pixels + (l*ty + y)*tx;
      for(x=0; x<tx; x++)
        r[x] = i[x<<1];
    }
  }
}

/* joint max-pooling and subsampling
*/
void _max_filter_3_and_subsample_layers( float_layers* img, float_layers* res, int n_thread ) {
  const int n_layers = res->tz;
  assert( img->tz==n_layers );
  const int tx = res->tx;
  const int ty = res->ty;
  assert( tx>=2 && ty>=2 );
  const int tx2 = img->tx;
  const int ty2 = img->ty;
  assert( (tx2+1)/2 == tx );  // tx2=3 => tx=2
  assert( (ty2+1)/2 == ty );
  
  long l;
  #if defined(USE_OPENMP)
  #pragma omp parallel for num_threads(n_thread)
  #endif
  for(l=0; l<n_layers; l++) {
    // reset output
    memset(res->pixels + l*tx*ty, 0, tx*ty*sizeof(float));
    
    int x,y;
    for(y=0; y<ty; y++) {
      float* i = img->pixels + (l*ty2 + (2*y))*tx2 ;
      float* r = res->pixels + (l*ty + y)*tx;
      float* r2 = (y+1<ty) ? r + tx : r;  // pointer to next row
      
      #define maxEq(v,m)  v = (m>v) ? m : v
      
      // even rows of img
      for(x=0; x<tx-1; x++) {
        maxEq( r[x+0], *i ); // i[2*x+0]
        i++;
        maxEq( r[x+0], *i );   // i[2*x+1]
        maxEq( r[x+1], *i ); // i[2*x+1]
        i++;
      }
      // r[x+1] does NOT exist anymore
      maxEq( r[x+0], *i );    // i[2*x+0]
      i++;
      if(x<tx2/2) { // i[2*x+i] exists
        maxEq( r[x+0], *i );  // i[2*x+1]
        i++;
      }
      assert((i-img->pixels)%tx2 == 0);
      
      // odd rows of img
      if (y<ty2/2) { 
        for(x=0; x<tx-1; x++) {
          maxEq( r [x+0], *i );  // i[2*x+0]
          maxEq( r2[x+0], *i );  // i[2*x+0]
          i++;
          
          maxEq( r [x+0], *i );  // i[2*x+1]
          maxEq( r [x+1], *i );  // i[2*x+1]
          maxEq( r2[x+0], *i );  // i[2*x+1]
          maxEq( r2[x+1], *i );  // i[2*x+1]
          i++;
        }
        // r[x+1] does NOT exist anymore
        maxEq( r [x+0], *i );   // i[2*x+0]
        maxEq( r2[x+0], *i );   // i[2*x+0]
        i++;
        if(x<tx2/2) { // i[2*x+i] exists
          maxEq( r [x+0], *i );   // i[2*x+1]
          maxEq( r2[x+0], *i );   // i[2*x+1]
          i++;
        }
      }
      
      assert((i-img->pixels)%tx2 == 0);
      
      #undef maxEq
    }
  }
}



/* Subsample an array, equivalent to res = trueimg[:,offset_y::2,offset_x::2]
   except at boundaries, where the rules are a bit more complex:
    if img->tx % 2 == 0:
      if offset_x % 2 == 0: 
        trueimg[offset_x+img->tx-1] is also sampled
      else:
        trueimg[offset_x] is also sampled
    elif img->tx % 2 == 1:
      trueimg[offset_x] is also sampled
   
   ...and likewise for y dimension.
*/
void _subsample2_offset( float_layers* img, int_image* offsets, float_layers* res, int n_thread ) {
  const int n_layers = res->tz;
  assert( img->tz==n_layers );
  assert( offsets->tx==2 && offsets->ty==n_layers );
  const int tx = res->tx;
  const int ty = res->ty;
  assert( (img->tx+2)/2 == tx );
  assert( (img->ty+2)/2 == ty );
  
  long l;
  #if defined(USE_OPENMP)
  #pragma omp parallel for num_threads(n_thread)
  #endif
  for(l=0; l<n_layers; l++) {
    int x,y;
    const int ox = (offsets->pixels[2*l]+0x10000) % 2;
    const int oy = (offsets->pixels[2*l+1]+0x10000) % 2;
    assert(ox>=0 && oy>=0);
    #define get_img_2pos(x,tx,ox) MAX(0, MIN(img->tx-1, 2*x-ox))
    
    for(y=0; y<ty; y++) {
      float* i = img->pixels + (l*img->ty + get_img_2pos(y,ty,oy))*img->tx;
      float* r = res->pixels + (l*ty + y)*tx;
      r[0] = i[get_img_2pos(0,tx,ox)];  // first is special case
      for(x=1; x<tx-1; x++)
        r[x] = i[2*x-ox];
      r[x] = i[get_img_2pos(x,tx,ox)];  // last is special case
    }
    
    #undef get_img_2pos
  }
}





/* Max-pool in 2x2 px non-overlapping cells
*/
void _maxpool2( float_layers* img, float_layers* res, int n_thread ) {
  const int n_layers = res->tz;
  assert( img->tz==n_layers );
  const int tx = res->tx;
  const int ty = res->ty;
  assert( (img->tx)/2 == tx );
  assert( (img->ty)/2 == ty );
  
  long l;
  #if defined(USE_OPENMP)
  #pragma omp parallel for num_threads(n_thread)
  #endif
  for(l=0; l<n_layers; l++) {
    int x,y;
    for(y=0; y<ty; y++) {
      float* i = img->pixels + (l*img->ty + (2*y))*img->tx ;
      float* j = i + img->tx;
      float* r = res->pixels + (l*ty + y)*tx;
      for(x=0; x<tx; x++,i+=2,j+=2) {
        float mi = MAX(i[0],i[1]);
        float mj = MAX(j[0],j[1]);
        r[x] = MAX(mi,mj);
      }
    }
  }
}


/* average-pool in 2x2 px non-overlapping cells
*/
void _avgpool2( float_layers* img, float_layers* res, int n_thread ) {
  const int n_layers = res->tz;
  assert( img->tz==n_layers );
  const int tx = res->tx;
  const int ty = res->ty;
  assert( (img->tx)/2 == tx );
  assert( (img->ty)/2 == ty );
  
  long l;
  #if defined(USE_OPENMP)
  #pragma omp parallel for num_threads(n_thread)
  #endif
  for(l=0; l<n_layers; l++) {
    int x,y;
    for(y=0; y<ty; y++) {
      float* i = img->pixels + (l*img->ty + (2*y))*img->tx ;
      float* j = i + img->tx;
      float* r = res->pixels + (l*ty + y)*tx;
      for(x=0; x<tx; x++,i+=2,j+=2) {
        r[x] = 0.25*(i[0] + i[1] + j[0] + j[1]);
      }
    }
  }
}


typedef struct {
  int scale;
  int layer;
  int x,y;
  float score;
} one_max;

typedef struct {
  one_max* list;
  int n_elems, n_alloc;
} maxima;


#include <pthread.h>
static pthread_mutex_t mutex0, mutex1;


static inline void add_one_max( maxima* list, int scale, int layer, int x, int y, float score ) {
  pthread_mutex_lock (&mutex0);
  if( list->n_alloc <= list->n_elems ) {
    list->n_alloc = 3*(list->n_alloc+64)/2;
    list->list = (one_max*)realloc(list->list, sizeof(one_max)*list->n_alloc);
  }
  one_max* m = &list->list[list->n_elems++];
  m->scale = scale;
  m->layer = layer;
  m->x = x;
  m->y = y;
  m->score = score;
  pthread_mutex_unlock (&mutex0);
}



void _get_list_parents( int_cube* children, int_image* res ) {
  const int np2 = children->tz;
  assert( np2 == res->tx );
  const int n_cells_at_prev_scale = res->ty;
  int* parents = res->pixels;
  memset(parents,0xFF,n_cells_at_prev_scale*np2*sizeof(int));  // =-1 by default
  int i,j,ncells=children->tx*children->ty;
  int* cur = children->pixels;
  for(i=0; i<ncells; i++)
    for(j=0; j<np2; j++) {
      int c = *cur++;
      if(c<0) continue; // this one is not a real children
      parents[np2*c + j] = i;
    }
}

static inline int* get_list_parents( int_cube* children, int n_cells_at_prev_scale ) {
  const int np2 = children->tz;
  int_image res = {NEWA(int, n_cells_at_prev_scale*np2 ), np2, n_cells_at_prev_scale};
  _get_list_parents( children, &res );
  return res.pixels;
}


/* Return a list of local maxima in the scale-space of scores
*/
void _extract_maxima( res_scale* scales, int n_scales, float_array* sc_factor, float th, int min_scale, float nlpow, 
                      int check_parents, int check_children, int nobordure, int_image* res_out, int n_thread ) {
  
  assert( sc_factor->tx == n_scales );
  assert( min_scale>=0 && min_scale<n_scales );
  const float* scf = sc_factor->pixels;
  
  maxima res = {NULL,0,0};
  int s;
  
  // compute the maximum filter for each scale separately
  const int min_scale_max = MAX(0,min_scale);
  for(s=min_scale_max; s<n_scales; s++) {
    res_scale* sc = scales + s;
    float_layers r = sc->res_map;
    assert(sc->max_map.pixels==NULL); // not already allocated
    sc->max_map = r; // initialize tx,ty,tz
    sc->max_map.pixels = NEWA(float, r.tx*r.ty*r.tz );
    _max_filter_3_layers( &r, &sc->max_map, n_thread );
  }
  
  // then localize the local maxima in the scale-space
  for(s=min_scale; s<n_scales; s++) {
    res_scale* sc = scales + s;
    const int tx = sc->res_map.tx;
    const int ty = sc->res_map.ty;
    const long npix = tx*ty;
    const int n_layers = sc->assign.tx;
    
    // helpful values...
    const int f = sc->f;
    
    const int upper_tx =   (s+1<n_scales) ? sc[+1].res_map.tx : 0;
    const int upper_ty =   (s+1<n_scales) ? sc[+1].res_map.ty : 0;
    const int upper_npix = upper_tx*upper_ty;
    const float upper_scf= (s+1<n_scales) ? scf[s]/scf[s+1] : 0;
    const int np2 =        (s+1<n_scales) ? sc[+1].children.tz : 0;
    const int np = (int)sqrt(np2);
    const int upper_f =    (s+1<n_scales) ? sc[+1].f : 0;
    const int upper_gap =  (s+1<n_scales) ? sc[+1].patch_size/4 : 0;
    const float* upper_layers = (s+1<n_scales) ? sc[+1].max_map.pixels : NULL;
    const int* upper_assign = (s+1<n_scales) ? sc[+1].assign.pixels : NULL;
    const int* list_parents = (s+1<n_scales) && check_parents ? get_list_parents(&sc[+1].children,sc->grid.tx*sc->grid.ty) : NULL;
    
    const int down_tx =   (s>min_scale_max) ? sc[-1].res_map.tx : 0;
    const int down_ty =   (s>min_scale_max) ? sc[-1].res_map.ty : 0;
    const int down_npix = down_tx*down_ty;
    const float down_scf= (s>min_scale_max) ? scf[s]/scf[s-1] : 0;
    const int nc2 =       (s>min_scale_max) ? sc->children.tz : 0;
    const int nc = (int)sqrt(nc2);
    const int down_gap = sc->patch_size/4;
    const int down_f =    (s>min_scale_max) ? sc[-1].f : 0;
    const float* down_layers = (s>min_scale_max) ? sc[-1].max_map.pixels : NULL;
    const int* down_assign = (s>min_scale_max) ? sc[-1].assign.pixels : NULL;
    
    int l;
    #if defined(USE_OPENMP)
    #pragma omp parallel for num_threads(n_thread)
    #endif
    for(l=0; l<n_layers; l++) {
      // compute maxima_filter for each layer
      if(sc->assign.pixels[l]<0)  continue; // no layer for this
      float* res_map = sc->res_map.pixels + sc->assign.pixels[l]*npix;
      float* max_map = sc->max_map.pixels + sc->assign.pixels[l]*npix;
      
      // for each point which is a local maxima, check
      int i;
      for(i=0; i<npix; i++)
        if( res_map[i]>th && res_map[i]==max_map[i] ) {
          // ok, we have a maxima at this scale <s>
          const float val = res_map[i];
          int x = i%tx;
          int y = i/tx;
          if( nobordure && (x<1 || y<1 || x>=tx-1 || y>=ty-1) )  continue; // not interested in maxima on image bordures
          
          //if(s==2 && l==344 && x==41 && y==4) getchar();
          
          // now compare with lower scale
          if( check_children && s>min_scale_max ) {
            float valref = down_scf*val;
            int* children = sc->children.pixels + l*nc2;
            int u,v,ok=1;
            for(v=0; ok && v<nc; v++) {
              int uy = (f*y + (2*v/(nc-1)-1)*down_gap)/down_f;
              if( uy>=0 && uy<down_ty )
              for(u=0; u<nc; u++) {
                int ch = children[v*nc+u];
                if( ch < 0 )  continue;
                int ux = (f*x + (2*u/(nc-1)-1)*down_gap)/down_f;
                if( (ux>=0 && ux<down_tx) && 
                    valref < pow(down_layers[down_assign[ch]*down_npix + uy*down_tx + ux],nlpow) ) {ok = 0; break;}
              }
            }
            if(!ok) continue; // this is not a maximum
          }
          
          //if(s==2 && l==344 && x==41 && y==4) getchar();
          
          // now compare with upper scale <s+1> and eliminate non-maxima
          if( check_parents && list_parents ) {
            float valref = upper_scf*val;
            const int* parents = list_parents + l*np2;
            int u,v,ok=1;
            for(v=0; ok && v<np; v++) {
              int uy = (f*y + (1-2*v/(np-1))*upper_gap)/upper_f;
              if( uy>=0 && uy<upper_ty )
              for(u=0; u<np; u++) {
                const int p = parents[v*np+u];
                if( p<0 )  continue;
                int ux = (f*x + (1-2*u/(np-1))*upper_gap)/upper_f;
                if( (ux>=0 && ux<upper_tx) && 
                    valref < upper_layers[upper_assign[p]*upper_npix + uy*upper_tx + ux] ) {ok = 0; break;}
              }
            }
            if(!ok) continue; // this is not a maximum
          }
          
          add_one_max( &res, s, l, x, y, res_map[i] );
        }
    }
    
    free((void*)list_parents);
  }
  
  // free memory
  for(s=min_scale_max; s<n_scales; s++) {
    free(scales[s].max_map.pixels);
    scales[s].max_map.pixels = NULL;
  }
  
  res_out->tx = 5;
  res_out->ty = res.n_elems;
  res_out->pixels = (int*)res.list;
}


/* Return the best local children assignement in a 3x3 neigborhood
   l,u,v is the approximate position of the children in the corresponding response map[l,v,u]
*/
static inline float _local_argmax( long l, int u, int v, const float_layers* map, int extended, /*float reg,*/ int* x, int* y ) {
  assert(0<=l && l<map->tz);
  int umin = MAX(0, u-1);
  int vmin = MAX(0, v-1);
  const int etx = map->tx-extended; // because of extended response map
  const int ety = map->ty-extended;
  int umax = MIN(etx, u+2);
  int vmax = MIN(ety, v+2);
  
  // determine best children in the neighborhood (argmax)
  const int tx = map->tx;
  int i,j,bestx=0,besty=0; float m=0.f;
  const float *r = map->pixels + l*tx*map->ty;
  for(j=vmin; j<vmax; j++)
    for(i=umin; i<umax; i++) {
      const int p = j*tx+i;
      if(r[p]>m) {m=r[p]; bestx=i; besty=j;}
    }
  *x = bestx;
  *y = besty;
  return m;
}

/* Return the best assignment (= list of correspondences) for a given maxima
   from a pyramid top, this function returns 
   a list of weigthed correspondences (matches) between
   img0 pixels and img1 pixels
*/
void _argmax_correspondences_rec( res_scale* scales, int s, int l, int x, int y, 
                                  float_cube* res0, int step0, float_cube* res1, int step1, 
                                  int index, float score ) {
  res_scale* sc = scales + s;
  
  if(s==0) {
    const int x0 = sc->grid.pixels[2*l];
    const int y0 = sc->grid.pixels[2*l+1];
    const int x1 = sc->f * x;
    const int y1 = sc->f * y;
    
    const int qx0 = x0/step0;
    const int qy0 = y0/step0;
    //assert(0<=l && l<sc->res_map.tz);
    
    if( qx0<res0->tx && qy0<res0->ty ) {
      assert(qx0>=0 && qy0>=0);
      float* r0 = res0->pixels + ((qy0*res0->tx + qx0))*res0->tz;
      //assert(res0->pixels<=r0 && r0+5<res0->pixels+res0->tx*res0->ty*res0->tz);
      
      pthread_mutex_lock (&mutex0);
      if( score > r0[4] ) {
        // r[0:2] = pos in img0
        r0[0] = x0;
        r0[1] = y0;
        // r[2:4] = pos in img1
        r0[2] = x1;
        r0[3] = y1;
        // r[4] = score
        r0[4] = score;
        r0[5] = index;
      }
      pthread_mutex_unlock (&mutex0);
      
      const int qx1 = x1/step1;
      const int qy1 = y1/step1;
      assert(qx1>=0 && qy1>=0);
      if( qx1<res1->tx && qy1<res1->ty ) {
      float* r1 = res1->pixels + ((qy1)*res1->tx + (qx1))*res1->tz;
      //assert(res1->pixels<=r1 && r1+5<res1->pixels+res1->tx*res1->ty*res1->tz);
      pthread_mutex_lock (&mutex1);
      if( score > r1[4] ) {
        // r[0:2] = pos in img0
        r1[0] = x0;
        r1[1] = y0;
        // r[2:4] = pos in img1
        r1[2] = x1;
        r1[3] = y1;
        // r[4] = score
        r1[4] = score;
        r1[5] = index;
      }
      pthread_mutex_unlock (&mutex1);
      }
    }
  } else {
    // mark this maximum as already processed
    assert(0<=l && l<sc->assign.tx);
    if( sc->passed.pixels ) {
      const long truel = sc->assign.pixels[l];
      const long offset = ((truel*sc->true_shape[1] + MAX(0,y))*sc->true_shape[0] + MAX(0,x)) % sc->passed.tx;
      //pthread_mutex_lock (&mutex);
      int useless = ( sc->passed.pixels[offset] >= score );
      if(!useless)  sc->passed.pixels[offset] = score;
      //pthread_mutex_unlock (&mutex);
      if(useless) return; // this maximum was already investigated with a better score
    }
    
    const int f = sc->f;
    const res_scale* lower = &scales[s-1];
    const int lower_f = lower->f;
    // position in lower response map
    x *= f/lower_f;
    y *= f/lower_f;
    const int lower_gap = sc->patch_size/(4*lower_f); // gap is equal to patch_size/4 in absolute size
    const int nc2 = sc->children.tz;
    const int nc = (nc2==4) ? 2 : 3;
    const int* children = sc->children.pixels + l*nc2;
    const int* lower_ass = lower->assign.pixels;
    
    // for all children
    int u,v,c=0;
    for(v=0; v<nc; v++) {
      for(u=0; u<nc; u++,c++) {
        const int ch = children[c];
        if(ch<0) continue;
        const long l = lower_ass[ch];
        if(l<0) continue;
        
        // position of children in child1 = parent1 - (parent0-child0)
        int yc = y + (2*v/(nc-1)-1)*lower_gap;
        int xc = x + (2*u/(nc-1)-1)*lower_gap;
        int ex = 1; // extended response_maps 
        
        if( lower->offsets.pixels ) {
          // take offsets into account
          xc -= lower->offsets.pixels[2*l+0];
          yc -= lower->offsets.pixels[2*l+1];
          ex = 0; // no extension... maybe
        }
        
        // position of children in child1 = parent1 - (parent0-child0)
        int xb, yb;
        float child_score = _local_argmax( lower_ass[ch], xc, yc, &lower->res_map, ex, &xb, &yb );
        
        if( lower->offsets.pixels ) {
          // back to real image coordinates
          xb += lower->offsets.pixels[2*l+0];
          yb += lower->offsets.pixels[2*l+1];
        }
        
        if( child_score )
          _argmax_correspondences_rec( scales, s-1, ch, xb, yb, res0, step0, res1, step1, index, score + child_score );
      }
    }
  }
}


void _argmax_correspondences( res_scale* scales, int s, int l, int x, int y, float score, 
                              float_cube* res0, int step0, float_cube* res1, int step1, 
                              int index ) {
  assert(res0->tz==6);
  if(res1)  assert(res0->tz==6);
  _argmax_correspondences_rec( scales, s, l, x, y, res0, step0, res1, step1, index, score );
}


void _argmax_correspondences_rec_v1( res_scale* scales, int s, int l, int x, int y, 
                                  float_cube* res0, int step0, float_cube* res1, int step1, 
                                  int index, float top_score ) {
  res_scale* sc = scales + s;
  const int f = sc->f;
  
  if(s==0) {
    const int* ass = sc->assign.pixels;
    const float score = top_score * sc->res_map.pixels[(ass[l]*sc->res_map.ty + y)*sc->res_map.tx + x];
    const int x0 = sc->grid.pixels[2*l];
    const int y0 = sc->grid.pixels[2*l+1];
    const int x1 = f * x;
    const int y1 = f * y;
    
    const int qx0 = x0/step0;
    const int qy0 = y0/step0;
    if( qx0<res0->tx && qy0<res0->ty ) {
    float* r0 = res0->pixels + ((qy0*res0->tx + qx0))*res0->tz;
    
    pthread_mutex_lock (&mutex0);
    if( score > r0[4] ) {
      // r[0:2] = pos in img0
      r0[0] = x0;
      r0[1] = y0;
      // r[2:4] = pos in img1
      r0[2] = x1;
      r0[3] = y1;
      // r[4] = score
      r0[4] = score;
      r0[5] = index;
    }
    pthread_mutex_unlock (&mutex0);
    
    if( res1 ) {
    const int qx1 = x1/step1;
    const int qy1 = y1/step1;
//    if( qx1<res1->tx && qy1<res1->ty ) {  // useless check
    float* r1 = res1->pixels + ((qy1)*res1->tx + (qx1))*res1->tz;
    pthread_mutex_lock (&mutex1);
    if( score > r1[4] ) {
      // r[0:2] = pos in img0
      r1[0] = x0;
      r1[1] = y0;
      // r[2:4] = pos in img1
      r1[2] = x1;
      r1[3] = y1;
      // r[4] = score
      r1[4] = score;
      r1[5] = index;
    }
    pthread_mutex_unlock (&mutex1);
    }}
    
  } else {
    const res_scale* lower = &scales[s-1];
    const int lower_f = lower->f;
    // position in lower response map
    x *= f/lower_f;
    y *= f/lower_f;
    const int lower_gap = sc->patch_size/(4*lower_f); // gap is equal to patch_size/4 in absolute size
    const int nc2 = sc->children.tz;
    const int nc = (nc2==4) ? 2 : 3;
    const int* children = sc->children.pixels + l*nc2;
    const int* lower_ass = lower->assign.pixels;
    
    // remember all scores for all children
    int u,v,c=0;
    for(v=0; v<nc; v++) {
      const int yc = y + (2*v/(nc-1)-1)*lower_gap;
      for(u=0; u<nc; u++,c++) {
        int ch = children[c];
        if(ch<0) continue;
        const int xc = x + (2*u/(nc-1)-1)*lower_gap;
        
        // position of children in child1 = parent1 - (parent0-child0)
        const int l = lower_ass[children[c]];
        int xb=0, yb=0;
        float child_score = _local_argmax( l, xc, yc, &lower->res_map, 1, &xb, &yb );
        
        if( child_score>0 )
          _argmax_correspondences_rec_v1( scales, s-1, ch, xb, yb, res0, step0, res1, step1, index, top_score );
      }
    }
  }
}

void _argmax_correspondences_v1( res_scale* scales, int s, int l, int x, int y, float top_score, 
                              float_cube* res0, int step0, float_cube* res1, int step1, 
                              int index ) {
  assert(res0->tz==6);
  if(res1)  assert(res0->tz==6);
  _argmax_correspondences_rec_v1( scales, s, l, x, y, res0, step0, res1, step1, index, top_score );
}



static float** get_list_corres( const float_cube* map, int* nb ) {
  const int tz = map->tz;
  float* m = map->pixels;
  const long npix = map->tx*map->ty;
  float** res = NEWA(float*,npix);
  
  int i,n=0;
  for(i=0; i<npix; i++,m+=tz)
    if(m[4]) { // if score non-null
      res[n++] = m; // remember pointer
    }
  
  *nb = n;
  return res;
}

static inline int cmp_corres( const void* a, const void* b) {
  return memcmp(*(float**)a,*(float**)b,4*sizeof(float));
}


/* Intersect 2 mappings: erase all correspondences that are not reciprocal 
*/
float* _intersect_corres( const float_cube* map0, const float_cube* map1, int* nres ) {
  const int tz = 6;
  assert( map0->tz==tz && map1->tz==tz );
  
  // build the list of triplets
  int n0,n1;
  float** const corres0 = get_list_corres(map0,&n0);
  float** const corres1 = get_list_corres(map1,&n1);
  
  // arg-sort the lists
  qsort( corres0, n0, sizeof(float*), cmp_corres );
  qsort( corres1, n1, sizeof(float*), cmp_corres );
  
  // remove all correspondences from map0/map1 that is not shared
  float** c0 = corres0;
  float** c1 = corres1;
  float** const c0max = corres0 + n0;
  float** const c1max = corres1 + n1;
  float* res = NEWA(float, tz*MIN(n1,n0) );
  float* r = res;
  while(c0<c0max && c1<c1max) {
    int d = memcmp(*c0,*c1,5*sizeof(float));
    if(d<0) { // corres0 < corres1
      c0++;
    } else 
    if(d>0) { // corres0 > corres1
      c1++;
    } else { // corres0 == corres1
      if( r==res || memcmp( r-tz, *c0, tz*sizeof(float) ) ) { // if not already copied
        memcpy( r, *c0, tz*sizeof(float) );
        r += tz;
      }
      c0++;
      c1++;
    }
  }
  
  free(corres0);
  free(corres1);
  *nres = (r-res)/tz;
  return res;
}


/* erase corres in the first array that are not in the second one
*/
void transfer_corres_score( const float_image* ref, float_cube* map0 ) {
  const int tz = 6;
  assert( map0->tz==tz && ref->tx==tz );
  
  // build the list of triplets
  int n0,n1;
  float** const corres0 = get_list_corres(map0,&n0);
  float_cube map1 = {ref->pixels,1,ref->ty,ref->tx};
  float** const corres1 = get_list_corres(&map1,&n1);
  
  // arg-sort the lists
  qsort( corres0, n0, sizeof(float*), cmp_corres );
  qsort( corres1, n1, sizeof(float*), cmp_corres );
  
  // remove all correspondences from map0/map1 that is not shared
  float** c0 = corres0;
  float** c1 = corres1;
  float** const c0max = corres0 + n0;
  float** const c1max = corres1 + n1;
  while(c0<c0max && c1<c1max) {
    int d = memcmp(*c0,*c1,4*sizeof(float));
    if(d<0) { // corres0 < corres1
      c0++;
    } else 
    if(d>0) { // corres0 > corres1
      assert(!"error: 'ref in map0' is not verified");
      c1++;
    } else { // corres0 == corres1
      (*c0)[4] = (*c1)[4]; // copy score from ref
      c0++;
      c1++;
    }
  }
  
  while(c0<c0max) memset( *c0++, 0, tz*sizeof(float));
  
  free(corres0);
  free(corres1);
}



static inline float ptdot( const float* m, float x, float y ) {
  return x*m[0] + y*m[1] + m[2];
}


static void merge_one_side( const float aff[6], int step, float_cube* corres, float tol,  
                            int all_step, float_cube* all_corres, int offset ) {
  assert( corres->tz==6 && all_corres->tz==6 );
  const float* corres_pix = corres->pixels;
  assert(tol>=1); 
  tol*=tol; // squared
  float dmax = 2*step / sqrt( aff[0]*aff[4] - aff[1]*aff[3] );
  dmax*=dmax; // squared
  
  // for each bin of the final histograms, we get the nearest-neighbour bin in corres0 and corres1
  int i,j;
  for(j=0; j<all_corres->ty; j++) 
    for(i=0; i<all_corres->tx; i++) {
      float* all_cor = all_corres->pixels + (j*all_corres->tx + i)*corres->tz;
      
      // center of the bin in the reference frame
      float x = i*all_step + all_step/2;
      float y = j*all_step + all_step/2;
      
      // center of the bin on the rescaled+rotated image
      float xr = ptdot( aff + 0, x, y ); 
      float yr = ptdot( aff + 3, x, y );
      
      // iterate on the nearby bins
      int xb = (int)(0.5+ xr/step); // rescaled+rotated image is binned with size <step>
      int yb = (int)(0.5+ yr/step);
      int u,v;
      float best = 9e9f;
      for(v=MAX(0,yb-1); v<MIN(corres->ty,yb+2); v++)
        for(u=MAX(0,xb-1); u<MIN(corres->tx,xb+2); u++) {
          const float* cor = corres_pix + (v*corres->tx + u)*corres->tz;
          float d = pow2(cor[offset]-x) + pow2(cor[offset+1]-y);
          if( d < best && d<dmax )  best = d;
        }
      
      for(v=MAX(0,yb-1); v<MIN(corres->ty,yb+2); v++)
        for(u=MAX(0,xb-1); u<MIN(corres->tx,xb+2); u++) {
          const float* cor = corres_pix + (v*corres->tx + u)*corres->tz;
          float d = pow2(cor[offset]-x) + pow2(cor[offset+1]-y);
          if( d <= tol*best )  { // spatially close
            // merge correspondence if score is better than actual
            if( cor[4] > all_cor[4] )
              memcpy( all_cor, cor, 6*sizeof(float) );
          }
        }
      
    }
}


/* merge correspondences from several rotated/scaled version of an image into a single common reference frame
    rot0 = 2x3 rotation matrix:   (pt in rotated img0) = rot0 * (pt in ref frame)
    rot1 = 2x3 rotation matrix:   (pt in rotated img1) = rot1 * (pt in ref frame)
    step0 and step1 are bin size of correspondences histograms
    tol >= 1 is the tolerance to grid rotation (default = 2)
    corres0, corres1:          correspondences histograms of rotated image 
    all_corres0, all_corres1:  correspondences histograms of reference frame (result)
*/
void merge_corres( float rot0[6], float rot1[6], int step0, int step1, 
                   float_cube* corres0, float_cube* corres1, float tol, 
                   int all_step0, int all_step1, float_cube* all_corres0, float_cube* all_corres1 ) {
  
  merge_one_side( rot0, step0, corres0, tol, all_step0, all_corres0, 0 );
  merge_one_side( rot1, step1, corres1, tol, all_step1, all_corres1, 2 );
}




















































