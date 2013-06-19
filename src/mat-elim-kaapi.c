#include <math.h>
#include <kaapic.h>
#include <getopt.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>

#include "include/pla-config.h"
#include "mat-elim-tools.h"

static inline void kaapiElim1D(
    int start, int end, int tid, int l, int m,
    unsigned int *a, unsigned int inv, unsigned int prime,
    unsigned int index
                              )
{
  int i, j, k;
  unsigned int mult;
  i = index;
  for (j = start; j < end; ++j) {
    mult  = a[i+j*m] * inv;
    for (k = i+1; k < m; ++k) {
      a[k+j*m]  +=  a[k+i*m] * mult;
    }
  }
}

// multiplies A*B^T and stores it in *this
void elim(int l,int m,int thrds,int bs) {

  printf("Naive Gaussian Elimination\n");
  struct timeval start, stop;
  clock_t cStart, cStop;
  int i, j, k;
  // kaapic stuff
  int blocksize     = bs;
  int threadNumber  = thrds;
  int err = kaapic_init(1);
  kaapic_foreach_attr_t attr;
  kaapic_init(1);
  kaapic_foreach_attr_init(&attr);
  kaapic_foreach_attr_set_grains(&attr, blocksize, blocksize);

  unsigned int *a = (unsigned int *)malloc(sizeof(unsigned int) * (l * m));
  srand(time(NULL));
  for (i=0; i< l*m; i++) {
    a[i]  = rand();
  }
 
  unsigned int boundary = (l > m) ? m : l;
  unsigned int inv;
  unsigned int prime = 65521;

  gettimeofday(&start, NULL);
  cStart  = clock();
  
  for (i = 0; i < boundary; ++i) {
    inv = negInverseModP(a[i+i*m], prime);

    kaapic_foreach(i+1, l, &attr, 6, kaapiElim1D, l, m, a, inv, prime, i);
  }

  err = kaapic_finalize();
  gettimeofday(&stop, NULL);
  cStop = clock();
  // compute FLOPS:
  // assume addition and multiplication in the mult kernel are 2 operations
  // done A.nRows() * B.nRows() * B.nCols()

  double flops = 0;
  flops = countGEPFlops(l, m);
  float epsilon = 0.0000000001;
  double realtime = ((stop.tv_sec - start.tv_sec) * 1e6 + 
                    (stop.tv_usec - start.tv_usec)) / 1e6;
  double cputime  = (double)((cStop - cStart)) / CLOCKS_PER_SEC;
  char buffer[50];
  // get digits before decimal point of cputime (the longest number) and setw
  // with it: digits + 1 (point) + 4 (precision) 
  int digits = sprintf(buffer,"%.0f",cputime);
  double ratio = cputime/realtime;
  printf("- - - - - - - - - - - - - - - - - - - - - - - - - -\n");
  printf("Method:           KAAPIC 1D\n");
  printf("#Threads:         %d\n", threadNumber);
  printf("Chunk size:       %d\n", bs);
  printf("Real time:        %.4f sec\n", realtime);
  printf("CPU time:         %.4f sec\n", cputime);
  if (cputime > epsilon)
    printf("CPU/real time:    %.4f\n", ratio);
  printf("- - - - - - - - - - - - - - - - - - - - - - - - - -\n");
  printf("GFLOPS/sec:       %.4f\n", flops / (1000000000 * realtime));
  printf("---------------------------------------------------\n");
}

// cache-oblivious implementation

void base_case( unsigned int *M, const unsigned int k1, const unsigned int i1,
                        const unsigned int j1, const unsigned int rows,
                        const unsigned int cols, unsigned int size,
                        unsigned int prime, unsigned int*neg_inv_piv) {
  unsigned int k;

  for (k = 0; k < size; k++) {
    M[(k1+k)+(k1+k)*cols] %= prime;
    // possibly the negative inverses of the pivots at place (k,k) were already
    // computed in another call. otherwise we need to compute and store it
    if (!neg_inv_piv[k+k1]) {
      if (M[(k1+k)+(k1+k)*cols] != 0) {
        neg_inv_piv[k+k1] = negInverseModP(M[(k1+k)+(k1+k)*cols], prime);
      }
    }
    const unsigned int inv_piv   = neg_inv_piv[k+k1];
    // if the pivots are in the same row part of the matrix as Mmdf then we can
    // always start at the next row (k+1), otherwise we need to start at
    // row 0
    const unsigned int istart  = (k1 == i1) ? k+1 : 0;
    for (unsigned int i = istart; i < size; i++) {
      const unsigned int tmp = (M[k+k1+(i1+i)*cols] * inv_piv);
      //const unsigned int tmp = (M[k+k1+(i1+i)*cols] * inv_piv) % prime;
      // if the pivots are in the same column part of the matrix as Mmdf then we can
      // always start at the next column (k+1), otherwise we need to start at
      // column 0
      const unsigned int jstart  = (k1 == j1) ? k+1 : 0;
      for (unsigned int j = jstart; j < size; j++) {
  	    M[(j1+j)+(i1+i)*cols]  +=  M[(j1+j)+(k1+k)*cols] * tmp;
  	   // M[(j1+j)+(i1+i)*cols]  %=  prime;
      }
    }
  }
}

void D1(unsigned int *M, const unsigned int k1, const unsigned int k2,
        const unsigned int i1, const unsigned int i2,
		    const unsigned int j1, const unsigned int j2,
		    const unsigned int rows, const unsigned int cols,
        unsigned int size, unsigned int prime,
        unsigned int *neg_inv_piv, unsigned int blocksize, int thrds) {
  if (i2 <= k1 || j2 <= k1)
    return;

  if (size <= blocksize) {
    base_case (M, k1, i1, j1, rows, cols, size, prime, neg_inv_piv);
  } else {
    size = size / 2;

    unsigned int km = (k1+k2) / 2 ;
    unsigned int im = (i1+i2) / 2;
    unsigned int jm = (j1+j2) / 2;

    kaapic_spawn_attr_t attr;

    kaapic_spawn_attr_init(&attr);

    // parallel - start
    kaapic_begin_parallel(KAAPIC_FLAG_DEFAULT);
    // X11
    kaapic_spawn(&attr, 14, D1,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, rows*cols, M,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, k1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, km,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, i1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, im,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, j1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, jm,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, rows,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, cols,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, size,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT64, 1, prime,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, sizeof(neg_inv_piv)/sizeof(neg_inv_piv[0]), neg_inv_piv,
        //KAAPIC_MODE_RW, KAAPIC_TYPE_INT, 4, neg_inv_piv,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, blocksize,
        KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, thrds);
    // X12
    kaapic_spawn(&attr, 14, D1,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, rows*cols, M,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, k1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, km,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, i1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, im,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, jm+1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, j2,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, rows,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, cols,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, size,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT64, 1, prime,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, sizeof(neg_inv_piv)/sizeof(neg_inv_piv[0]), neg_inv_piv,
        //KAAPIC_MODE_RW, KAAPIC_TYPE_INT, 4, neg_inv_piv,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, blocksize,
        KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, thrds);
    // X21
    kaapic_spawn(&attr, 14, D1,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, rows*cols, M,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, k1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, km,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, im+1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, i2,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, j1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, jm,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, rows,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, cols,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, size,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT64, 1, prime,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, sizeof(neg_inv_piv)/sizeof(neg_inv_piv[0]), neg_inv_piv,
        //KAAPIC_MODE_RW, KAAPIC_TYPE_INT, 4, neg_inv_piv,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, blocksize,
        KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, thrds);
    // X22
    kaapic_spawn(&attr, 14, D1,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, rows*cols, M,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, k1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, km,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, im+1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, i2,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, jm+1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, j2,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, rows,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, cols,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, size,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT64, 1, prime,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, sizeof(neg_inv_piv)/sizeof(neg_inv_piv[0]), neg_inv_piv,
        //KAAPIC_MODE_RW, KAAPIC_TYPE_INT, 4, neg_inv_piv,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, blocksize,
        KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, thrds);
    /*
    D1( M, k1, km, i1, im, j1, jm, rows, cols, size,
        prime, neg_inv_piv, thrds, blocksize);
    // X12
    D1( M, k1, km, i1, im, jm+1, j2, rows, cols, size,
        prime, neg_inv_piv, thrds, blocksize);
    // X21
    D1( M, k1, km, im+1, i2, j1, jm, rows, cols, size,
        prime, neg_inv_piv, thrds, blocksize);
    // X22
    D1( M, k1, km, im+1, i2, jm+1, j2, rows, cols, size,
        prime, neg_inv_piv, thrds, blocksize);
    */
    kaapic_sync();
    // parallel - end

    // parallel - start
    // X11
    kaapic_spawn(&attr, 14, D1,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, rows*cols, M,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, km+1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, k2,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, i1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, im,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, j1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, jm,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, rows,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, cols,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, size,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT64, 1, prime,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, sizeof(neg_inv_piv)/sizeof(neg_inv_piv[0]), neg_inv_piv,
        //KAAPIC_MODE_RW, KAAPIC_TYPE_INT, 4, neg_inv_piv,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, blocksize,
        KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, thrds);
    // X12
    kaapic_spawn(&attr, 14, D1,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, rows*cols, M,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, km+1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, k2,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, i1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, im,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, jm+1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, j2,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, rows,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, cols,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, size,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT64, 1, prime,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, sizeof(neg_inv_piv)/sizeof(neg_inv_piv[0]), neg_inv_piv,
        //KAAPIC_MODE_RW, KAAPIC_TYPE_INT, 4, neg_inv_piv,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, blocksize,
        KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, thrds);
    // X21
    kaapic_spawn(&attr, 14, D1,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, rows*cols, M,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, km+1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, k2,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, im+1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, i2,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, j1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, jm,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, rows,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, cols,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, size,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT64, 1, prime,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, sizeof(neg_inv_piv)/sizeof(neg_inv_piv[0]), neg_inv_piv,
        //KAAPIC_MODE_RW, KAAPIC_TYPE_INT, 4, neg_inv_piv,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, blocksize,
        KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, thrds);
    // X22
    kaapic_spawn(&attr, 14, D1,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, rows*cols, M,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, km+1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, k2,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, im+1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, i2,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, jm+1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, j2,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, rows,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, cols,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, size,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT64, 1, prime,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, sizeof(neg_inv_piv)/sizeof(neg_inv_piv[0]), neg_inv_piv,
        //KAAPIC_MODE_RW, KAAPIC_TYPE_INT, 4, neg_inv_piv,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, blocksize,
        KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, thrds);
    /*
    // X11
    D1( M, km+1, k2, i1, im, j1, jm, rows, cols, size,
        prime, neg_inv_piv, thrds, blocksize);
    // X12
    D1( M, km+1, k2, i1, im, jm+1, j2, rows, cols, size,
        prime, neg_inv_piv, thrds, blocksize);
    // X21
    D1( M, km+1, k2, im+1, i2, j1, jm, rows, cols, size,
        prime, neg_inv_piv, thrds, blocksize);
    // X22
    D1( M, km+1, k2, im+1, i2, jm+1, j2, rows, cols, size,
        prime, neg_inv_piv, thrds, blocksize);
    */
    kaapic_sync();
    kaapic_end_parallel(KAAPIC_FLAG_DEFAULT);
    // parallel - end
  }
}

void C1(unsigned int *M, const unsigned int k1, const unsigned int k2,
        const unsigned int i1, const unsigned int i2,
		    const unsigned int j1, const unsigned int j2,
		    const unsigned int rows, const unsigned int cols,
        unsigned int size, unsigned int prime,
        unsigned int *neg_inv_piv, unsigned int blocksize, int thrds) {
  if (i2 <= k1 || j2 <= k1)
    return;

  if (size <= blocksize) {
    base_case (M, k1, i1, j1, rows, cols, size, prime, neg_inv_piv);
  } else {
    size = size / 2;

    unsigned int km = (k1+k2) / 2 ;
    unsigned int im = (i1+i2) / 2;
    unsigned int jm = (j1+j2) / 2;

    kaapic_spawn_attr_t attr;

    kaapic_spawn_attr_init(&attr);

    // parallel - start
    kaapic_begin_parallel(KAAPIC_FLAG_DEFAULT);
    // X11
    kaapic_spawn(&attr, 14, C1,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, rows*cols, M,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, k1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, km,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, i1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, im,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, j1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, jm,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, rows,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, cols,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, size,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT64, 1, prime,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, sizeof(neg_inv_piv)/sizeof(neg_inv_piv[0]), neg_inv_piv,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, blocksize,
        KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, thrds);
    // X21
    kaapic_spawn(&attr, 14, C1,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, rows*cols, M,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, k1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, km,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, im+1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, i2,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, j1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, jm,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, rows,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, cols,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, size,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT64, 1, prime,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, sizeof(neg_inv_piv)/sizeof(neg_inv_piv[0]), neg_inv_piv,
        //KAAPIC_MODE_RW, KAAPIC_TYPE_INT, 4, neg_inv_piv,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, blocksize,
        KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, thrds);
    kaapic_sync();
    // parallel - end

    // parallel - start
    // X12
    kaapic_spawn(&attr, 14, D1,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, rows*cols, M,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, k1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, km,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, i1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, im,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, jm+1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, j2,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, rows,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, cols,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, size,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT64, 1, prime,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, sizeof(neg_inv_piv)/sizeof(neg_inv_piv[0]), neg_inv_piv,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, blocksize,
        KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, thrds);
    // X22
    kaapic_spawn(&attr, 14, D1,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, rows*cols, M,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, k1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, km,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, im+1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, i2,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, jm+1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, j2,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, rows,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, cols,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, size,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT64, 1, prime,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, sizeof(neg_inv_piv)/sizeof(neg_inv_piv[0]), neg_inv_piv,
        //KAAPIC_MODE_RW, KAAPIC_TYPE_INT, 4, neg_inv_piv,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, blocksize,
        KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, thrds);
    kaapic_sync();
    // parallel - end

    // parallel - start
    // X12
    kaapic_spawn(&attr, 14, C1,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, rows*cols, M,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, km+1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, k2,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, i1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, im,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, jm+1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, j2,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, rows,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, cols,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, size,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT64, 1, prime,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, sizeof(neg_inv_piv)/sizeof(neg_inv_piv[0]), neg_inv_piv,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, blocksize,
        KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, thrds);
    // X22
    kaapic_spawn(&attr, 14, C1,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, rows*cols, M,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, km+1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, k2,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, im+1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, i2,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, jm+1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, j2,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, rows,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, cols,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, size,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT64, 1, prime,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, sizeof(neg_inv_piv)/sizeof(neg_inv_piv[0]), neg_inv_piv,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, blocksize,
        KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, thrds);
    kaapic_sync();
    // parallel - end

    // parallel - start
    // X11
    kaapic_spawn(&attr, 14, D1,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, rows*cols, M,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, km+1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, k2,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, i1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, im,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, j1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, jm,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, rows,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, cols,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, size,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT64, 1, prime,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, sizeof(neg_inv_piv)/sizeof(neg_inv_piv[0]), neg_inv_piv,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, blocksize,
        KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, thrds);
    // X12
    kaapic_spawn(&attr, 14, D1,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, rows*cols, M,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, km+1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, k2,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, im+1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, i2,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, j1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, jm,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, rows,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, cols,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, size,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT64, 1, prime,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, sizeof(neg_inv_piv)/sizeof(neg_inv_piv[0]), neg_inv_piv,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, blocksize,
        KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, thrds);
    kaapic_sync();
    kaapic_end_parallel(KAAPIC_FLAG_DEFAULT);
    // parallel - end

  }
}

void B1(unsigned int *M, const unsigned int k1, const unsigned int k2,
        const unsigned int i1, const unsigned int i2,
		    const unsigned int j1, const unsigned int j2,
		    const unsigned int rows, const unsigned int cols,
        unsigned int size, unsigned int prime,
        unsigned int *neg_inv_piv, unsigned int blocksize, int thrds) {
  if (i2 <= k1 || j2 <= k1)
    return;

  if (size <= blocksize) {
    base_case (M, k1, i1, j1, rows, cols, size, prime, neg_inv_piv);
  } else {
    size = size / 2;

    unsigned int km = (k1+k2) / 2 ;
    unsigned int im = (i1+i2) / 2;
    unsigned int jm = (j1+j2) / 2;

    kaapic_spawn_attr_t attr;

    kaapic_spawn_attr_init(&attr);

    // parallel - start
    kaapic_begin_parallel(KAAPIC_FLAG_DEFAULT);
    // X11
    kaapic_spawn(&attr, 14, B1,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, rows*cols, M,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, k1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, km,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, i1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, im,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, j1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, jm,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, rows,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, cols,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, size,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT64, 1, prime,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, sizeof(neg_inv_piv)/sizeof(neg_inv_piv[0]), neg_inv_piv,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, blocksize,
        KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, thrds);
    // X12
    kaapic_spawn(&attr, 14, B1,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, rows*cols, M,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, k1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, km,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, i1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, im,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, jm+1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, j2,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, rows,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, cols,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, size,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT64, 1, prime,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, sizeof(neg_inv_piv)/sizeof(neg_inv_piv[0]), neg_inv_piv,
        //KAAPIC_MODE_RW, KAAPIC_TYPE_INT, 4, neg_inv_piv,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, blocksize,
        KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, thrds);
    kaapic_sync();
    // parallel - end

    // parallel - start
    // X21
    kaapic_spawn(&attr, 14, D1,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, rows*cols, M,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, k1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, km,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, im+1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, i2,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, j1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, jm,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, rows,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, cols,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, size,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT64, 1, prime,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, sizeof(neg_inv_piv)/sizeof(neg_inv_piv[0]), neg_inv_piv,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, blocksize,
        KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, thrds);
    // X22
    kaapic_spawn(&attr, 14, D1,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, rows*cols, M,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, k1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, km,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, im+1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, i2,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, jm+1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, j2,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, rows,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, cols,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, size,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT64, 1, prime,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, sizeof(neg_inv_piv)/sizeof(neg_inv_piv[0]), neg_inv_piv,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, blocksize,
        KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, thrds);
    kaapic_sync();
    // parallel - end

    // parallel - start
    // X21
    kaapic_spawn(&attr, 14, B1,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, rows*cols, M,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, km+1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, k2,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, im+1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, i2,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, j1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, jm,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, rows,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, cols,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, size,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT64, 1, prime,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, sizeof(neg_inv_piv)/sizeof(neg_inv_piv[0]), neg_inv_piv,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, blocksize,
        KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, thrds);
    // X22
    kaapic_spawn(&attr, 14, B1,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, rows*cols, M,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, km+1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, k2,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, im+1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, i2,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, jm+1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, j2,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, rows,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, cols,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, size,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT64, 1, prime,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, sizeof(neg_inv_piv)/sizeof(neg_inv_piv[0]), neg_inv_piv,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, blocksize,
        KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, thrds);
    kaapic_sync();
    // parallel - end

    // parallel - start
    // X11
    kaapic_spawn(&attr, 14, D1,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, rows*cols, M,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, km+1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, k2,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, i1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, im,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, j1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, jm,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, rows,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, cols,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, size,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT64, 1, prime,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, sizeof(neg_inv_piv)/sizeof(neg_inv_piv[0]), neg_inv_piv,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, blocksize,
        KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, thrds);
    // X12
    kaapic_spawn(&attr, 14, D1,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, rows*cols, M,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, km+1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, k2,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, i1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, im,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, jm+1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, j2,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, rows,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, cols,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, size,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT64, 1, prime,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, sizeof(neg_inv_piv)/sizeof(neg_inv_piv[0]), neg_inv_piv,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, blocksize,
        KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, thrds);
    kaapic_sync();
    kaapic_end_parallel(KAAPIC_FLAG_DEFAULT);
    // parallel - end

  }
}

void A( unsigned int *M, const unsigned int k1, const unsigned int k2,
        const unsigned int i1, const unsigned int i2,
		    const unsigned int j1, const unsigned int j2,
		    const unsigned int rows, const unsigned int cols,
        unsigned int size, unsigned int prime,
        unsigned int *neg_inv_piv, unsigned int blocksize, int thrds) {
  if (i2 <= k1 || j2 <= k1)
    return;
  //
  //if (size <= 2) {
  if (size <= blocksize) {
    base_case (M, k1, i1, j1, rows, cols, size, prime, neg_inv_piv);
  } else {
    size = size / 2;

    unsigned int km = (k1+k2) / 2 ;
    unsigned int im = (i1+i2) / 2;
    unsigned int jm = (j1+j2) / 2;

    kaapic_spawn_attr_t attr;

    kaapic_spawn_attr_init(&attr);

    // forward step
    /*
    A(M, k1, km, i1, im, j1, jm, rows, cols, size,
      prime, neg_inv_piv, thrds, blocksize);
    */
    kaapic_spawn(&attr, 14, A,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, rows*cols, M,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, k1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, km,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, i1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, im,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, j1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, jm,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, rows,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, cols,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, size,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT64, 1, prime,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, sizeof(neg_inv_piv)/sizeof(neg_inv_piv[0]), neg_inv_piv,
        //KAAPIC_MODE_RW, KAAPIC_TYPE_INT, 4, neg_inv_piv,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, blocksize,
        KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, thrds);

    kaapic_sync();

    // parallel - start
    kaapic_begin_parallel(KAAPIC_FLAG_DEFAULT);
    /*
    B1( M, k1, km, i1, im, jm+1, j2, rows, cols, size,
        prime, neg_inv_piv, thrds, blocksize);
    C1( M, k1, km, im+1, i2, j1, jm, rows, cols, size,
        prime, neg_inv_piv, thrds, blocksize);
    */
    kaapic_spawn(&attr, 14, B1,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, rows*cols, M,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, k1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, km,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, i1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, im,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, jm+1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, j2,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, rows,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, cols,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, size,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT64, 1, prime,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, sizeof(neg_inv_piv)/sizeof(neg_inv_piv[0]), neg_inv_piv,
        //KAAPIC_MODE_RW, KAAPIC_TYPE_INT, 4, neg_inv_piv,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, blocksize,
        KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, thrds);
    kaapic_spawn(0, 14, C1,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, rows*cols, M,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, k1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, km,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, im+1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, i2,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, j1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, jm,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, rows,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, cols,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, size,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT64, 1, prime,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, sizeof(neg_inv_piv)/sizeof(neg_inv_piv[0]), neg_inv_piv,
        //KAAPIC_MODE_RW, KAAPIC_TYPE_INT, 4, neg_inv_piv,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, blocksize,
        KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, thrds);
    kaapic_sync();
    kaapic_end_parallel(KAAPIC_FLAG_DEFAULT);
    // parallel - end

    /*
    D1( M, k1, km, im+1, i2, jm+1, j2, rows, cols, size,
        prime, neg_inv_piv, thrds, blocksize);
    */
    kaapic_spawn(&attr, 14, D1,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, rows*cols, M,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, k1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, km,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, im+1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, i2,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, jm+1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, j2,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, rows,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, cols,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, size,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT64, 1, prime,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, sizeof(neg_inv_piv)/sizeof(neg_inv_piv[0]), neg_inv_piv,
        //KAAPIC_MODE_RW, KAAPIC_TYPE_INT, 4, neg_inv_piv,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, blocksize,
        KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, thrds);

    kaapic_sync();

    // backward step

    /*
    A(M, km+1, k2, im+1, i2, jm+1, j2, rows, cols, size,
      prime, neg_inv_piv, thrds, blocksize);
    */
    kaapic_spawn(&attr, 14, A,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, rows*cols, M,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, km+1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, k2,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, im+1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, i2,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, jm+1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, j2,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, rows,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, cols,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, size,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT64, 1, prime,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, sizeof(neg_inv_piv)/sizeof(neg_inv_piv[0]), neg_inv_piv,
        //KAAPIC_MODE_RW, KAAPIC_TYPE_INT, 4, neg_inv_piv,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, blocksize,
        KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, thrds);

    kaapic_sync();
  }
}



void elim_co(int l,int m, int thrds, int bs) {

  int err           = kaapic_init(1);
  int threadNumber  = kaapic_get_concurrency();
  //C.resize(l*m);
  unsigned int blocksize;
  if (bs == 0)
    blocksize = __PLA_CPU_L1_CACHE / 8;
  else
    blocksize = bs;
  printf("Cache-oblivious Gaussian Elimination\n");
  struct timeval start, stop;
  clock_t cStart, cStop;
  int i, j, k;
  unsigned int prime = 65521;
  unsigned int *a = (unsigned int *)malloc(sizeof(unsigned int) * (l * m));
  srand(time(NULL));
  for (i=0; i< l*m; i++) {
    a[i]  = rand();
  }
  unsigned int boundary     = (l > m) ? m : l;
  unsigned int *neg_inv_piv = (unsigned int *)
    malloc(boundary * sizeof(unsigned int));
  a[0]            %=  prime;
  neg_inv_piv[0]  =   negInverseModP(a[0], prime);

  gettimeofday(&start, NULL);
  cStart  = clock();

  // computation of blocks
  kaapic_spawn_attr_t attr;

  kaapic_spawn_attr_init(&attr);

  kaapic_spawn(&attr, 14, A,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, l*m, a,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, 0,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, boundary-1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, 0,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, l-1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, 0,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, m-1,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, l,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, m,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, boundary,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT64, 1, prime,
        KAAPIC_MODE_RW, KAAPIC_TYPE_UINT64, sizeof(neg_inv_piv)/sizeof(neg_inv_piv[0]), neg_inv_piv,
        //KAAPIC_MODE_RW, KAAPIC_TYPE_INT, 4, neg_inv_piv,
        KAAPIC_MODE_V, KAAPIC_TYPE_UINT32, 1, blocksize,
        KAAPIC_MODE_V, KAAPIC_TYPE_INT, 1, thrds);

  kaapic_sync();

  err = kaapic_finalize();
  
  gettimeofday(&stop, NULL);
  cStop = clock();
  // compute FLOPS:
  // assume addition and multiplication in the mult kernel are 2 operations
  // done A.nRows() * B.nRows() * B.nCols()

  double flops = 0;
  flops = countGEPFlops(l, m);
  float epsilon = 0.0000000001;
  double realtime = ((stop.tv_sec - start.tv_sec) * 1e6 + 
                    (stop.tv_usec - start.tv_usec)) / 1e6;
  double cputime  = (double)((cStop - cStart)) / CLOCKS_PER_SEC;
  char buffer[50];
  // get digits before decimal point of cputime (the longest number) and setw
  // with it: digits + 1 (point) + 4 (precision) 
  int digits = sprintf(buffer,"%.0f",cputime);
  double ratio = cputime/realtime;
  printf("- - - - - - - - - - - - - - - - - - - - - - - - - -\n");
  printf("Method:           KAAPIC Spawn\n");
  printf("#Threads:         %d\n", threadNumber);
  printf("Chunk size:       %d\n", bs);
  printf("Real time:        %.4f sec\n", realtime);
  printf("CPU time:         %.4f sec\n", cputime);
  if (cputime > epsilon)
    printf("CPU/real time:    %.4f\n", ratio);
  printf("- - - - - - - - - - - - - - - - - - - - - - - - - -\n");
  printf("GFLOPS/sec:       %.4f\n", flops / (1000000000 * realtime));
  printf("---------------------------------------------------\n");
}

void print_help(int exval) {
  printf("DESCRIPTION\n");
  printf("       Computes the Gaussian Elimination of a matrix A with\n");
  printf("       unsigned integer entries.\n");
  printf("       It uses the KAAPIC parallel scheduler.\n");

  printf("OPTIONS\n");
  printf("       -b SIZE   block- resp. chunksize\n");
  printf("                 default: L1 cache size\n");
  printf("       -c        cache-oblivious Gaussian Elimination\n");
  printf("       -h        print help\n");
  printf("       -l ROWSA  row size of matrix A\n");
  printf("                 default: 2000\n");
  printf("       -m COLSA  column size of matrix A and row size of matrix B\n");
  printf("                 default: 2000\n");
  printf("       -t THRDS  number of threads\n");
  printf("                 default: 1\n");
  printf("                 Note that you have to put 'KAAPI_CPUCOUNT=value of t you want'\n"); 
  printf("                 in front of your call, otherwise the number of cores is not set\n");

  exit(exval);
}


int main(int argc, char *argv[]) {
  int opt;
  // default values
  int l = 2000, m = 2000, t=1, bs=0, c=0;
  // biggest prime < 2^16

  /* 
  // no arguments given
  */
  if(argc == 1) {
    //fprintf(stderr, "This program needs arguments....\n\n");
    //print_help(1);
  }

  while((opt = getopt(argc, argv, "hl:m:t:b:c")) != -1) {
    switch(opt) {
      case 'h':
        print_help(0);
        break;
      case 'l': 
        l = atoi(strdup(optarg));
        break;
      case 'm': 
        m = atoi(strdup(optarg));
        break;
      case 't': 
        t = atoi(strdup(optarg));
        break;
      case 'b': 
        bs = atoi(strdup(optarg));
        break;
      case 'c': 
        c = 1;
        break;
    }
  }
  if (c)
    elim_co(l,m,t,bs);
  else
    elim(l,m,t,bs);

  return 0;
}
