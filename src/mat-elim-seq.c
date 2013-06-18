#include <math.h>
#include <getopt.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>

#include "include/pla-config.h"
#include "mat-elim-tools.h"

// multiplies A*B^T and stores it in *this
void elim(int l,int m) {

  //C.resize(l*m);
  printf("Naive Gaussian Elimination\n");
  struct timeval start, stop;
  clock_t cStart, cStop;
  int i, j, k;
  unsigned int prime = 65521;
  unsigned int *a = (unsigned int *)malloc(sizeof(unsigned int) * (l * m));
  srand(time(NULL));
  for (i=0; i< l*m; i++) {
    a[i]  = rand();
  }
  unsigned int sum = 0;
 
  unsigned int boundary = (l > m) ? m : l;
  unsigned int inv, mult;

  gettimeofday(&start, NULL);
  cStart  = clock();
  
  for (i = 0; i < boundary; ++i) {
    inv = negInverseModP(a[i+i*m], prime);
    for (j = i+1; j < l; ++j) {
      mult = a[i+j*m] * inv;
      for (k = i+1; k < m; ++k) {
        a[k+j*m]  += a[k+i*m] * mult;
      }
    }
  }
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
  printf("Method:           Raw sequential\n");
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
        unsigned int *neg_inv_piv, unsigned int blocksize) {
  if (i2 <= k1 || j2 <= k1)
    return;

  if (size <= blocksize) {
    base_case (M, k1, i1, j1, rows, cols, size, prime, neg_inv_piv);
  } else {
    size = size / 2;

    unsigned int km = (k1+k2) / 2 ;
    unsigned int im = (i1+i2) / 2;
    unsigned int jm = (j1+j2) / 2;

    // parallel - start
    // X11
    D1( M, k1, km, i1, im, j1, jm, rows, cols, size,
        prime, neg_inv_piv, blocksize);
    // X12
    D1( M, k1, km, i1, im, jm+1, j2, rows, cols, size,
        prime, neg_inv_piv, blocksize);
    // X21
    D1( M, k1, km, im+1, i2, j1, jm, rows, cols, size,
        prime, neg_inv_piv, blocksize);
    // X22
    D1( M, k1, km, im+1, i2, jm+1, j2, rows, cols, size,
        prime, neg_inv_piv, blocksize);
    // parallel - end

    // parallel - start
    // X11
    D1( M, km+1, k2, i1, im, j1, jm, rows, cols, size,
        prime, neg_inv_piv, blocksize);
    // X12
    D1( M, km+1, k2, i1, im, jm+1, j2, rows, cols, size,
        prime, neg_inv_piv, blocksize);
    // X21
    D1( M, km+1, k2, im+1, i2, j1, jm, rows, cols, size,
        prime, neg_inv_piv, blocksize);
    // X22
    D1( M, km+1, k2, im+1, i2, jm+1, j2, rows, cols, size,
        prime, neg_inv_piv, blocksize);
    // parallel - end
  }
}

void C1(unsigned int *M, const unsigned int k1, const unsigned int k2,
        const unsigned int i1, const unsigned int i2,
		    const unsigned int j1, const unsigned int j2,
		    const unsigned int rows, const unsigned int cols,
        unsigned int size, unsigned int prime,
        unsigned int *neg_inv_piv, unsigned int blocksize) {
  if (i2 <= k1 || j2 <= k1)
    return;

  if (size <= blocksize) {
    base_case (M, k1, i1, j1, rows, cols, size, prime, neg_inv_piv);
  } else {
    size = size / 2;

    unsigned int km = (k1+k2) / 2 ;
    unsigned int im = (i1+i2) / 2;
    unsigned int jm = (j1+j2) / 2;

    // parallel - start
    // X11
    C1( M, k1, km, i1, im, j1, jm, rows, cols, size,
        prime, neg_inv_piv, blocksize);
    // X21
    C1( M, k1, km, im+1, i2, j1, jm, rows, cols, size,
        prime, neg_inv_piv, blocksize);
    // parallel - end

    // parallel - start
    // X12
    D1( M, k1, km, i1, im, jm+1, j2, rows, cols, size,
        prime, neg_inv_piv, blocksize);
    // X22
    D1( M, k1, km, im+1, i2, jm+1, j2, rows, cols, size,
        prime, neg_inv_piv, blocksize);
    // parallel - end

    // parallel - start
    // X12
    C1( M, km+1, k2, i1, im, jm+1, j2, rows, cols, size,
        prime, neg_inv_piv, blocksize);
    // X22
    C1( M, km+1, k2, im+1, i2, jm+1, j2, rows, cols, size,
        prime, neg_inv_piv, blocksize);
    // parallel - end

    // parallel - start
    // X11
    D1( M, km+1, k2, i1, im, j1, jm, rows, cols, size,
        prime, neg_inv_piv, blocksize);
    // X12
    D1( M, km+1, k2, im+1, i2, j1, jm, rows, cols, size,
        prime, neg_inv_piv, blocksize);
    // parallel - end
  }
}

void B1(unsigned int *M, const unsigned int k1, const unsigned int k2,
        const unsigned int i1, const unsigned int i2,
		    const unsigned int j1, const unsigned int j2,
		    const unsigned int rows, const unsigned int cols,
        unsigned int size, unsigned int prime,
        unsigned int *neg_inv_piv, unsigned int blocksize) {
  if (i2 <= k1 || j2 <= k1)
    return;

  if (size <= blocksize) {
    base_case (M, k1, i1, j1, rows, cols, size, prime, neg_inv_piv);
  } else {
    size = size / 2;

    unsigned int km = (k1+k2) / 2 ;
    unsigned int im = (i1+i2) / 2;
    unsigned int jm = (j1+j2) / 2;

    // parallel - start
    // X11
    B1( M, k1, km, i1, im, j1, jm, rows, cols, size,
        prime, neg_inv_piv, blocksize);
    // X12
    B1( M, k1, km, i1, im, jm+1, j2, rows, cols, size,
        prime, neg_inv_piv, blocksize);
    // parallel - end

    // parallel - start
    // X21
    D1( M, k1, km, im+1, i2, j1, jm, rows, cols, size,
        prime, neg_inv_piv, blocksize);
    // X22
    D1( M, k1, km, im+1, i2, jm+1, j2, rows, cols, size,
        prime, neg_inv_piv, blocksize);
    // parallel - end

    // parallel - start
    // X21
    B1( M, km+1, k2, im+1, i2, j1, jm, rows, cols, size,
        prime, neg_inv_piv, blocksize);
    // X22
    B1( M, km+1, k2, im+1, i2, jm+1, j2, rows, cols, size,
        prime, neg_inv_piv, blocksize);
    // parallel - end

    // parallel - start
    // X11
    D1( M, km+1, k2, i1, im, j1, jm, rows, cols, size,
        prime, neg_inv_piv, blocksize);
    // X12
    D1( M, km+1, k2, i1, im, jm+1, j2, rows, cols, size,
        prime, neg_inv_piv, blocksize);
    // parallel - end

  }
}

void A( unsigned int *M, const unsigned int k1, const unsigned int k2,
        const unsigned int i1, const unsigned int i2,
		    const unsigned int j1, const unsigned int j2,
		    const unsigned int rows, const unsigned int cols,
        unsigned int size, unsigned int prime,
        unsigned int *neg_inv_piv, unsigned int blocksize) {
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

    // forward step

    A(M, k1, km, i1, im, j1, jm, rows, cols, size,
      prime, neg_inv_piv, blocksize);
    // parallel - start
    B1( M, k1, km, i1, im, jm+1, j2, rows, cols, size,
        prime, neg_inv_piv, blocksize);
    C1( M, k1, km, im+1, i2, j1, jm, rows, cols, size,
        prime, neg_inv_piv, blocksize);
    // parallel - end
    D1( M, k1, km, im+1, i2, jm+1, j2, rows, cols, size,
        prime, neg_inv_piv, blocksize);
    // backward step
    A(M, km+1, k2, im+1, i2, jm+1, j2, rows, cols, size,
      prime, neg_inv_piv, blocksize);
  }
}



void elim_co(int l,int m, int bs) {

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
  
  A(a, 0, boundary-1, 0, l-1, 0, m-1, l, m, boundary,
    prime, neg_inv_piv, blocksize);

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
  printf("Method:           Raw sequential\n");
  printf("Real time:        %.4f sec\n", realtime);
  printf("CPU time:         %.4f sec\n", cputime);
  if (cputime > epsilon)
    printf("CPU/real time:    %.4f\n", ratio);
  printf("- - - - - - - - - - - - - - - - - - - - - - - - - -\n");
  printf("GFLOPS/sec:       %.4f\n", flops / (1000000000 * realtime));
  printf("---------------------------------------------------\n");
}


int main(int argc, char *argv[]) {
  int opt;
  // default values
  int l = 2000, m = 2000, c=0, b=0;
  // biggest prime < 2^16

  /* 
  // no arguments given
  */
  if(argc == 1) {
    //fprintf(stderr, "This program needs arguments....\n\n");
    //print_help(1);
  }

  while((opt = getopt(argc, argv, "l:m:b:c:")) != -1) {
    switch(opt) {
      case 'l': 
        l = atoi(strdup(optarg));
        break;
      case 'm': 
        m = atoi(strdup(optarg));
        break;
      case 'b': 
        b = atoi(strdup(optarg));
        break;
      case 'c': 
        c = atoi(strdup(optarg));
        break;
    }
  }
  if (c)
    elim_co(l,m,b);
  else
    elim(l,m);

  return 0;
}
