#include <math.h>
#include <omp.h>
#include <getopt.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>

#include "mat-elim-tools.h"

// multiplies A*B^T and stores it in *this
void elim(int l,int m,int thrds,int bs) {

  printf("Naive Gaussian Elimination\n");
  struct timeval start, stop;
  clock_t cStart, cStop;
  int i, j, k;
  // open mp stuff
  int threadNumber  = thrds;
  int blocksize     = bs;

  unsigned int *a = (unsigned int *)malloc(sizeof(unsigned int) * (l * m));
  srand(time(NULL));
  for (i=0; i< l*m; i++) {
    a[i]  = rand();
  }
  unsigned int sum = 0;
  if (thrds > 0)
    omp_set_num_threads(thrds);
 
  unsigned int boundary = (l > m) ? m : l;
  unsigned int inv, mult;

  gettimeofday(&start, NULL);
  cStart  = clock();
  
  for (i = 0; i < boundary; ++i) {
    inv = negInverseModP(a[i+i*m]);
#pragma omp parallel shared(a) private(mult)
{
#pragma omp master 
  {
    threadNumber = omp_get_num_threads();
  }
#pragma omp for schedule(dynamic,bs) private(j,k)
    for (j = i+1; j < l; ++j) {
      mult = a[i+j*m] * inv;
      for (k = i+1; k < m; ++k) {
        a[k+j*m]  += a[k+i*m] * mult;
      }
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
  printf("Method:           Open MP\n");
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

int main(int argc, char *argv[]) {
  int opt;
  // default values
  int l = 2000, m = 2000, t=1, bs=1;
  // biggest prime < 2^16

  /* 
  // no arguments given
  */
  if(argc == 1) {
    //fprintf(stderr, "This program needs arguments....\n\n");
    //print_help(1);
  }

  while((opt = getopt(argc, argv, "l:m:t:b:")) != -1) {
    switch(opt) {
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
    }
  }

  elim(l,m,t,bs);

  return 0;
}
