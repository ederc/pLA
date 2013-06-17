#include <math.h>
#include <pthread.h>
#include <getopt.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>

#include "mat-elim-tools.h"

struct paramsElim {
  unsigned int *a;
  int size;
  unsigned int inv;
  unsigned int prime;
  unsigned int index;
  unsigned int start;
  unsigned int m;
  unsigned int n;
  int tid;
};

struct paramsCoElim {
  unsigned int *M;
  unsigned int* neg_inv_piv;
  unsigned int size;
  unsigned int  prime;
  unsigned int rows;
  unsigned int cols;
  unsigned int blocksize;
  unsigned int i1;
  unsigned int i2;
  unsigned int j1;
  unsigned int j2;
  unsigned int k1;
  unsigned int k2;
  int nthrds;
};


void *elimPTHRD(void *p) {
  struct paramsElim *_p = (struct paramsElim *)p;
  int start           = _p->start + _p->index;
  int end             = start + _p->size;
  int n               = _p->n;
  int i               = _p->index;
  unsigned int prime  = _p->prime;
  unsigned int inv    = _p->inv;
  unsigned int mult;
  int j, k;
  for (j = start+1; j < end+1; ++j) {
    mult  = (_p->a[i+j*n] * inv);// % prime;
    for (k = i+1; k < n; ++k) {
      _p->a[k+j*n]  +=  _p->a[k+i*n] * mult;
      //_p->a[k+j*n]  %=  prime;
    }
  }
  return 0;
}

// multiplies A*B^T and stores it in *this
void elim(int l,int m,int thrds,int bs) {

  printf("Naive Gaussian Elimination\n");
  struct timeval start, stop;
  clock_t cStart, cStop;
  int i, j, k, t;
  // holds thread information
  pthread_t threads[thrds];
  struct paramsElim *thread_params = (struct paramsElim *) malloc(thrds * sizeof(struct paramsElim));
  unsigned int chunkSize;
  int threadNumber  = thrds;
  unsigned int pad;
  unsigned int ctr;

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
    chunkSize = (m - i - 1) / thrds;
    pad       = (m - i - 1) % thrds;
    ctr = 0;
    for (t = 0; t < thrds; ++t) {
      thread_params[t].a      = a;
      thread_params[t].prime  = prime;
      thread_params[t].index  = i;
      if (t < pad) {
        thread_params[t].size   = chunkSize + 1;
        thread_params[t].start  = ctr;
        ctr +=  chunkSize + 1;
      } else {
        thread_params[t].size = chunkSize;
        thread_params[t].start  = ctr;
        ctr +=  chunkSize;
      }
      thread_params[t].n    = m;
      thread_params[t].inv  = inv;
      // real computation
      pthread_create(&threads[t], NULL, elimPTHRD, (void *) &thread_params[t]);
    }

    // join threads back again
    for (t = 0; t < thrds; ++t)
      pthread_join(threads[t], NULL);
  }
  free(thread_params);

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
  printf("Method:           pthreads\n");
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
