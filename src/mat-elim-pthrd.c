#include <math.h>
#include <pthread.h>
#include <getopt.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>

#include "include/pla-config.h"
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

struct thrdPool {
  int maxNumThreads;
  int runningJobs;
  // with the bitmask we know which thread is available:
  // bit set      =>  thread is running another job already
  // bit not set  =>  thread is idle and can be used
  // 64 cores should be enough for the moment
  unsigned long bitmask;
  // pointer to array of threads
  pthread_t *threads;
};

struct thrdData {
  struct thrdPool      *pool;
  struct paramsCoElim  *params;
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
      // if the pivots are in the olumn part of the matrix as Mmdf then we can
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

void* D1(void *p) {
  struct thrdData *_data       = (struct thrdData *)p;
  struct thrdPool *pool        = _data->pool;
  struct paramsCoElim *params  = _data->params;
  unsigned int size     = params->size;
  
  if (params->i2 <= params->k1 || params->j2 <= params->k1)
    return 0;

  if (size <= params->blocksize) {
    base_case(params->M, params->k1, params->i1, params->j1,
              params->rows, params->cols, params->size,
              params->prime, params->neg_inv_piv);
  } else {
    const unsigned int i1 = params->i1;
    const unsigned int i2 = params->i2;
    const unsigned int j1 = params->j1;
    const unsigned int j2 = params->j2;
    const unsigned int k1 = params->k1;
    const unsigned int k2 = params->k2;

    size = size / 2;

    unsigned int km = (k1+k2) / 2 ;
    unsigned int im = (i1+i2) / 2;
    unsigned int jm = (j1+j2) / 2;

    pthread_t thread[4];
    struct paramsCoElim *thread_params = (struct paramsCoElim *)
                                    malloc(4 * sizeof(struct paramsCoElim));
    struct thrdData *data  = (struct thrdData *) malloc(4 * sizeof(struct thrdData));
    for (int i = 0; i < 4; ++i) {
      data[i].pool    = pool;
      data[i].params  = &thread_params[i];
    }

    // get not thread-specific parameters -- once for all
    for (int i = 0; i < 4; ++i) {
      thread_params[i].M            = params->M;
      thread_params[i].neg_inv_piv  = params->neg_inv_piv;
      thread_params[i].size         = size;
      thread_params[i].blocksize    = params->blocksize;
      thread_params[i].nthrds       = params->nthrds;
      thread_params[i].prime        = params->prime;
      thread_params[i].rows         = params->rows;
      thread_params[i].cols         = params->cols;
    }

    // X11
    thread_params[0].i1 = i1;
    thread_params[0].i2 = im;
    thread_params[0].j1 = j1;
    thread_params[0].j2 = jm;
    thread_params[0].k1 = k1;
    thread_params[0].k2 = km;
    // X12
    thread_params[1].i1 = i1;
    thread_params[1].i2 = im;
    thread_params[1].j1 = jm+1;
    thread_params[1].j2 = j2;
    thread_params[1].k1 = k1;
    thread_params[1].k2 = km;
    // X21
    thread_params[2].i1 = im+1;
    thread_params[2].i2 = i2;
    thread_params[2].j1 = j1;
    thread_params[2].j2 = jm;
    thread_params[2].k1 = k1;
    thread_params[2].k2 = km;
    // X22
    thread_params[3].i1 = im+1;
    thread_params[3].i2 = i2;
    thread_params[3].j1 = jm+1;
    thread_params[3].j2 = j2;
    thread_params[3].k1 = k1;
    thread_params[3].k2 = km;

    // parallel - start
    for (int i = 0; i < 4; ++i)
      pthread_create(&thread[i], NULL, &D1, (void *) &data[i]);
    for (int i = 0; i < 4; ++i)
      pthread_join(thread[i], NULL);
    // parallel - end
    // X11
    thread_params[0].i1 = i1;
    thread_params[0].i2 = im;
    thread_params[0].j1 = j1;
    thread_params[0].j2 = jm;
    thread_params[0].k1 = km+1;
    thread_params[0].k2 = k2;
    // X12
    thread_params[1].i1 = i1;
    thread_params[1].i2 = im;
    thread_params[1].j1 = jm+1;
    thread_params[1].j2 = j2;
    thread_params[1].k1 = km+1;
    thread_params[1].k2 = k2;
    // X21
    thread_params[2].i1 = im+1;
    thread_params[2].i2 = i2;
    thread_params[2].j1 = j1;
    thread_params[2].j2 = jm;
    thread_params[2].k1 = km+1;
    thread_params[2].k2 = k2;
    // X22
    thread_params[3].i1 = im+1;
    thread_params[3].i2 = i2;
    thread_params[3].j1 = jm+1;
    thread_params[3].j2 = j2;
    thread_params[3].k1 = km+1;
    thread_params[3].k2 = k2;

    // parallel - start
    for (int i = 0; i < 4; ++i)
      pthread_create(&thread[i], NULL, &D1, (void *) &data[i]);
    for (int i = 0; i < 4; ++i)
      pthread_join(thread[i], NULL);
    // parallel - end
  }
  return 0;
}


void* C1(void *p) {
  struct thrdData *_data       = (struct thrdData *)p;
  struct thrdPool *pool        = _data->pool;
  struct paramsCoElim *params  = _data->params;
  if (params->i2 <= params->k1 || params->j2 <= params->k1)
    return 0;

  //printf("[C] Running jobs: %d\n", pool->runningJobs);
  unsigned int size = params->size;

  if (size <= params->blocksize) {
    base_case(params->M, params->k1, params->i1, params->j1,
              params->rows, params->cols, params->size,
              params->prime, params->neg_inv_piv);
  } else {
    const unsigned int i1 = params->i1;
    const unsigned int i2 = params->i2;
    const unsigned int j1 = params->j1;
    const unsigned int j2 = params->j2;
    const unsigned int k1 = params->k1;
    const unsigned int k2 = params->k2;

    size = size / 2;

    const unsigned int km = (k1+k2) / 2;
    const unsigned int im = (i1+i2) / 2;
    const unsigned int jm = (j1+j2) / 2;

    pthread_t thread[2];
    struct paramsCoElim *thread_params = (struct paramsCoElim *)
                                    malloc(2 * sizeof(struct paramsCoElim));
    struct thrdData *data  = (struct thrdData *) malloc(2 * sizeof(struct thrdData));
    for (int i = 0; i < 2; ++i) {
      data[i].pool    = pool;
      data[i].params  = &thread_params[i];
    }

    // get not thread-specific parameters -- once for all
    for (int i = 0; i < 2; ++i) {
      thread_params[i].M            = params->M;
      thread_params[i].neg_inv_piv  = params->neg_inv_piv;
      thread_params[i].size         = size;
      thread_params[i].blocksize    = params->blocksize;
      thread_params[i].nthrds       = params->nthrds;
      thread_params[i].prime        = params->prime;
      thread_params[i].rows         = params->rows;
      thread_params[i].cols         = params->cols;
    }

    // X11
    thread_params[0].i1 = i1;
    thread_params[0].i2 = im;
    thread_params[0].j1 = j1;
    thread_params[0].j2 = jm;
    thread_params[0].k1 = k1;
    thread_params[0].k2 = km;
    // X21
    thread_params[1].i1 = im+1;
    thread_params[1].i2 = i2;
    thread_params[1].j1 = j1;
    thread_params[1].j2 = jm;
    thread_params[1].k1 = k1;
    thread_params[1].k2 = km;

    // parallel - start
    for (int i = 0; i < 2; ++i)
      pthread_create(&thread[i], NULL, &C1, (void *) &data[i]);
    for (int i = 0; i < 2; ++i)
      pthread_join(thread[i], NULL);
    // parallel - end

    // X12
    thread_params[0].i1 = i1;
    thread_params[0].i2 = im;
    thread_params[0].j1 = jm+1;
    thread_params[0].j2 = j2;
    thread_params[0].k1 = k1;
    thread_params[0].k2 = km;
    // X22
    thread_params[1].i1 = im+1;
    thread_params[1].i2 = i2;
    thread_params[1].j1 = jm+1;
    thread_params[1].j2 = j2;
    thread_params[1].k1 = k1;
    thread_params[1].k2 = km;
    // parallel - start
    for (int i = 0; i < 2; ++i)
      pthread_create(&thread[i], NULL, &D1, (void *) &data[i]);
    for (int i = 0; i < 2; ++i)
      pthread_join(thread[i], NULL);
    // parallel - end

    // X12
    thread_params[0].i1 = i1;
    thread_params[0].i2 = im;
    thread_params[0].j1 = jm+1;
    thread_params[0].j2 = j2;
    thread_params[0].k1 = km+1;
    thread_params[0].k2 = k2;
    // X22
    thread_params[1].i1 = im+1;
    thread_params[1].i2 = i2;
    thread_params[1].j1 = jm+1;
    thread_params[1].j2 = j2;
    thread_params[1].k1 = km+1;
    thread_params[1].k2 = k2;
    // parallel - start
    for (int i = 0; i < 2; ++i)
      pthread_create(&thread[i], NULL, &C1, (void *) &data[i]);
    for (int i = 0; i < 2; ++i)
      pthread_join(thread[i], NULL);
    // parallel - end

    // X11
    thread_params[0].i1 = i1;
    thread_params[0].i2 = im;
    thread_params[0].j1 = j1;
    thread_params[0].j2 = jm;
    thread_params[0].k1 = km+1;
    thread_params[0].k2 = k2;
    // X12
    thread_params[1].i1 = im+1;
    thread_params[1].i2 = i2;
    thread_params[1].j1 = j1;
    thread_params[1].j2 = jm;
    thread_params[1].k1 = km+1;
    thread_params[1].k2 = k2;
    // parallel - start
    for (int i = 0; i < 2; ++i)
      pthread_create(&thread[i], NULL, &D1, (void *) &data[i]);
    for (int i = 0; i < 2; ++i)
      pthread_join(thread[i], NULL);
    // parallel - end
  }
  return 0;
}

void* B1(void *p) {
  struct thrdData *_data       = (struct thrdData *)p;
  struct thrdPool *pool        = _data->pool;
  struct paramsCoElim *params  = _data->params;
  if (params->i2 <= params->k1 || params->j2 <= params->k1)
    return 0;

  //printf("[B] Running jobs: %d\n", pool->runningJobs);
  unsigned int size = params->size;

  if (size <= params->blocksize) {
    base_case(params->M, params->k1, params->i1, params->j1,
              params->rows, params->cols, params->size,
              params->prime, params->neg_inv_piv);
  } else {
    const unsigned int i1 = params->i1;
    const unsigned int i2 = params->i2;
    const unsigned int j1 = params->j1;
    const unsigned int j2 = params->j2;
    const unsigned int k1 = params->k1;
    const unsigned int k2 = params->k2;

    size = size / 2;

    const unsigned int km = (k1+k2) / 2;
    const unsigned int im = (i1+i2) / 2;
    const unsigned int jm = (j1+j2) / 2;

    pthread_t thread[2];
    struct paramsCoElim *thread_params = (struct paramsCoElim *)
                                    malloc(2 * sizeof(struct paramsCoElim));
    struct thrdData *data  = (struct thrdData *) malloc(2 * sizeof(struct thrdData));
    for (int i = 0; i < 2; ++i) {
      data[i].pool    = pool;
      data[i].params  = &thread_params[i];
    }

    // get not thread-specific parameters -- once for all
    for (int i = 0; i < 2; ++i) {
      thread_params[i].M            = params->M;
      thread_params[i].neg_inv_piv  = params->neg_inv_piv;
      thread_params[i].size         = size;
      thread_params[i].blocksize    = params->blocksize;
      thread_params[i].nthrds       = params->nthrds;
      thread_params[i].prime        = params->prime;
      thread_params[i].rows         = params->rows;
      thread_params[i].cols         = params->cols;
    }

    // X11
    thread_params[0].i1 = i1;
    thread_params[0].i2 = im;
    thread_params[0].j1 = j1;
    thread_params[0].j2 = jm;
    thread_params[0].k1 = k1;
    thread_params[0].k2 = km;
    // X12
    thread_params[1].i1 = i1;
    thread_params[1].i2 = im;
    thread_params[1].j1 = jm+1;
    thread_params[1].j2 = j2;
    thread_params[1].k1 = k1;
    thread_params[1].k2 = km;
    // parallel - start
    for (int i = 0; i < 2; ++i)
      pthread_create(&thread[i], NULL, &B1, (void *) &data[i]);
    for (int i = 0; i < 2; ++i)
      pthread_join(thread[i], NULL);
    // parallel - end

    // X21
    thread_params[0].i1 = im+1;
    thread_params[0].i2 = i2;
    thread_params[0].j1 = j1;
    thread_params[0].j2 = jm;
    thread_params[0].k1 = k1;
    thread_params[0].k2 = km;
    // X22
    thread_params[1].i1 = im+1;
    thread_params[1].i2 = i2;
    thread_params[1].j1 = jm+1;
    thread_params[1].j2 = j2;
    thread_params[1].k1 = k1;
    thread_params[1].k2 = km;
    // parallel - start

    for (int i = 0; i < 2; ++i)
      pthread_create(&thread[i], NULL, &D1, (void *) &data[i]);
    for (int i = 0; i < 2; ++i)
      pthread_join(thread[i], NULL);
    // parallel - end

    // X21
    thread_params[0].i1 = im+1;
    thread_params[0].i2 = i2;
    thread_params[0].j1 = j1;
    thread_params[0].j2 = jm;
    thread_params[0].k1 = km+1;
    thread_params[0].k2 = k2;
    // X22
    thread_params[1].i1 = im+1;
    thread_params[1].i2 = i2;
    thread_params[1].j1 = jm+1;
    thread_params[1].j2 = j2;
    thread_params[1].k1 = km+1;
    thread_params[1].k2 = k2;
    // parallel - start

    for (int i = 0; i < 2; ++i)
      pthread_create(&thread[i], NULL, &B1, (void *) &data[i]);
    for (int i = 0; i < 2; ++i)
      pthread_join(thread[i], NULL);
    // parallel - end

    // X11
    thread_params[0].i1 = i1;
    thread_params[0].i2 = im;
    thread_params[0].j1 = j1;
    thread_params[0].j2 = jm;
    thread_params[0].k1 = km+1;
    thread_params[0].k2 = k2;
    // X12
    thread_params[1].i1 = i1;
    thread_params[1].i2 = im;
    thread_params[1].j1 = jm+1;
    thread_params[1].j2 = j2;
    thread_params[1].k1 = km+1;
    thread_params[1].k2 = k2;
    // parallel - start
    for (int i = 0; i < 2; ++i)
      pthread_create(&thread[i], NULL, &D1, (void *) &data[i]);
    for (int i = 0; i < 2; ++i)
      pthread_join(thread[i], NULL);
    // parallel - end
  }
  return 0;
}

void* A(void *p) {
  struct thrdData *_data       = (struct thrdData *)p;
  struct thrdPool *pool        = _data->pool;
  struct paramsCoElim *params  = _data->params;
  if (params->i2 <= params->k1 || params->j2 <= params->k1)
    return 0;

  //printf("[A] Running jobs: %d\n", pool->runningJobs);
  unsigned int size = params->size;

  if (size <= params->blocksize) {
    base_case(params->M, params->k1, params->i1, params->j1,
              params->rows, params->cols, params->size,
              params->prime, params->neg_inv_piv);
  } else {
    const unsigned int i1 = params->i1;
    const unsigned int i2 = params->i2;
    const unsigned int j1 = params->j1;
    const unsigned int j2 = params->j2;
    const unsigned int k1 = params->k1;
    const unsigned int k2 = params->k2;

    size = size / 2;

    const unsigned int km = (k1+k2) / 2;
    const unsigned int im = (i1+i2) / 2;
    const unsigned int jm = (j1+j2) / 2;

    pthread_t thread[2];
    struct paramsCoElim *thread_params = (struct paramsCoElim *)
                                    malloc(2 * sizeof(struct paramsCoElim));
    struct thrdData *data  = (struct thrdData *) malloc(2 * sizeof(struct thrdData));
    for (int i = 0; i < 2; ++i) {
      data[i].pool    = pool;
      data[i].params  = &thread_params[i];
    }

    // get not thread-specific parameters -- once for all
    for (int i = 0; i < 2; ++i) {
      thread_params[i].M            = params->M;
      thread_params[i].neg_inv_piv  = params->neg_inv_piv;
      thread_params[i].size         = size;
      thread_params[i].blocksize    = params->blocksize;
      thread_params[i].nthrds       = params->nthrds;
      thread_params[i].prime        = params->prime;
      thread_params[i].rows         = params->rows;
      thread_params[i].cols         = params->cols;
    }

    // X11
    thread_params[0].i1 = i1;
    thread_params[0].i2 = im;
    thread_params[0].j1 = j1;
    thread_params[0].j2 = jm;
    thread_params[0].k1 = k1;
    thread_params[0].k2 = km;

    // forward step
    A((void *) &data[0]);

    // X12
    thread_params[0].i1 = i1;
    thread_params[0].i2 = im;
    thread_params[0].j1 = jm+1;
    thread_params[0].j2 = j2;
    thread_params[0].k1 = k1;
    thread_params[0].k2 = km;
    // X21
    thread_params[1].i1 = im+1;
    thread_params[1].i2 = i2;
    thread_params[1].j1 = j1;
    thread_params[1].j2 = jm;
    thread_params[1].k1 = k1;
    thread_params[1].k2 = km;

    // parallel - start
    //if (data->pool->runningJobs < data->pool->maxNumThreads)
    pthread_create(&thread[0], NULL, &B1, (void *) &data[0]);
    pthread_create(&thread[1], NULL, &C1, (void *) &data[1]);
    for (int i = 0; i < 2; ++i)
      pthread_join(thread[i], NULL);
    // parallel - end

    // get parameters
    thread_params[0].i1 = im+1;
    thread_params[0].i2 = i2;
    thread_params[0].j1 = jm+1;
    thread_params[0].j2 = j2;
    thread_params[0].k1 = k1;
    thread_params[0].k2 = km;

    D1((void *) &data[0]);

    // backward step

    thread_params[0].i1 = im+1;
    thread_params[0].i2 = i2;
    thread_params[0].j1 = jm+1;
    thread_params[0].j2 = j2;
    thread_params[0].k1 = km+1;
    thread_params[0].k2 = k2;

    A((void *) &data[0]);
  }

  return 0;
}



void elim_co(int l,int m, int thrds, int bs) {

  if (thrds <= 0) {
    thrds  = 1;
  }
  // allocated thread pool
  pthread_t *threads          = (pthread_t *) malloc(thrds * sizeof(pthread_t));
  struct thrdPool *pool       = (struct thrdPool *) malloc(sizeof(struct thrdPool));
  struct paramsCoElim *thread_params = (struct paramsCoElim *)
                                  malloc(sizeof(struct paramsCoElim));
  struct thrdData * data             = (struct thrdData *) malloc(sizeof(struct thrdData));
  data->pool          = pool;
  data->params        = thread_params;
  pool->maxNumThreads = thrds;
  pool->runningJobs   = 0;
  pool->threads       = threads;
  int threadNumber  = thrds;
  //
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

  thread_params->M            = a;
  thread_params->neg_inv_piv  = neg_inv_piv;
  thread_params->blocksize    = blocksize;
  thread_params->nthrds       = thrds;
  thread_params->rows         = l;
  thread_params->cols         = m;
  thread_params->size         = boundary;
  thread_params->prime        = prime;
  thread_params->i1           = 0;
  thread_params->i2           = l - 1;
  thread_params->j1           = 0;
  thread_params->j2           = m - 1;
  thread_params->k1           = 0;
  thread_params->k2           = boundary - 1;


  gettimeofday(&start, NULL);
  cStart  = clock();
  
  A((void *)data);

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
  int l = 2000, m = 2000, t=1, bs=0, c=0;
  // biggest prime < 2^16

  /* 
  // no arguments given
  */
  if(argc == 1) {
    //fprintf(stderr, "This program needs arguments....\n\n");
    //print_help(1);
  }

  while((opt = getopt(argc, argv, "l:m:t:b:c:")) != -1) {
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
      case 'c': 
        c = atoi(strdup(optarg));
        break;
    }
  }

  if (c)
    elim_co(l,m,t,bs);
  else
    elim(l,m,t,bs);

  return 0;
}
