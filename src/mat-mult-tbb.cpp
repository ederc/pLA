#include <math.h>
#include <tbb/tbb.h>
#include <float.h>
#include <getopt.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>


// multiplies A*B^T and stores it in *this
void mult_auto(int l,int m,int n, int thrds, int bs) {

  //C.resize(l*m);
  printf("Matrix Multiplication\n");
  struct timeval start, stop;
  clock_t cStart, cStop;
  int i, j, k;
  // open mp stuff
  int threadNumber  = thrds;
  int blocksize     = bs;

  unsigned int *a   = (unsigned int *)malloc(sizeof(unsigned int) * (l * m));
  unsigned int *b   = (unsigned int *)malloc(sizeof(unsigned int) * (n * m));
  unsigned int *c   = (unsigned int *)malloc(sizeof(unsigned int) * (l * n));
  srand(time(NULL));
  for (i=0; i< l*m; i++) {
    a[i]  = rand();
  }
  for (i=0; i< n*m; i++) {
    b[i]  = rand();
  }
  //unsigned sum = 0;
  if (thrds <= 0)
    thrds  = tbb::task_scheduler_init::default_num_threads();
  tbb::task_scheduler_init init(thrds);
  tbb::affinity_partitioner ap;
  gettimeofday(&start, NULL);
  cStart  = clock();
  tbb::parallel_for(tbb::blocked_range<size_t>(0, l, blocksize),
      [&](const tbb::blocked_range<size_t>& r)
      {
        for( size_t i=r.begin(); i!=r.end(); ++i )
          for( size_t j=0; j!=n; ++j ) {
            unsigned int sum = 0;
            for( size_t k=0; k<m; ++k )
              sum += a[k+i*m] * b[k+j*m];
            c[j+i*n]  = sum;
          }
      });
  gettimeofday(&stop, NULL);
  cStop = clock();
  // compute FLOPS:
  // assume addition and multiplication in the mult kernel are 2 operations
  // done A.nRows() * B.nRows() * B.nCols()

  double flops = 0;
  flops = (double)(2) * l * m * n;
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
  printf("Method:           Intel TBB 1D auto partitioner\n");
  printf("#Threads:         %d\n", threadNumber);
  printf("Blocksize:        %d\n", bs);
  printf("Real time:        %.4f sec\n", realtime);
  printf("CPU time:         %.4f sec\n", cputime);
  if (cputime > epsilon)
    printf("CPU/real time:    %.4f\n", ratio);
  printf("- - - - - - - - - - - - - - - - - - - - - - - - - -\n");
  printf("GFLOPS/sec:       %.4f\n", flops / (1000000000 * realtime));
  printf("---------------------------------------------------\n");
}

void mult_affine(int l,int m,int n, int thrds, int bs) {

  //C.resize(l*m);
  printf("Matrix Multiplication\n");
  struct timeval start, stop;
  clock_t cStart, cStop;
  int i, j, k;
  // open mp stuff
  int threadNumber  = thrds;
  int blocksize     = bs;

  unsigned int *a   = (unsigned int *)malloc(sizeof(unsigned int) * (l * m));
  unsigned int *b   = (unsigned int *)malloc(sizeof(unsigned int) * (n * m));
  unsigned int *c   = (unsigned int *)malloc(sizeof(unsigned int) * (l * n));
  srand(time(NULL));
  for (i=0; i< l*m; i++) {
    a[i]  = rand();
  }
  for (i=0; i< n*m; i++) {
    b[i]  = rand();
  }
  //unsigned sum = 0;
  if (thrds <= 0)
    thrds  = tbb::task_scheduler_init::default_num_threads();
  tbb::task_scheduler_init init(thrds);
  tbb::affinity_partitioner ap;
  gettimeofday(&start, NULL);
  cStart  = clock();
  tbb::parallel_for(tbb::blocked_range<size_t>(0, l, blocksize),
      [&](const tbb::blocked_range<size_t>& r)
      {
        for( size_t i=r.begin(); i!=r.end(); ++i )
          for( size_t j=0; j!=n; ++j ) {
            unsigned int sum = 0;
            for( size_t k=0; k<m; ++k )
              sum += a[k+i*m] * b[k+j*m];
            c[j+i*n]  = sum;
          }
      }, ap);
  gettimeofday(&stop, NULL);
  cStop = clock();
  // compute FLOPS:
  // assume addition and multiplication in the mult kernel are 2 operations
  // done A.nRows() * B.nRows() * B.nCols()

  double flops = 0;
  flops = (double)(2) * l * m * n;
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
  printf("Method:           Intel TBB 1D affinity partitioner\n");
  printf("#Threads:         %d\n", threadNumber);
  printf("Blocksize:        %d\n", bs);
  printf("Real time:        %.4f sec\n", realtime);
  printf("CPU time:         %.4f sec\n", cputime);
  if (cputime > epsilon)
    printf("CPU/real time:    %.4f\n", ratio);
  printf("- - - - - - - - - - - - - - - - - - - - - - - - - -\n");
  printf("GFLOPS/sec:       %.4f\n", flops / (1000000000 * realtime));
  printf("---------------------------------------------------\n");
}

void mult_simple(int l,int m,int n, int thrds, int bs) {

  //C.resize(l*m);
  printf("Matrix Multiplication\n");
  struct timeval start, stop;
  clock_t cStart, cStop;
  int i, j, k;
  // open mp stuff
  int threadNumber  = thrds;
  int blocksize     = bs;

  unsigned int *a   = (unsigned int *)malloc(sizeof(unsigned int) * (l * m));
  unsigned int *b   = (unsigned int *)malloc(sizeof(unsigned int) * (n * m));
  unsigned int *c   = (unsigned int *)malloc(sizeof(unsigned int) * (l * n));
  srand(time(NULL));
  for (i=0; i< l*m; i++) {
    a[i]  = rand();
  }
  for (i=0; i< n*m; i++) {
    b[i]  = rand();
  }
  //unsigned sum = 0;
  if (thrds <= 0)
    thrds  = tbb::task_scheduler_init::default_num_threads();
  tbb::task_scheduler_init init(thrds);
  tbb::simple_partitioner sp;
  gettimeofday(&start, NULL);
  cStart  = clock();
  tbb::parallel_for(tbb::blocked_range<size_t>(0, l, blocksize),
      [&](const tbb::blocked_range<size_t>& r)
      {
        for( size_t i=r.begin(); i!=r.end(); ++i )
          for( size_t j=0; j!=n; ++j ) {
            unsigned int sum = 0;
            for( size_t k=0; k<m; ++k )
              sum += a[k+i*m] * b[k+j*m];
            c[j+i*n]  = sum;
          }
      });
  gettimeofday(&stop, NULL);
  cStop = clock();
  // compute FLOPS:
  // assume addition and multiplication in the mult kernel are 2 operations
  // done A.nRows() * B.nRows() * B.nCols()

  double flops = 0;
  flops = (double)(2) * l * m * n;
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
  printf("Method:           Intel TBB 1D simple partitioner\n");
  printf("#Threads:         %d\n", threadNumber);
  printf("Blocksize:        %d\n", bs);
  printf("Real time:        %.4f sec\n", realtime);
  printf("CPU time:         %.4f sec\n", cputime);
  if (cputime > epsilon)
    printf("CPU/real time:    %.4f\n", ratio);
  printf("- - - - - - - - - - - - - - - - - - - - - - - - - -\n");
  printf("GFLOPS/sec:       %.4f\n", flops / (1000000000 * realtime));
  printf("---------------------------------------------------\n");
}

void mult_2d_simple(int l,int m,int n, int thrds, int bs) {

  //C.resize(l*m);
  printf("Matrix Multiplication\n");
  struct timeval start, stop;
  clock_t cStart, cStop;
  int i, j, k;
  // open mp stuff
  int threadNumber  = thrds;
  int blocksize     = bs;

  unsigned int *a   = (unsigned int *)malloc(sizeof(unsigned int) * (l * m));
  unsigned int *b   = (unsigned int *)malloc(sizeof(unsigned int) * (n * m));
  unsigned int *c   = (unsigned int *)malloc(sizeof(unsigned int) * (l * n));
  srand(time(NULL));
  for (i=0; i< l*m; i++) {
    a[i]  = rand();
  }
  for (i=0; i< n*m; i++) {
    b[i]  = rand();
  }
  //unsigned sum = 0;
  if (thrds <= 0)
    thrds  = tbb::task_scheduler_init::default_num_threads();
  tbb::task_scheduler_init init(thrds);
  tbb::simple_partitioner sp;
  gettimeofday(&start, NULL);
  cStart  = clock();
  tbb::parallel_for(tbb::blocked_range2d<size_t>(0, l, blocksize, 0, n, blocksize),
      [&](const tbb::blocked_range2d<size_t>& r)
      {
        for( size_t i=r.rows().begin(); i!=r.rows().end(); ++i )
          for( size_t j=r.cols().begin(); j!=r.cols().end(); ++j ) {
            unsigned int sum = 0;
            for( size_t k=0; k<m; ++k )
              sum += a[k+i*m] * b[k+j*m];
            c[j+i*n]  = sum;
          }
      });
  gettimeofday(&stop, NULL);
  cStop = clock();
  // compute FLOPS:
  // assume addition and multiplication in the mult kernel are 2 operations
  // done A.nRows() * B.nRows() * B.nCols()

  double flops = 0;
  flops = (double)(2) * l * m * n;
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
  printf("Method:           Intel TBB 2D simple partitioner\n");
  printf("#Threads:         %d\n", threadNumber);
  printf("Blocksize:        %d\n", bs);
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
  printf("       Computes the matrix multiplication of two matrices A and B with\n");
  printf("       unsigned integer entries. It uses the Open MP parallel scheduler.\n");

  printf("OPTIONS\n");
  printf("       -b        block- resp. chunksize\n");
  printf("                 default: 1\n");
  printf("       -h        print help\n");
  printf("       -l        row size of matrix A\n");
  printf("                 default: 2000\n");
  printf("       -m        column size of matrix A and row size of matrix B\n");
  printf("                 default: 2000\n");
  printf("       -n        column size of matrix B\n");
  printf("                 default: 2000\n");
  printf("       -t        number of threads\n");
  printf("                 default: 1\n");
  printf("       -v        version of the partitioner you want\n");
  printf("                 0 = auto partitioner\n");
  printf("                 1 = affinity partitioner\n");
  printf("                 2 = simple partitioner\n");
  printf("                 3 = 2D simple partitioner\n");
  printf("                 default: 0\n");

  exit(exval);
}

int main(int argc, char *argv[]) {
  int opt;
  // default values
  int l = 2000, m = 2000, n = 2000, t=1, bs=1, v=0;
  // biggest prime < 2^16

  /* 
  // no arguments given
  */
  if(argc == 1) {
    //fprintf(stderr, "This program needs arguments....\n\n");
    //print_help(1);
  }

  while((opt = getopt(argc, argv, "hl:m:n:t:b:v:")) != -1) {
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
      case 'n': 
        n = atoi(strdup(optarg));
        break;
      case 't': 
        t = atoi(strdup(optarg));
        break;
      case 'b': 
        bs = atoi(strdup(optarg));
        break;
      case 'v': 
        v = atoi(strdup(optarg));
        break;
    }
  }
  switch(v) {
    case 0:
      mult_auto(l,m,n,t,bs);
      break;
    case 1:
      mult_affine(l,m,n,t,bs);
      break;
    case 2:
      mult_simple(l,m,n,t,bs);
      break;
    case 3:
      mult_2d_simple(l,m,n,t,bs);
      break;
  }

  return 0;
}
