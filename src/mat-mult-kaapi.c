#include <math.h>
#include <kaapic.h>
#include <float.h>
#include <getopt.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>


static inline void kaapiMult1D(
    int start, int end, int tid, int m, int n,
    unsigned int *c, const unsigned int *a, const unsigned int *b
                              )
{
  int i, j, k;
  unsigned int sum;
  for (i = start; i < end; ++i) {
    for (j = 0; j < n; ++j) {
      sum = 0;
      for (k = 0; k < m; ++k) {
        sum +=  a[k+i*m] * b[k+j*m];
      }
      c[j+i*m]  = sum;
    }
  }
}

// multiplies A*B^T and stores it in *this
void mult(int l, int m, int n, int thrds, int bs) {

  //C.resize(l*m);
  printf("Matrix Multiplication\n");
  struct timeval start, stop;
  clock_t cStart, cStop;
  int i, j, k;
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
  unsigned sum = 0;

  // kaapic stuff
  int blocksize     = bs;
  int threadNumber  = thrds;
  kaapic_foreach_attr_t attr;
  kaapic_init(1);
  kaapic_foreach_attr_init(&attr);
  kaapic_foreach_attr_set_grains(&attr, blocksize, blocksize);

  gettimeofday(&start, NULL);
  cStart  = clock();

  kaapic_foreach(0, l, &attr, 5, kaapiMult1D, m, n, c, a, b);

  gettimeofday(&stop, NULL);
  cStop = clock();

  kaapic_finalize();
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
  printf("Method:           KAAPIC\n");
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
  printf("       Computes the matrix multiplication of two matrices A and B with\n");
  printf("       unsigned integer entries. It uses the KAAPIC parallel scheduler.\n");

  printf("OPTIONS\n");
  printf("       -b        block- resp. chunksize\n");
  printf("                 default: 64\n");
  printf("       -h        print help\n");
  printf("       -l        row size of matrix A\n");
  printf("                 default: 2000\n");
  printf("       -m        column size of matrix A and row size of matrix B\n");
  printf("                 default: 2000\n");
  printf("       -n        column size of matrix B\n");
  printf("                 default: 2000\n");
  printf("       -t        number of threads\n");
  printf("                 default: 1\n");
  printf("                 Note that you have to put 'KAAPI_CPUCOUNT=value of t you want'\n"); 
  printf("                 in front of your call, otherwise the number of cores is not set\n");

  exit(exval);
}

int main(int argc, char *argv[]) {
  int opt;
  // default values
  int l = 2000, m = 2000, n = 2000, t=1, bs=64;
  // biggest prime < 2^16

  /* 
  // no arguments given
  */
  if(argc == 1) {
    //fprintf(stderr, "This program needs arguments....\n\n");
    //print_help(1);
  }

  while((opt = getopt(argc, argv, "hl:m:n:t:b:")) != -1) {
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
    }
  }

  mult(l,m,n,t,bs);

  return 0;
}
