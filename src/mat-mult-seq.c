#include <math.h>
#include <float.h>
#include <getopt.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>


// multiplies A*B^T and stores it in *this
void mult(int l,int m,int n) {

  //C.resize(l*m);
  printf("Matrix Multiplication\n");
  struct timeval start, stop;
  clock_t cStart, cStop;
  int i, j, k;
  unsigned int *a = (unsigned int *)malloc(sizeof(unsigned int) * (l * m));
  unsigned int *b = (unsigned int *)malloc(sizeof(unsigned int) * (n * m));
  unsigned int *c = (unsigned int *)malloc(sizeof(unsigned int) * (l * n));
  srand(time(NULL));
  for (i=0; i< l*m; i++) {
    a[i]  = rand();
  }
  for (i=0; i< n*m; i++) {
    b[i]  = rand();
  }
  unsigned int sum = 0;
  
  gettimeofday(&start, NULL);
  cStart  = clock();
  for (i = 0; i < l; ++i) {
    for (j = 0; j < n; ++j) {
      sum = 0;
      for (k = 0; k < m; k++) {
        sum += a[k+i*m] * b[k+j*m];
      }
      c[j+i*m] = sum;
    }
  }
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
  int l = 2000, m = 2000, n = 2000;
  // biggest prime < 2^16

  /* 
  // no arguments given
  */
  if(argc == 1) {
    //fprintf(stderr, "This program needs arguments....\n\n");
    //print_help(1);
  }

  while((opt = getopt(argc, argv, "l:m:n:")) != -1) {
    switch(opt) {
      case 'l': 
        l = atoi(strdup(optarg));
        break;
      case 'm': 
        m = atoi(strdup(optarg));
        break;
      case 'n': 
        n = atoi(strdup(optarg));
        break;
    }
  }

  mult(l,m,n);

  return 0;
}
