#include <math.h>
#include <float.h>
#include <getopt.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>
#include <starpu.h>
#include <starpu_data.h>



struct block_conf {
  uint32_t bl;
  uint32_t bm;
  uint32_t bn;
  uint32_t pad;
};

static struct block_conf conf __attribute__ ((aligned (128)));
static unsigned niter = 10;
// number of elements per slice in l resp. n direction
static unsigned nslicesl = 1<<0;
static unsigned nslicesn = 1<<0;

void callback_func(void *callback_arg) {
  printf("Callback function (arg %x)\n", callback_arg);
}

static void starpu_gemm_cpu(void *descr[], int type) {
  unsigned int i, j, k;
  unsigned int *sub_a = (unsigned int *)STARPU_MATRIX_GET_PTR(descr[0]);
  unsigned int *sub_b = (unsigned int *)STARPU_MATRIX_GET_PTR(descr[1]);
  unsigned int *sub_c = (unsigned int *)STARPU_MATRIX_GET_PTR(descr[2]);
  /*
  unsigned int nc1 = STARPU_MATRIX_GET_NX(descr[1]);
  unsigned int lc1 = STARPU_MATRIX_GET_NY(descr[1]);
  unsigned int xxa = STARPU_MATRIX_GET_NX(descr[0]);
  unsigned int yya = STARPU_MATRIX_GET_NY(descr[0]);
  unsigned int xxb = STARPU_MATRIX_GET_NX(descr[1]);
  unsigned int yyb = STARPU_MATRIX_GET_NY(descr[1]);
  unsigned int xxc = STARPU_MATRIX_GET_NX(descr[2]);
  unsigned int yyc = STARPU_MATRIX_GET_NY(descr[2]);
  unsigned int la1 = STARPU_MATRIX_GET_NY(descr[0]);
  unsigned int offa = STARPU_MATRIX_GET_OFFSET(descr[0]);
  unsigned int offb = STARPU_MATRIX_GET_OFFSET(descr[1]);
  //unsigned int nc = nc1 / STARPU_MATRIX_GET_LD(descr[2])/nslicesl;
  //unsigned int lc = lc1 / STARPU_MATRIX_GET_LD(descr[1])/nslicesn;
  //unsigned int la = la1 / STARPU_MATRIX_GET_LD(descr[0]);
  printf("li %u\n",li);
  printf("lj %u\n",lj);
  printf("lk %u\n",lk);
  printf("nc1 %u\n",nc1);
  printf("lc1 %u\n",lc1);
  printf("xxa %u\n",xxa);
  printf("yya %u\n",yya);
  printf("xxb %u\n",xxb);
  printf("yyb %u\n",yyb);
  printf("xxc %u\n",xxc);
  printf("yyc %u\n",yyc);
  printf("offa %u\n",offa);
  printf("offb %u\n",offb);
  printf("la1 %u\n",la1);
  li = 0;
  lj = 0;
  */
  unsigned int li = STARPU_MATRIX_GET_LD(descr[1])/nslicesl;
  unsigned int lj = STARPU_MATRIX_GET_LD(descr[1])/nslicesn;
  unsigned int la = STARPU_MATRIX_GET_LD(descr[0]);
  unsigned int lk = la;
  unsigned int sum;
  int worker_size = starpu_combined_worker_get_size();
  int rank        = starpu_combined_worker_get_rank();
  if (worker_size == 1) {
        sum = 0;
        for (k = 0; k < lk; ++k) {
          //printf("k %u\n",k);
          sum += sub_a[k] * sub_b[k];
          //printf("sum[%u] %u = sub_a[%u] %u * sub_b[%u] %u\n",
          //    0, sum, k, sub_a[k], k, sub_b[k]);
        }
        sub_c[0] += sum;
    //printf("TASK DONE\n");
  } else {
    printf("parallel workers\n");
    int rank        = starpu_combined_worker_get_rank();
    int block_size  = (li + worker_size -1)/worker_size;
    int new_li      = STARPU_MIN(li, block_size * (rank + 1)) - block_size * rank;
    li              = new_li;
    STARPU_ASSERT(li = STARPU_MATRIX_GET_NY(descr[1]));
    sum = 0;
    for (k = 0; k < lk; ++k) {
      //printf("k %u\n",k);
      sum += sub_a[k] * sub_b[k];
      //printf("sum[%u] %u = sub_a[%u] %u * sub_b[%u] %u\n",
      //    j+i*la, sum, k+i*la, sub_a[k+i*la], k+j*la, sub_b[k+j*la]);
    }
    sub_c[0] += sum;
  }
}

static void cpu_mult(void *descr[], __attribute__((unused)) void *arg) {
  starpu_gemm_cpu(descr, STARPU_CPU);
}

struct starpu_codelet cl = {
  .type             = STARPU_SPMD,
  .max_parallelism  = 64,
  .where            = STARPU_CPU|STARPU_CUDA,
  .cpu_funcs        = {cpu_mult, NULL},
  .nbuffers         = 3,
  .modes            = {STARPU_R, STARPU_R, STARPU_RW}
};

static void launch_codelets(int l, int m, int n,
    starpu_data_handle_t a_hdl, starpu_data_handle_t b_hdl,
    starpu_data_handle_t c_hdl) {
  int i, j, k, ret;
  int maxk  = m/16;
  int maxi  = nslicesl;
  int maxj  = nslicesn;
  for (k = 0; k < maxk; k++) {
    for (j = 0; j < l; ++j) {
      for (i = 0; i < m; ++i) {
        struct starpu_task *task  = starpu_task_create();

        task->cl          = &cl;
        //task->cl_arg      = &conf;
        //task->cl_arg_size = sizeof(struct block_conf);

        //task->callback_func = callback_func;
        //task->callback_arg  = NULL;
        //printf("i %d -- j %d -- k %d\n",i,j,k);
        task->handles[0] = starpu_data_get_sub_data(a_hdl, 2, j, k);
        task->handles[1] = starpu_data_get_sub_data(b_hdl, 2, i, k);
        task->handles[2] = starpu_data_get_sub_data(c_hdl, 2, j, i);
        ret = starpu_task_submit(task);
        if (ret == -ENODEV)
          ret = 77;
        STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
      }
    }
  }
}


static void mult(int l, int m, int n, int thrds, int bs, int sh, int sv) {
  printf("Matrix Multiplication\n");
  nslicesl = sh;
  nslicesn = sv;
  struct timeval start, stop;
  clock_t cStart, cStop;
  int i, j, k;
  // starpu stuff
  int ret;
  int threadNumber  = thrds;
  int blocksize     = bs;
  
  ret = starpu_init(NULL);
  
  unsigned int *a, *b, *c;
  starpu_malloc((void **)&a, l*m*sizeof(unsigned int));
  starpu_malloc((void **)&b, n*m*sizeof(unsigned int));
  starpu_malloc((void **)&c, l*n*sizeof(unsigned int));

  starpu_data_handle_t a_hdl, b_hdl, c_hdl;

  starpu_matrix_data_register(&a_hdl, 0, (uintptr_t)a,
      l, l, m, sizeof(unsigned int));
  starpu_matrix_data_register(&b_hdl, 0, (uintptr_t)b,
      n, n, m, sizeof(unsigned int));
  starpu_matrix_data_register(&c_hdl, 0, (uintptr_t)c,
      l, l, n, sizeof(unsigned int));

  //starpu_data_set_wt_mask(c_hdl, 1<<0);

  //conf.bl = l/l; 
  //conf.bm = m;
  //conf.bn = n;

  struct starpu_data_filter fa, fb, fc1, fc2, fd;
  memset(&fa, 0, sizeof(fa));
  memset(&fb, 0, sizeof(fb));
  memset(&fd, 0, sizeof(fd));
  memset(&fc1, 0, sizeof(fc1));
  memset(&fc2, 0, sizeof(fc2));
  fa.filter_func    = starpu_matrix_filter_vertical_block;
  fa.nchildren      = l;
  fd.filter_func    = starpu_matrix_filter_block;
  fd.nchildren      = m/16;
  fb.filter_func    = starpu_matrix_filter_vertical_block;
  fb.nchildren      = n;
  fc1.filter_func   = starpu_matrix_filter_block;
  fc1.nchildren     = l;
  fc2.filter_func   = starpu_matrix_filter_vertical_block;
  fc2.nchildren     = n;

  //starpu_data_partition(a_hdl, &fa); 
  //starpu_data_partition(b_hdl, &fb);

  starpu_data_map_filters(a_hdl, 2, &fd, &fa);
  starpu_data_map_filters(b_hdl, 2, &fd, &fb);
  starpu_data_map_filters(c_hdl, 2, &fc1, &fc2);

  // fill matrices
  srand(time(NULL));
  for (i=0; i< l*m; i++) {
    //a[i]  = i;
    a[i]  = rand() % 20;
    //printf("a[%d] = %u\n",i,a[i]);
  }
  for (i=0; i< n*m; i++) {
    //b[i]  = 50 + i;
    b[i]  = rand() % 20;
    //printf("b[%d] = %u\n",i,b[i]);
  }
  
  for (i=0; i< n*l; i++) {
    c[i]  = 0;
  }

  gettimeofday(&start, NULL);
  cStart  = clock();

  launch_codelets(l, m, n, a_hdl, b_hdl, c_hdl);
  starpu_task_wait_for_all();

  gettimeofday(&stop, NULL);
  cStop = clock();

  starpu_data_unpartition(c_hdl, 0);
  starpu_data_unregister(c_hdl);

  starpu_shutdown();
  /*
  for (i=0; i< l*n; i++) {
    printf("c[%d] = %u\n",i,c[i]);
  }
  */

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
  printf("Method:           StarPU\n");
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
  printf("       unsigned integer entries. It uses the StarPU parallel scheduler.\n");

  printf("OPTIONS\n");
  printf("       -b SIZE   block- resp. chunksize\n");
  printf("                 default: 1\n");
  printf("       -h        print help\n");
  printf("       -l ROWSA  row size of matrix A\n");
  printf("                 default: 2000\n");
  printf("       -m COLSA  column size of matrix A and row size of matrix B\n");
  printf("                 default: 2000\n");
  printf("       -n COLSB  column size of matrix B\n");
  printf("                 default: 2000\n");
  printf("       -t THRDS  number of threads\n");
  printf("                 default: 1\n");

  exit(exval);
}

int main(int argc, char *argv[]) {
  int opt;
  // default matrix dimensions
  unsigned int l = 2000, m = 2000, n = 2000;
  // default number of threads to be used
  unsigned int t=1;
  // default blocksize
  unsigned int bs=1;
  // default size of horizontal resp. vertical tiles
  unsigned int sh = 0, sv = 0;
  // biggest prime < 2^16

  /* 
  // no arguments given
  */
  if(argc == 1) {
    //fprintf(stderr, "This program needs arguments....\n\n");
    //print_help(1);
  }

  while((opt = getopt(argc, argv, "hl:m:n:t:b:H:V:")) != -1) {
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
      case 'H': 
        sh = atoi(strdup(optarg));
        break;
      case 'V': 
        sv = atoi(strdup(optarg));
        break;
    }
  }

  if (sh == 0)
    sh  = 1;
  if (sv == 0)
    sv  = 1;

  mult(l,m,n,t,bs,sh,sv);

  return 0;
}
