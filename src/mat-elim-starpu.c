#include <math.h>
#include <omp.h>
#include <getopt.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>
#include <starpu.h>
#include <starpu_data.h>

#include "include/pla-config.h"
#include "mat-elim-tools.h"

#define DEBUG 0
#define CHECK_RESULT 1
// cache-oblivious implementation

unsigned int prime      = 32003;
unsigned int blocksize  = 0;
unsigned int *neg_inv_piv;

// multiplies A*B^T and stores it in *this
void elim(unsigned int *a, int l,int m) {

  //C.resize(l*m);
  printf("Naive Gaussian Elimination\n");
  struct timeval start, stop;
  clock_t cStart, cStop;
  int i, j, k;
  unsigned int sum = 0;
 
  unsigned int boundary = (l > m) ? m : l;
  unsigned int inv, mult;

  gettimeofday(&start, NULL);
  cStart  = clock();
  
  for (i = 0; i < boundary; ++i) {
    inv = negInverseModP(a[i+i*m], prime);
    for (j = i+1; j < l; ++j) {
      mult = a[i+j*m] * inv;
      mult %= prime;
      for (k = i+1; k < m; ++k) {
        a[k+j*m]  += a[k+i*m] * mult;
        a[k+j*m]  %= prime;
      }
    }
  }
  printf("--------------------------------------------------------\n");
  printf("SEQ RESULTS\n");
  printf("--------------------------------------------------------\n");
  for (i=0; i<l; ++i) {
    for (j=i; j<m; ++j) {
      printf("a[%d] = %u, %d, %d\n", j+i*m, a[j+i*m], i, j);
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


static void getri(void *descr[], int type) {
  // Computes the Echelon form of A, stores the corresponding inverted multiples
  // in the lower part
  // ------------------------------
  // | A |   |   |   |   |   |    | 
  // |----------------------------|
  // |   |   |   |   |   |   |    |
  // |----------------------------|
  // |   |   |   |   |   |   |    |
  // |----------------------------|
  // |   |   |   |   |   |   |    |
  // ------------------------------
  unsigned int i, j, k;
  unsigned int *sub_a   = (unsigned int *)STARPU_MATRIX_GET_PTR(descr[0]);
  unsigned int x_dim    = STARPU_MATRIX_GET_NX(descr[0]);
  unsigned int y_dim    = STARPU_MATRIX_GET_NY(descr[0]);
  unsigned int offset_a = STARPU_MATRIX_GET_OFFSET(descr[0]);
  unsigned int ld_a     = STARPU_MATRIX_GET_LD(descr[0]);
  unsigned int mult     = 0;
 
#if DEBUG
  printf("\n --- GETRI ---\n");
  printf("x_dim = %u\n", x_dim);
  printf("y_dim = %u\n", y_dim);
  printf("ld_a  = %u\n", ld_a);
#endif
  for (i = 0; i < y_dim; ++i) {  
    // compute inverse
    neg_inv_piv[i+offset_a] = negInverseModP(sub_a[i+i*ld_a], prime);
#if DEBUG
    printf("sub_a[%u] = %u\n", i+i*ld_a,sub_a[i+i*ld_a]);
    printf("inv  = %u\n", neg_inv_piv[i+offset_a]);
#endif
    for (j = i+1; j < x_dim; ++j) {
      // multiply by corresponding coeff
      mult  = (neg_inv_piv[i+offset_a] * sub_a[i+j*ld_a]) % prime;
#if DEBUG
      printf("sub_a[%u] = %u\n", i+j*ld_a,sub_a[i+j*ld_a]);
      printf("mult      = %u\n", mult);
#endif
      sub_a[i+j*ld_a] = mult;
      for (k = i+1; k < y_dim; ++k) {
        sub_a[k+j*ld_a] +=  (sub_a[k+i*ld_a] * mult);
        sub_a[k+j*ld_a] %=  prime;
      }
    }
  }  
}

static void getri_base(void *descr[], __attribute__((unused)) void *arg) {
  getri(descr, STARPU_CPU);
}

struct starpu_codelet getri_cl = {
  .type             = STARPU_SEQ,
  .max_parallelism  = INT_MAX,
  .where            = STARPU_CPU|STARPU_CUDA,
  .cpu_funcs        = {getri_base, NULL},
  .nbuffers         = 1,
  .modes            = {STARPU_RW}
};


static void gessm(void *descr[], int type) {
  // B reduces itself using the  corresponding inverted multiples that
  // are already stored in the lower part of A
  // ------------------------------
  // | A |   | B |   |   |   |    | 
  // |----------------------------|
  // |   |   |   |   |   |   |    |
  // |----------------------------|
  // |   |   |   |   |   |   |    |
  // |----------------------------|
  // |   |   |   |   |   |   |    |
  // ------------------------------
  unsigned int i, j, k;
  unsigned int *sub_a   = (unsigned int *)STARPU_MATRIX_GET_PTR(descr[0]);
  unsigned int x_dim_a  = STARPU_MATRIX_GET_NX(descr[0]);
  unsigned int y_dim_a  = STARPU_MATRIX_GET_NY(descr[0]);
  unsigned int ld_a     = STARPU_MATRIX_GET_LD(descr[0]);
  unsigned int *sub_b   = (unsigned int *)STARPU_MATRIX_GET_PTR(descr[1]);
  unsigned int x_dim_b  = STARPU_MATRIX_GET_NX(descr[1]);
  unsigned int y_dim_b  = STARPU_MATRIX_GET_NY(descr[1]);
  unsigned int mult     = 0;
 
#if DEBUG
  printf("\n --- GESSM ---\n");
  printf("ld_a  = %u\n", ld_a);
#endif
  for (i = 0; i < x_dim_a - 1; ++i) {  
    for (j = i+1; j < y_dim_a ; ++j) {  
#if DEBUG
      printf("i %u -- j %u\n",i,j);
      printf("mult      = %u\n", mult);
#endif
      mult  = sub_a[i+j*ld_a];
      for (k = 0; k < x_dim_b; ++k) {
        sub_b[k+j*ld_a] +=  (sub_b[k+i*ld_a] * mult);
        sub_b[k+j*ld_a] %=  prime;
#if DEBUG
        printf("sub_b[%u] = %u\n", k+j*ld_a,sub_b[k+j*ld_a]);
#endif
      }
    }
  }
}

static void gessm_base(void *descr[], __attribute__((unused)) void *arg) {
  gessm(descr, STARPU_CPU);
}

struct starpu_codelet gessm_cl = {
  .type             = STARPU_SEQ,
  .max_parallelism  = INT_MAX,
  .where            = STARPU_CPU|STARPU_CUDA,
  .cpu_funcs        = {gessm_base, NULL},
  .nbuffers         = 2,
  .modes            = {STARPU_R, STARPU_RW}
};


static void trsti(void *descr[], int type) {
  // B gets reduced by A (already in Echelon form), corresponding inverted
  // multiples are stored in B
  // ------------------------------
  // | A |   |   |   |   |   |    | 
  // |----------------------------|
  // |   |   |   |   |   |   |    |
  // |----------------------------|
  // | B |   |   |   |   |   |    |
  // |----------------------------|
  // |   |   |   |   |   |   |    |
  // ------------------------------
  unsigned int i, j, k;
  unsigned int *sub_a   = (unsigned int *)STARPU_MATRIX_GET_PTR(descr[0]);
  unsigned int x_dim_a  = STARPU_MATRIX_GET_NX(descr[0]);
  unsigned int y_dim_a  = STARPU_MATRIX_GET_NY(descr[0]);
  unsigned int ld_a     = STARPU_MATRIX_GET_LD(descr[0]);
  unsigned int offset_a = STARPU_MATRIX_GET_OFFSET(descr[0]);
  unsigned int *sub_b   = (unsigned int *)STARPU_MATRIX_GET_PTR(descr[1]);
  unsigned int x_dim_b  = STARPU_MATRIX_GET_NX(descr[1]);
  unsigned int y_dim_b  = STARPU_MATRIX_GET_NY(descr[1]);
  unsigned int mult     = 0;
  
#if DEBUG
  printf("\n --- TRSTI ---\n");
  printf("x_dim_a = %u\n", x_dim_a);
  printf("y_dim_a = %u\n", y_dim_a);
  printf("ld_a  = %u\n", ld_a);
#endif
  for (i = 0; i < x_dim_a; ++i) {  
    // compute inverse
#if DEBUG
    printf("sub_a[%u] = %u\n", i+i*ld_a,sub_a[i+i*ld_a]);
    printf("inv  = %u\n", neg_inv_piv[i+offset_a]);
#endif
    for (j = 0; j < y_dim_b; ++j) {
      // multiply by corresponding coeff
      mult  = (neg_inv_piv[i+offset_a] * sub_b[i+j*ld_a]) % prime;
#if DEBUG
      printf("sub_b[%u] = %u\n", i+j*ld_a,sub_b[i+j*ld_a]);
      printf("mult      = %u\n", mult);
#endif
      sub_b[i+j*ld_a] = mult;
#if DEBUG
      printf("<> sub_b[%u] = %u\n", i+j*ld_a,sub_b[i+j*ld_a]);
#endif
      for (k = i+1; k < x_dim_b; ++k) {
        sub_b[k+j*ld_a] +=  (sub_a[k+i*ld_a] * mult);
        sub_b[k+j*ld_a] %=  prime;
      }
    }
  }  
 
}

static void trsti_base(void *descr[], __attribute__((unused)) void *arg) {
  trsti(descr, STARPU_CPU);
}

struct starpu_codelet trsti_cl = {
  .type             = STARPU_SEQ,
  .max_parallelism  = INT_MAX,
  .where            = STARPU_CPU|STARPU_CUDA,
  .cpu_funcs        = {trsti_base, NULL},
  .nbuffers         = 2,
  .modes            = {STARPU_R, STARPU_RW}
};


static void ssssm(void *descr[], int type) {
  // C = C + A * B where A consists of all corresponding inverted multiples
  // ------------------------------
  // | \ |   |   | B |   |   |    | 
  // |----------------------------|
  // |   |   |   |   |   |   |    |
  // |----------------------------|
  // | A |   |   | C |   |   |    |
  // |----------------------------|
  // |   |   |   |   |   |   |    |
  // ------------------------------
  unsigned int i, j, k;
  unsigned int *sub_a   = (unsigned int *)STARPU_MATRIX_GET_PTR(descr[0]);
  unsigned int x_dim_a  = STARPU_MATRIX_GET_NX(descr[0]);
  unsigned int y_dim_a  = STARPU_MATRIX_GET_NY(descr[0]);
  unsigned int ld_a     = STARPU_MATRIX_GET_LD(descr[0]);
  unsigned int offset_a = STARPU_MATRIX_GET_OFFSET(descr[0]);
  unsigned int *sub_b   = (unsigned int *)STARPU_MATRIX_GET_PTR(descr[1]);
  unsigned int x_dim_b  = STARPU_MATRIX_GET_NX(descr[1]);
  unsigned int y_dim_b  = STARPU_MATRIX_GET_NY(descr[1]);
  unsigned int *sub_c   = (unsigned int *)STARPU_MATRIX_GET_PTR(descr[2]);
  unsigned int x_dim_c  = STARPU_MATRIX_GET_NX(descr[2]);
  unsigned int y_dim_c  = STARPU_MATRIX_GET_NY(descr[2]);
  unsigned int mult     = 0;

  assert(x_dim_b == x_dim_c);
  assert(y_dim_a == y_dim_c);

#if DEBUG
  printf("\n --- SSSSM ---\n");
  printf("x_dim_a = %u\n", x_dim_a);
  printf("y_dim_a = %u\n", y_dim_a);
  printf("ld_a  = %u\n", ld_a);
  printf("offset_a  = %u\n", offset_a);
#endif
  for (i = 0; i < x_dim_a; ++i) {  
    // compute inverse
#if DEBUG
    printf("sub_a[%u] = %u\n", i+i*ld_a,sub_a[i+i*ld_a]);
#endif
    for (j = 0; j < y_dim_a; ++j) {
      // multiply by corresponding coeff
      for (k = 0; k < x_dim_b; ++k) {
#if DEBUG
        printf("sub_c[%u] = %u\n", k+j*ld_a,sub_c[k+j*ld_a]);
        printf("sub_a[%u] = %u\n", i+j*ld_a,sub_a[i+j*ld_a]);
        printf("sub_b[%u] = %u\n", k+i*ld_a,sub_b[k+i*ld_a]);
#endif
        sub_c[k+j*ld_a] +=  (sub_a[i+j*ld_a] * sub_b[k+i*ld_a]) ;
        sub_c[k+j*ld_a] %=  prime;
#if DEBUG
        printf("-- sub_c[%u] = %u\n", k+j*ld_a,sub_c[k+j*ld_a]);
#endif
      }
    }
  }  
}

static void ssssm_base(void *descr[], __attribute__((unused)) void *arg) {
  ssssm(descr, STARPU_CPU);
}

struct starpu_codelet ssssm_cl = {
  .type             = STARPU_SEQ,
  .max_parallelism  = INT_MAX,
  .where            = STARPU_CPU|STARPU_CUDA,
  .cpu_funcs        = {ssssm_base, NULL},
  .nbuffers         = 3,
  .modes            = {STARPU_R, STARPU_R, STARPU_RW}
};


static void launch_codelets(unsigned int nb_vert_tiles,
    unsigned int nb_horiz_tiles, starpu_data_handle_t a_hdl) {
  
  unsigned int i, j, k, l, idx_gessm, idx_trsti, idx_ssssm, ret, nb_tasks;

  // keep track of the number of remaining vertical and horizontal tiles
  unsigned int rem_horiz_tiles  = nb_horiz_tiles;
  unsigned int rem_vert_tiles   = nb_vert_tiles;

  // create tasks needed for the first outer loop
  // => only need to remove some of them for each inner looping
  //    no recreation of tasks during the computation

  // only 1 getri_task is needed at a time
  // nb_horiz_tiles - 1 gessm_tasks are needed at the beginning
  // nb_vert_tiles - 1 tstri_tasks are needed at the beginning
  // (nb_vert_tiles - 1) * (nb_horiz_tiles -1) ssssm_tasks are needed at the beginning
  // => 1 + (nb_horiz_tiles - 1) + (nb_vert_tiles - 1) + ((nb_vert_tiles - 1) *
  //    (nb_horiz_tiles - 1))
  //    =
  //    nb_horiz_tiles * nb_vert_tiles
  // first do GETRI
  for (i = 0; i < nb_vert_tiles; ++i) {
    nb_tasks = rem_horiz_tiles * rem_vert_tiles;
#if DEBUG
    printf("horiz_tiles %u -- vert_tiles %u ===>> k %u\n",rem_horiz_tiles, rem_vert_tiles, nb_tasks);
#endif
    struct starpu_task **tasks;
    starpu_malloc((void **)&tasks, nb_tasks * sizeof(struct starpu_task *));
    for (j = 0; j < nb_tasks; ++j)
      tasks[j] = starpu_task_create();

    idx_gessm = 1;
    idx_trsti = rem_horiz_tiles;
    idx_ssssm = rem_horiz_tiles + rem_vert_tiles - 1;

    // tasks[0] is always GETRI
    tasks[0]->cl          = &getri_cl;
    tasks[0]->handles[0]  = starpu_data_get_sub_data(a_hdl, 2, i, i);
    
    // index of last ssssm entry in tasks array is 
    // idx_trsti + ((rem_vert_tiles - 1) * (rem_horiz_tiles) - 1) - 1
#if DEBUG
    printf("idx_gessm %u\n", idx_gessm);
    printf("idx_trsti %u\n", idx_trsti);
    printf("idx_ssssm %u\n", idx_ssssm);
#endif
    // declare dependencies of the tasks
    k = 0;
    for (j = idx_gessm; j < rem_horiz_tiles; ++j) {
      k++;
      tasks[j]->cl          = &gessm_cl;
      tasks[j]->handles[0]  = starpu_data_get_sub_data(a_hdl, 2, i, i);
      tasks[j]->handles[1]  = starpu_data_get_sub_data(a_hdl, 2, i, i+k);
      starpu_task_declare_deps_array(tasks[j], 1, tasks);
    }
    k = 0;
    for (j = idx_trsti; j < rem_vert_tiles + idx_trsti - 1; ++j) {
      k++;
#if DEBUG
      printf("k %u -- i+k %u\n",k,i+k);
#endif
      tasks[j]->cl  = &trsti_cl;
      tasks[j]->handles[0]  = starpu_data_get_sub_data(a_hdl, 2, i, i);
      tasks[j]->handles[1]  = starpu_data_get_sub_data(a_hdl, 2, i+k, i);
      starpu_task_declare_deps_array(tasks[j], 1, tasks);
    }
    j = idx_ssssm;
    // ssssm depends on all previous defined tasks
#if DEBUG
    printf("idx_ssssm - idx_trsti = %u\n",idx_ssssm - idx_trsti);
    printf("(nb_horiz_tiles - i) * (nb_vert_tiles - i) = %u\n",(nb_horiz_tiles - i) * (nb_vert_tiles - i));
#endif
    for (k = i+1; k < nb_horiz_tiles; ++k) {
      for (l = i+1; l < nb_vert_tiles; ++l) {
#if DEBUG
        printf("j %u\n",j);
        printf("k %u -- l %u\n",k,l);
#endif
        tasks[j]->cl  = &ssssm_cl;
        tasks[j]->handles[0]  = starpu_data_get_sub_data(a_hdl, 2, k, i);
        tasks[j]->handles[1]  = starpu_data_get_sub_data(a_hdl, 2, i, l);
        tasks[j]->handles[2]  = starpu_data_get_sub_data(a_hdl, 2, k, l);
        starpu_task_declare_deps_array(tasks[j], idx_trsti, tasks);
        j++;
      }
    }

    // submit all tasks
    for (j = 0; j < nb_tasks; ++j) {
      ret = starpu_task_submit(tasks[j]);
    }
  
    rem_vert_tiles--;
    rem_horiz_tiles--;
    if (ret == -ENODEV)
      ret = 77;
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
    starpu_task_wait_for_all();
  }

}

static inline unsigned int get_tile_size() {
  return (unsigned int) floor(sqrt((double)__PLA_CPU_L1_CACHE / (8*3)));
}

void elim_co(int l,int m, int thrds, int bs) {

  int ret = starpu_init(NULL);
  int threadNumber  = starpu_worker_get_count();
  
  // compute size of tiles/slices for matrix
  unsigned int tile_size;
  if (bs == 0)
    tile_size = get_tile_size();
  else
    tile_size = bs;

  if (l < tile_size || m < tile_size)
    tile_size = 2;

  struct timeval start, stop;
  clock_t cStart, cStop;
  unsigned int i, j, k;
  unsigned int boundary = (l>m) ? m : l;
  starpu_malloc((void **)&neg_inv_piv, boundary * sizeof(unsigned int));

  unsigned int *a;
  starpu_malloc((void **)&a, l * m * sizeof(unsigned int));
  unsigned int *b;
  starpu_malloc((void **)&b, l * m * sizeof(unsigned int));
  
  srand(time(NULL));
  for (i=0; i< l*m ; i++) {
    //a[i+l*m/2]  = 100 - i;
    a[i]  = rand() % prime;
#if CHECK_RESULT
    b[i]  = a[i];
    printf("a[%u] = %u\n", i, a[i]);
    printf("b[%u] = %u\n", i, b[i]);
#endif
  }
#if CHECK_RESULT
  elim(b, l, m);
#endif

  printf("Cache-oblivious Gaussian Elimination\n");
  
  ret = starpu_init(NULL);

  starpu_data_handle_t a_hdl;
  starpu_matrix_data_register(&a_hdl, 0, (uintptr_t)a,
      l, l, m, sizeof(unsigned int));

  struct starpu_data_filter fl, fm;
  memset(&fl, 0, sizeof(fl));
  memset(&fm, 0, sizeof(fm));
  unsigned int nb_vert_tiles  = l/tile_size;
  unsigned int nb_horiz_tiles = m/tile_size;
  fm.filter_func  = starpu_matrix_filter_vertical_block;
  fm.nchildren    = nb_vert_tiles;
  fl.filter_func  = starpu_matrix_filter_block;
  fl.nchildren    = nb_horiz_tiles;

  starpu_data_map_filters(a_hdl, 2, &fm, &fl);
  launch_codelets(nb_vert_tiles, nb_horiz_tiles, a_hdl); 

  gettimeofday(&stop, NULL);
  cStop = clock();

#if DEBUG
  printf("--------------------------------------------------------\n");
  printf("STARPU RESULTS\n");
  printf("--------------------------------------------------------\n");
  for (i=0; i<l; ++i) {
    for (j=i; j<m; ++j) {
      printf("a[%d] = %u\n", j+i*m, a[j+i*m]);
    }
  }
#endif
#if CHECK_RESULT
  unsigned int ctr  = 0, ctr2 = 0;
  for (i=0; i<l; ++i) {
    for (j=i; j<m; ++j) {
      ctr++;
      if (a[j+i*m] != b[j+i*m]) {
        ctr2++;
        printf("not matchting: a[%d] = %u =/= %u b[%d]\n", j+i*m, a[j+i*m], b[j+i*m], j+i*m);
      }
    }
  }
  printf("%u / %u elements NOT matching\n", ctr2, ctr);
#endif

  //starpu_data_unpartition(a_hdl, 0);
  //starpu_data_unregister(a_hdl);
  starpu_free(a);
  starpu_free(b);
  starpu_shutdown();

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
  printf("Method:           StarPU\n");
  printf("#Threads:         %d\n", threadNumber);
  printf("Chunk size:       %d\n", tile_size);
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
  printf("       It uses Open MP.\n");

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

  exit(exval);
}


int main(int argc, char *argv[]) {
  int opt;
  // default values
  int l = 2000, m = 2000, t=1, bs=0, c=1;
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

  return 0;
}
