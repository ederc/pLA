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


// cache-oblivious implementation

unsigned int prime      = 65521;
unsigned int blocksize  = 0;
unsigned int *neg_inv_piv;

static void getri(void *descr[], int type) {
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
}

static void gessm_base(void *descr[], __attribute__((unused)) void *arg) {
  getri(descr, STARPU_CPU);
}

struct starpu_codelet gessm_cl = {
  .type             = STARPU_SEQ,
  .max_parallelism  = INT_MAX,
  .where            = STARPU_CPU|STARPU_CUDA,
  .cpu_funcs        = {gessm_base, NULL},
  .nbuffers         = 1,
  .modes            = {STARPU_R, STARPU_RW}
};


static void trsti(void *descr[], int type) {
}

static void trsti_base(void *descr[], __attribute__((unused)) void *arg) {
  trsti(descr, STARPU_CPU);
}

struct starpu_codelet trsti_cl = {
  .type             = STARPU_SEQ,
  .max_parallelism  = INT_MAX,
  .where            = STARPU_CPU|STARPU_CUDA,
  .cpu_funcs        = {trsti_base, NULL},
  .nbuffers         = 1,
  .modes            = {STARPU_R, STARPU_RW}
};


static void ssssm(void *descr[], int type) {
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
    printf("horiz_tiles %u -- vert_tiles %u ===>> k %u\n",rem_horiz_tiles, rem_vert_tiles, nb_tasks);
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
    printf("idx_gessm %u\n", idx_gessm);
    printf("idx_trsti %u\n", idx_trsti);
    printf("idx_ssssm %u\n", idx_ssssm);
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
      printf("k %u -- i+k %u\n",k,i+k);
      tasks[j]->cl  = &trsti_cl;
      tasks[j]->handles[0]  = starpu_data_get_sub_data(a_hdl, 2, i, i);
      tasks[j]->handles[1]  = starpu_data_get_sub_data(a_hdl, 2, i+k, i);
      starpu_task_declare_deps_array(tasks[j], 1, tasks);
    }
    j = idx_ssssm;
    // ssssm depends on all previous defined tasks
    printf("idx_ssssm - idx_trsti = %u\n",idx_ssssm - idx_trsti);
    printf("(nb_horiz_tiles - i) * (nb_vert_tiles - i) = %u\n",(nb_horiz_tiles - i) * (nb_vert_tiles - i));
    //assert(idx_ssssm - idx_trsti == (nb_horiz_tiles - 1 - i) * (nb_vert_tiles - 1 - i));
    for (k = i+1; k < nb_horiz_tiles; ++k) {
      for (l = i+1; l < nb_vert_tiles; ++l) {
    //for (j = idx_trsti; j < idx_ssssm; ++j) {
        printf("j %u\n",j);
        printf("k %u -- l %u\n",k,l);
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
    printf("i %u\n",i);
  
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
  int i, j, k;

  unsigned int *a;
  starpu_malloc((void **)&a, l * m * sizeof(unsigned int));
  
  srand(time(NULL));
  for (i=0; i< l*m; i++) {
    a[i]  = rand();
  }

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
  fl.filter_func  = starpu_matrix_filter_vertical_block;
  fl.nchildren    = nb_vert_tiles;
  fm.filter_func  = starpu_matrix_filter_block;
  fm.nchildren    = nb_horiz_tiles;

  starpu_data_map_filters(a_hdl, 2, &fl, &fm);
  launch_codelets(nb_vert_tiles, nb_horiz_tiles, a_hdl); 

  gettimeofday(&stop, NULL);
  cStop = clock();

  starpu_data_unpartition(a_hdl, 0);
  starpu_data_unregister(a_hdl);

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