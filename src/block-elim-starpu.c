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

#define DEBUG0 0
#define DEBUG 0
#define CHECK_RESULT 0
// cache-oblivious implementation

typedef unsigned int TYPE;

TYPE *neg_inv_piv;
TYPE *A, *A_saved;
static unsigned prime     = 32003;
static unsigned long l    = 4096;
static unsigned long m    = 4096;
static unsigned nblocks   = 16;
static unsigned check     = 0;
static unsigned pivot     = 0;
static unsigned no_stride = 0;
static unsigned profile   = 0;
static unsigned bound     = 0;
static unsigned bounddeps = 0;
static unsigned boundprio = 0;
static unsigned no_prio   = 0;

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

#define TAG11(k)	((starpu_tag_t)( (1ULL<<60) | (unsigned long long)(k)))
#define TAG12(k,i)	((starpu_tag_t)(((2ULL<<60) | (((unsigned long long)(k))<<32)	\
					| (unsigned long long)(i))))
#define TAG21(k,j)	((starpu_tag_t)(((3ULL<<60) | (((unsigned long long)(k))<<32)	\
					| (unsigned long long)(j))))
#define TAG22(k,i,j)	((starpu_tag_t)(((4ULL<<60) | ((unsigned long long)(k)<<32) 	\
					| ((unsigned long long)(i)<<16)	\
					| (unsigned long long)(j))))

static void parse_args(int argc, char **argv)
{
	int i;
	for (i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-l") == 0)
		{
			char *argptr;
			l = strtol(argv[++i], &argptr, 10);
		}

    else if (strcmp(argv[i], "-m") == 0)
		{
			char *argptr;
			m = strtol(argv[++i], &argptr, 10);
		}

		else if (strcmp(argv[i], "-nblocks") == 0)
		{
			char *argptr;
			nblocks = strtol(argv[++i], &argptr, 10);
		}

		else if (strcmp(argv[i], "-check") == 0)
		{
			check = 1;
		}

		else if (strcmp(argv[i], "-piv") == 0)
		{
			pivot = 1;
		}

		else if (strcmp(argv[i], "-no-stride") == 0)
		{
			no_stride = 1;
		}

		else if (strcmp(argv[i], "-profile") == 0)
		{
			profile = 1;
		}

		else if (strcmp(argv[i], "-bound") == 0)
		{
			bound = 1;
		}
		else if (strcmp(argv[i], "-bounddeps") == 0)
		{
			bound = 1;
			bounddeps = 1;
		}
		else if (strcmp(argv[i], "-bounddepsprio") == 0)
		{
			bound = 1;
			bounddeps = 1;
			boundprio = 1;
		}
		else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
		{
			fprintf(stderr,"usage: lu [-size n] [-nblocks b] [-piv] [-no-stride] [-profile] [-bound] [-bounddeps] [-bounddepsprio]\n");
			exit(0);
		}
	}
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
 
#if DEBUG0
  printf("\n --- GETRI ---\n");
#endif
#if DEBUG
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
      mult  = (neg_inv_piv[i+offset_a] * sub_a[i+j*ld_a]); //% prime;
#if DEBUG
      printf("sub_a[%u] = %u\n", i+j*ld_a,sub_a[i+j*ld_a]);
      printf("mult      = %u\n", mult);
#endif
      sub_a[i+j*ld_a] = mult;
      for (k = i+1; k < y_dim; ++k) {
        sub_a[k+j*ld_a] +=  (sub_a[k+i*ld_a] * mult);
        //sub_a[k+j*ld_a] %=  prime;
      }
    }
  }  
#if DEBUG0
  printf("\n --- GETRI DONE ---\n");
  printf("TASKS READY     %d\n", starpu_task_nready());
  printf("TASKS SUBMITTED %d\n", starpu_task_nsubmitted());
#endif
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
 
#if DEBUG0
  printf("\n --- GESSM ---\n");
#endif
#if DEBUG0
  printf("ld_a  = %u\n", ld_a);
#endif
  for (i = 0; i < x_dim_a - 1; ++i) {  
    for (j = i+1; j < y_dim_a ; ++j) {  
#if DEBUG0
      printf("i %u -- j %u\n",i,j);
#endif
      mult  = sub_a[i+j*ld_a];
      for (k = 0; k < x_dim_b; ++k) {
#if DEBUG0
      printf("mult      = %u\n", mult);
      printf("i %u -- j %u -- k %u\n",i,j, k);
#endif
        sub_b[k+j*ld_a] +=  (sub_b[k+i*ld_a] * mult);
        //sub_b[k+j*ld_a] %=  prime;
#if DEBUG0
          printf("sub_b[%u][%u] = %u\n", j,k,sub_b[k+j*ld_a]);
#endif
      }
    }
  }
#if DEBUG0
  printf("\n --- GESSM DONE ---\n");
  printf("TASKS READY     %d\n", starpu_task_nready());
  printf("TASKS SUBMITTED %d\n", starpu_task_nsubmitted());
#endif
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
  
#if DEBUG0
  printf("\n --- TRSTI ---\n");
#endif
#if DEBUG
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
      mult  = (neg_inv_piv[i+offset_a] * sub_b[i+j*ld_a]);// % prime;
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
        //sub_b[k+j*ld_a] %=  prime;
      }
    }
  }  
#if DEBUG0
  printf("\n --- TRSTI DONE ---\n");
  printf("TASKS READY     %d\n", starpu_task_nready());
  printf("TASKS SUBMITTED %d\n", starpu_task_nsubmitted());
#endif
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

#if DEBUG0
  printf("\n --- SSSSM ---\n");
#endif
#if DEBUG
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
        //sub_c[k+j*ld_a] %=  prime;
#if DEBUG
        printf("-- sub_c[%u] = %u\n", k+j*ld_a,sub_c[k+j*ld_a]);
#endif
      }
    }
  }  
#if DEBUG0
  printf("\n --- SSSSM DONE ---\n");
  printf("TASKS READY     %d\n", starpu_task_nready());
  printf("TASKS SUBMITTED %d\n", starpu_task_nsubmitted());
#endif
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

/*
 *	Construct the DAG
 */

static struct starpu_task *create_task(starpu_tag_t id)
{
	struct starpu_task *task = starpu_task_create();
		task->cl_arg = NULL;

	task->use_tag = 1;
	task->tag_id = id;

	return task;
}

static struct starpu_task *create_task_11(starpu_data_handle_t dataA, unsigned k)
{
/*	printf("task 11 k = %d TAG = %llx\n", k, (TAG11(k))); */

	struct starpu_task *task = create_task(TAG11(k));

	task->cl = &getri_cl;

	/* which sub-data is manipulated ? */
	task->handles[0] = starpu_data_get_sub_data(dataA, 2, k, k);

	/* this is an important task */
	if (!no_prio)
		task->priority = STARPU_MAX_PRIO;

	/* enforce dependencies ... */
	if (k > 0)
	{
		starpu_tag_declare_deps(TAG11(k), 1, TAG22(k-1, k, k));
	}

	return task;
}

static int create_task_12(starpu_data_handle_t dataA, unsigned k, unsigned j)
{
	int ret;

/*	printf("task 12 k,i = %d,%d TAG = %llx\n", k,i, TAG12(k,i)); */

	struct starpu_task *task = create_task(TAG12(k, j));

	task->cl = &gessm_cl;

	/* which sub-data is manipulated ? */
	task->handles[0] = starpu_data_get_sub_data(dataA, 2, k, k);
	task->handles[1] = starpu_data_get_sub_data(dataA, 2, k, j);

	if (!no_prio && (j == k+1))
	{
		task->priority = STARPU_MAX_PRIO;
	}

	/* enforce dependencies ... */
	if (k > 0)
	{
		starpu_tag_declare_deps(TAG12(k, j), 2, TAG11(k), TAG22(k-1, k, j));
	}
	else
	{
		starpu_tag_declare_deps(TAG12(k, j), 1, TAG11(k));
	}

	ret = starpu_task_submit(task);
	if (ret != -ENODEV) STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	return ret;
}

static int create_task_21(starpu_data_handle_t dataA, unsigned k, unsigned i)
{
	int ret;
	struct starpu_task *task = create_task(TAG21(k, i));

	task->cl = &trsti_cl;

	/* which sub-data is manipulated ? */
	task->handles[0] = starpu_data_get_sub_data(dataA, 2, k, k);
	task->handles[1] = starpu_data_get_sub_data(dataA, 2, i, k);

	if (!no_prio && (i == k+1))
	{
		task->priority = STARPU_MAX_PRIO;
	}

	/* enforce dependencies ... */
	if (k > 0)
	{
		starpu_tag_declare_deps(TAG21(k, i), 2, TAG11(k), TAG22(k-1, i, k));
	}
	else
	{
		starpu_tag_declare_deps(TAG21(k, i), 1, TAG11(k));
	}

	ret = starpu_task_submit(task);
	if (ret != -ENODEV) STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	return ret;
}

static int create_task_22(starpu_data_handle_t dataA, unsigned k, unsigned i, unsigned j)
{
	int ret;

/*	printf("task 22 k,i,j = %d,%d,%d TAG = %llx\n", k,i,j, TAG22(k,i,j)); */

	struct starpu_task *task = create_task(TAG22(k, i, j));

	task->cl = &ssssm_cl;

	/* which sub-data is manipulated ? */
	task->handles[0] = starpu_data_get_sub_data(dataA, 2, i, k); /* produced by TAG21(k, i) */
	task->handles[1] = starpu_data_get_sub_data(dataA, 2, k, j); /* produced by TAG12(k, j) */
	task->handles[2] = starpu_data_get_sub_data(dataA, 2, i, j); /* produced by TAG22(k-1, i, j) */

	if (!no_prio &&  (i == k + 1) && (j == k +1) )
	{
		task->priority = STARPU_MAX_PRIO;
	}

	/* enforce dependencies ... */
	if (k > 0)
	{
		starpu_tag_declare_deps(TAG22(k, i, j), 3, TAG22(k-1, i, j), TAG12(k, j), TAG21(k, i));
	}
	else
	{
		starpu_tag_declare_deps(TAG22(k, i, j), 2, TAG12(k, j), TAG21(k, i));
	}

	ret = starpu_task_submit(task);
	if (ret != -ENODEV) STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	return ret;
}

/*
 *	code to bootstrap the factorization
 */

static int dw_codelet_facto_v3(starpu_data_handle_t dataA, unsigned nblocks)
{
	int ret;
	struct starpu_task *entry_task = NULL;

	/* create all the DAG nodes */
	unsigned i,j,k;

	for (k = 0; k < nblocks; k++)
	{
		struct starpu_task *task = create_task_11(dataA, k);

		/* we defer the launch of the first task */
		if (k == 0)
		{
			entry_task = task;
		}
		else
		{
			ret = starpu_task_submit(task);
			if (ret == -ENODEV) return ret;
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
		}

		for (i = k+1; i<nblocks; i++)
		{
			ret = create_task_12(dataA, k, i);
			if (ret == -ENODEV) return ret;
			ret = create_task_21(dataA, k, i);
			if (ret == -ENODEV) return ret;
		}

		for (i = k+1; i<nblocks; i++)
		{
			for (j = k+1; j<nblocks; j++)
			{
			     ret = create_task_22(dataA, k, i, j);
			     if (ret == -ENODEV) return ret;
			}
		}
	}

	/* schedule the codelet */
	ret = starpu_task_submit(entry_task);
	if (ret == -ENODEV) return ret;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	/* stall the application until the end of computations */
	starpu_tag_wait(TAG11(nblocks-1));

	return 0;
}

static inline unsigned int get_tile_size() {
  return (unsigned int) floor(sqrt((double)__PLA_CPU_L1_CACHE / (8*3)));
}

static void check_result(void)
{
	unsigned i,j;
  elim(A_saved, l, m);
  unsigned int ctr  = 0, ctr2 = 0, ctr3 = 0;
  for (i=0; i<l; ++i) {
    for (j=i; j<m; ++j) {
      ctr++;
      if (A[j+i*m] != A_saved[j+i*m]) {
        ctr2++;
        if (j+i*m - ctr3 != 1) {
          printf("\n");
        }
        ctr3 = j+i*m;
        printf("not matchting: A[%d][%d] = %u =/= %u A_saved[%d][%d]\n", i,j, A[j+i*m], A_saved[j+i*m], i,j);
      }
    }
  }
  printf("%u / %u elements NOT matching\n", ctr2, ctr);
}

static void init_matrix(void)
{
	/* allocate matrix */
	starpu_malloc((void **)&A, (size_t)l*m*sizeof(TYPE));
	STARPU_ASSERT(A);

	starpu_srand48((long int)time(NULL));
	/* starpu_srand48(0); */

	/* initialize matrix content */
	unsigned long i,j;
	for (j = 0; j < l; j++)
	{
		for (i = 0; i < m; i++)
		{
			A[i + j*m] = (TYPE)starpu_drand48();
#ifdef COMPLEX_LU
			/* also randomize the imaginary component for complex number cases */
			A[i + j*m] += (TYPE)(I*starpu_drand48());
#endif
		}
	}
}

static void save_matrix(void)
{
	A_saved = malloc((size_t)l*m*sizeof(TYPE));
	STARPU_ASSERT(A_saved);

	memcpy(A_saved, A, (size_t)l*m*sizeof(TYPE));
}

int lu_decomposition(TYPE *matA, unsigned l, unsigned m, unsigned nblocks)
{
  struct timeval start, stop;
  clock_t cStart, cStop;
  printf("Cache-oblivious Gaussian Elimination\n");
  
	starpu_data_handle_t dataA;

	/* monitor and partition the A matrix into blocks :
	 * one block is now determined by 2 unsigned (i,j) */
	starpu_matrix_data_register(&dataA, 0, (uintptr_t)matA, m, l, m, sizeof(TYPE));

	/* We already enforce deps by hand */
	starpu_data_set_sequential_consistency_flag(dataA, 0);

	struct starpu_data_filter f =
	{
		.filter_func = starpu_matrix_filter_vertical_block,
		.nchildren = nblocks
	};

	struct starpu_data_filter f2 =
	{
		.filter_func = starpu_matrix_filter_block,
		.nchildren = nblocks
	};

	starpu_data_map_filters(dataA, 2, &f, &f2);

  gettimeofday(&start, NULL);
  cStart  = clock();

	int ret = dw_codelet_facto_v3(dataA, nblocks);

  gettimeofday(&stop, NULL);
  cStop = clock();

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

	/* gather all the data */
	starpu_data_unpartition(dataA, 0);
	starpu_data_unregister(dataA);

	return ret;
}

int main(int argc, char *argv[]) {
	int ret;

	parse_args(argc, argv);

#ifdef STARPU_QUICK_CHECK
	size /= 4;
	nblocks /= 4;
#endif

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_cublas_init();

	init_matrix();

	unsigned *ipiv = NULL;
	if (check)
		save_matrix();

	display_matrix(A, l, m, "A");

	if (bound)
		starpu_bound_start(bounddeps, boundprio);

	if (profile)
		starpu_profiling_status_set(STARPU_PROFILING_ENABLE);


	ret = lu_decomposition(A, l, m, nblocks);

	if (profile)
	{
		FPRINTF(stderr, "Setting profile\n");
		starpu_profiling_status_set(STARPU_PROFILING_DISABLE);
		starpu_profiling_bus_helper_display_summary();
	}

	if (bound)
	{
		double min;
		FPRINTF(stderr, "Setting bound\n");
		starpu_bound_stop();
		if (bounddeps)
		{
			FILE *f = fopen("lu.pl", "w");
			starpu_bound_print_lp(f);
			FPRINTF(stderr,"system printed to lu.pl\n");
			fclose(f);
			f = fopen("lu.mps", "w");
			starpu_bound_print_mps(f);
			FPRINTF(stderr,"system printed to lu.mps\n");
			fclose(f);
			f = fopen("lu.dot", "w");
			starpu_bound_print_dot(f);
			FPRINTF(stderr,"system printed to lu.mps\n");
			fclose(f);
		}
		else
		{
			starpu_bound_compute(&min, NULL, 0);
			if (min != 0.)
				FPRINTF(stderr, "theoretical min: %f ms\n", min);
		}
	}

	if (check)
	{
		FPRINTF(stderr, "Checking result\n");
		if (pivot) {
			pivot_saved_matrix(ipiv);
			free(ipiv);
		}

		check_result();
	}

	starpu_free(A);

	FPRINTF(stderr, "Shutting down\n");
	starpu_cublas_shutdown();

	starpu_shutdown();

	if (ret == -ENODEV) return 77; else return 0;
  // compute FLOPS:
  // assume addition and multiplication in the mult kernel are 2 operations
  // done A.nRows() * B.nRows() * B.nCols()
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
  int t=1, bs=0, c=1;
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
