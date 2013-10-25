#include <math.h>
#include <omp.h>
#include <getopt.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>
#include <limits.h>
#include <starpu.h>
#include <starpu_data.h>

#include "pla-config.h"
//#include "mat-elim-tools.h"

/*
 * The algorithm might extend the matrix dimensions l resp. m in order to fit to
 * the tile size given. If ZEROFILL is defined then the corresponding border of
 * the matrix is filled with zeros and ones only on the diagonal.
 * Right now we have no pivoting included thus we do not check for zeros. So we
 * cannot fill with zeros only at the moment.
 */
#define ZEROFILL            1
/*
 * MODULUS and DELAYED_MODULUS need to be combined:
 * 1. no modulus computation:       MODULUS 0, DELAYED_MODULUS 0/1
 * 2. direct modulus computations:  MODULUS 1, DELAYED_MODULUS 0
 * 3. delayed modulus computations: MODULUS 1, DELAYED_MODULUS 1
 */
#define MODULUS             1
// !! NOTE AGAIN: DELAYED_MODULUS without MODULUS does nothing !!
#define DELAYED_MODULUS     1

/*
 * for debugging code. if RANDOM_MAT == 0 then a specific matrix is generated
 * and thus values in the matrix of several runs of the code coincides.
 */
#define DEBUG               0
#define RANDOM_MAT          1

/*
 * transpose B part in TRSTI to receive a better cache alignment in SSSSM
 * -------------------------------------------------------------------------
 * NOTE on 25/10/2013: transposing is usually slower since SSSSM has already
 * good alignments even it not transposed
 */
#define TRANSPOSE           0

typedef unsigned long TYPE;

TYPE *neg_inv_piv;
TYPE *A, *A_saved;
static TYPE prime           = 65521;
static unsigned l_init      = 4096;
static unsigned m_init      = 4096;
static unsigned l           = 4096;
static unsigned m           = 4096;
static unsigned lblocks     = 0;
static unsigned mblocks     = 0;
static unsigned tile_size   = 96;
static unsigned check       = 0;
static unsigned display     = 0;
static unsigned pivot       = 0;
static unsigned no_stride   = 0;
static unsigned profile     = 1;
static unsigned bound       = 0;
static unsigned bounddeps   = 0;
static unsigned boundprio   = 0;
static unsigned no_prio     = 0;


static inline TYPE negInverseModP(TYPE a, TYPE prime) {
  // we do two turns of the extended Euclidian algorithm per
  // loop. Usually the sign of x changes each time through the loop,
  // but we avoid that by representing every other x as its negative,
  // which is the value minusLastX. This way no negative values show
  // up.
  TYPE b           = prime;
  TYPE minusLastX  = 0;
  TYPE x           = 1;
  while (1) {
    // 1st turn
    if (a == 1)
      break;
    const TYPE firstQuot  =   b / a;
    b                     -=  firstQuot * a;
    minusLastX            +=  firstQuot * x;

    // 2nd turn
    if (b == 1) {
      x = prime - minusLastX;
      break;
    }
    const TYPE secondQuot =   a / b;
    a                     -=  secondQuot * b;
    x                     +=  secondQuot * minusLastX;
  }
  return prime - x;
}

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

#define TAG11(k)	((starpu_tag_t)( (1ULL<<60) | (unsigned long long)(k)))
#define TAG12(k,i)	((starpu_tag_t)(((2ULL<<60) | (((unsigned long long)(k))<<32)	\
					| (unsigned long long)(i))))
#define TAG21(k,j)	((starpu_tag_t)(((3ULL<<60) | (((unsigned long long)(k))<<32)	\
					| (unsigned long long)(j))))
#define TAG22(k,i,j)	((starpu_tag_t)(((4ULL<<60) | ((unsigned long long)(k)<<32) 	\
					| ((unsigned long long)(i)<<16)	\
					| (unsigned long long)(j))))

// multiplies A*B^T and stores it in *this
void elim(TYPE *cc, unsigned l, unsigned m) {
  unsigned i, j, k;

  //C.resize(l*m);
  unsigned sum = 0;
 
  unsigned boundary = (l > m) ? m : l;
  unsigned inv, mult;

  struct timeval start, stop;
  clock_t cStart, cStop;
  
  gettimeofday(&start, NULL);
  cStart  = clock();

  for (i = 0; i < boundary; ++i) {
    inv = negInverseModP(cc[i+i*m], prime);
    for (j = i+1; j < l; ++j) {
      mult = cc[i+j*m] * inv;
      mult %= prime;
      for (k = i+1; k < m; ++k) {
        cc[k+j*m]  += cc[k+i*m] * mult;
        cc[k+j*m]  %= prime;
      }
    }
  }
  gettimeofday(&stop, NULL);
  cStop = clock();
  float epsilon = 0.0000000001;
  double realtime = ((stop.tv_sec - start.tv_sec) * 1e6 + 
                    (stop.tv_usec - start.tv_usec)) / 1e6;
  double cputime  = (double)((cStop - cStart)) / CLOCKS_PER_SEC;
  
  double ratio = cputime/realtime;
  printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
  printf("Method:           Naive\n");
  printf("Real time:        %.4f sec\n", realtime);
  printf("CPU time:         %.4f sec\n", cputime);
  if (cputime > epsilon)
    printf("CPU/real time:    %.4f\n", ratio);
  printf("-------------------------------------------------------\n");

}

static void display_matrix(TYPE *a, unsigned l, unsigned m, unsigned ld, char *str)
{
  if (l < 33 && m < 33)
  {
    printf("=======================================================\n");
    printf("Start -- matrix %s\n", str);
    printf("-------------------------------------------------------\n");
    unsigned i,j;
    for (j = 0; j < l; j++)
    {
      for (i = 0; i < m; i++)
      {
// print corresponding memory addresses only in debug mode
#if DEBUG
        printf("%d|%p\t", a[i+j*ld],&a[i+j*ld]);
#else
        printf("%d\t", a[i+j*ld]);
#endif
      }
      printf("\n");
    }
    printf("-------------------------------------------------------\n");
    printf("End -- matrix %s\n", str);
    printf("=======================================================\n\n\n");
  }
}


void print_help(int exval) {
  printf("DESCRIPTION\n");
  printf("       Computes the Gaussian Elimination of a matrix A with\n");
  printf("       unsignedeger entries.\n");
  printf("       It uses StarPU.\n");

  printf("OPTIONS\n");
  printf("       -b SIZE   block- resp. tile size\n");
  printf("                 default: 96\n");
  printf("       -c        check result against naive sequential GEP\n");
  printf("       -d        display matrix if l<=32 and m<=32\n");
  printf("       -h        print help\n");
  printf("       -l ROWSA  row size of matrix A\n");
  printf("                 default: 4096\n");
  printf("       -m COLSA  column size of matrix A and row size of matrix B\n");
  printf("                 default: 4096\n");
  printf("       -p        field characteristic\n");
  printf("                 default: 65521\n"); 

  exit(exval);
}

static int parse_args(int argc, char **argv)
{
	int i, opt, ret = 0;
  if(argc == 1) {
  }

  while((opt = getopt(argc, argv, "hl:m:b:p:cd")) != -1) {
    switch(opt) {
      case 'h':
        print_help(0);
        ret = 1;
        break;
      case 'l': 
        l = atoi(strdup(optarg));
        break;
      case 'm': 
        m = atoi(strdup(optarg));
        break;
      case 'b': 
        tile_size = atoi(strdup(optarg));
        break;
      case 'c': 
        check = 1;
        break;
      case 'd': 
        display = 1;
        break;
      case 'p': 
        prime = atoi(strdup(optarg));
        break;
    }
  }
  return ret;
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
  unsigned i, j, k;
  TYPE *sub_a       = (TYPE *)STARPU_MATRIX_GET_PTR(descr[0]);
  unsigned tile_dim = STARPU_MATRIX_GET_NX(descr[0]);
  unsigned offset_a = STARPU_VECTOR_GET_OFFSET(descr[0]);
  unsigned ld       = STARPU_MATRIX_GET_LD(descr[0]);

  offset_a  = (offset_a / sizeof(TYPE)) %  ld;

  for (i = 0; i < tile_dim-1; ++i) {  
#if MODULUS == 1 && DELAYED_MODULUS == 1
        sub_a[i+i*ld] %=  prime;
#endif
    // compute inverse
    neg_inv_piv[i+offset_a] = negInverseModP(sub_a[i+i*ld], prime);
#if MODULUS == 1 && DELAYED_MODULUS == 1
    for (j = i+1; j < tile_dim; ++j) {
        sub_a[j+i*ld] %=  prime;
    }
#endif
    for (j = i+1; j < tile_dim; ++j) {
      // multiply by corresponding coeff
      sub_a[i+j*ld] *= neg_inv_piv[i+offset_a];
#if MODULUS == 1
      sub_a[i+j*ld] %=  prime;
#endif
      for (k = i+1; k < tile_dim; ++k) {
        sub_a[k+j*ld] +=  (sub_a[k+i*ld] * sub_a[i+j*ld]);
// don't do this if delayed modulus. we take care of this in the next round of
// the outer for loop going over i
#if MODULUS == 1 && DELAYED_MODULUS == 0
        sub_a[k+j*ld] %=  prime;
#endif
      }
    }
  }
// if we delay modulus this last element on the diagonal is not reduced w.r.t.
// prime in the above for loop going over i till
#if MODULUS == 1 && DELAYED_MODULUS == 1
  sub_a[(tile_dim-1)+(tile_dim-1)*ld] %=  prime;
  neg_inv_piv[tile_dim-1+offset_a] = negInverseModP(sub_a[(tile_dim-1)+(tile_dim-1)*ld], prime);
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
  unsigned i, j, k;
  TYPE *sub_a       = (TYPE *)STARPU_MATRIX_GET_PTR(descr[0]);
  TYPE *sub_b       = (TYPE *)STARPU_MATRIX_GET_PTR(descr[1]);
  unsigned tile_dim = STARPU_MATRIX_GET_NX(descr[0]);
  unsigned ld       = STARPU_MATRIX_GET_LD(descr[0]);
 
  for (i = 0; i < tile_dim - 1; ++i) {  
    // reduce entries in this line mod prime
    // no other task will work on them anymore
#if MODULUS == 1 && DELAYED_MODULUS == 1
    for (k = 0; k < tile_dim; ++k) {
        sub_b[k+i*ld] %=  prime;
    }
#endif
    for (j = i+1; j < tile_dim; ++j) {  
      for (k = 0; k < tile_dim; ++k) {
        sub_b[k+j*ld] +=  (sub_b[k+i*ld] * sub_a[i+j*ld]);
#if MODULUS == 1 && DELAYED_MODULUS == 0
        sub_b[k+j*ld] %=  prime;
#endif
      }
    }
  }
// if we delay modulus this last element on the diagonal is not reduced w.r.t.
// prime in the above for loop going over i till
#if MODULUS == 1 && DELAYED_MODULUS == 1
  for (k = 0; k < tile_dim; ++k) {
        sub_b[k+(tile_dim-1)*ld] %=  prime;
  }
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
  unsigned i, j, k;
  TYPE *sub_a       = (TYPE *)STARPU_MATRIX_GET_PTR(descr[0]);
  TYPE *sub_b       = (TYPE *)STARPU_MATRIX_GET_PTR(descr[1]);
  unsigned tile_dim = STARPU_MATRIX_GET_NX(descr[0]);
  unsigned ld       = STARPU_MATRIX_GET_LD(descr[0]);
  unsigned offset_a = STARPU_MATRIX_GET_OFFSET(descr[0]);

  offset_a  = (offset_a / sizeof(TYPE)) %  ld;
  for (i = 0; i < tile_dim; ++i) {
    // compute inverse
    for (j = 0; j < tile_dim; ++j) {
      // multiply by corresponding coeff
      sub_b[i+j*ld] *=  neg_inv_piv[i+offset_a];
#if MODULUS == 1
      sub_b[i+j*ld] %=  prime;
#endif
      for (k = i+1; k < tile_dim; ++k) {
        sub_b[k+j*ld] +=  (sub_a[k+i*ld] * sub_b[i+j*ld]);
#if MODULUS == 1 && DELAYED_MODULUS == 0
        sub_b[k+j*ld] %=  prime;
#endif
      }
    }
  }  
#if TRANSPOSE
  // transpose b
  TYPE temp  = 0;
  for (i=0; i<tile_dim; ++i) {
    for (j=i+1; j<tile_dim; ++j) {
      temp          = sub_b[i+j*ld];
      sub_b[i+j*ld] = sub_b[j+i*ld];
      sub_b[j+i*ld] = temp;
    }
  }
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
  unsigned i, j, k;
  TYPE *sub_a       = (TYPE *)STARPU_MATRIX_GET_PTR(descr[0]);
  TYPE *sub_b       = (TYPE *)STARPU_MATRIX_GET_PTR(descr[1]);
  TYPE *sub_c       = (TYPE *)STARPU_MATRIX_GET_PTR(descr[2]);
  unsigned tile_dim = STARPU_MATRIX_GET_NX(descr[0]);
  unsigned ld       = STARPU_MATRIX_GET_LD(descr[0]);
  unsigned offset_a = STARPU_MATRIX_GET_OFFSET(descr[0]);

    // compute inverse
  for (j = 0; j < tile_dim; ++j) {
    for (i = 0; i < tile_dim; ++i) {  
      // multiply by corresponding coeff
      for (k = 0; k < tile_dim; ++k) {
#if TRANSPOSE
        sub_c[k+j*ld] +=  (sub_a[j+i*ld] * sub_b[k+i*ld]) ;
#else
        sub_c[k+j*ld] +=  (sub_a[i+j*ld] * sub_b[k+i*ld]) ;
#endif

/*
 * This is the only place where we do not need to compute an immediate modulus
 * If we do not compute it here, we need to add some modulus computations in
 * GETRI, GESSM and TRSTI (macro DELAYED_MODULUS)
 */
#if MODULUS == 1 && DELAYED_MODULUS == 0
        sub_c[k+j*ld] %=  prime;
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

static int dw_codelet_facto_v3(starpu_data_handle_t dataA, unsigned boundary)
{
	int ret;
	struct starpu_task *entry_task = NULL;

	/* create all the DAG nodes */
	unsigned i,j,k;

	for (k = 0; k < boundary; k++)
	{
		struct starpu_task *task = create_task_11(dataA, k);

		/* we defer the launch of the first task */
		if (k == 0) {
			entry_task = task;
		} else {
			ret = starpu_task_submit(task);
			if (ret == -ENODEV) return ret;
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
		}

		for (i = k+1; i<lblocks; i++) {
			ret = create_task_21(dataA, k, i);
			if (ret == -ENODEV) return ret;
		}

		for (i = k+1; i<mblocks; i++) {
			ret = create_task_12(dataA, k, i);
			if (ret == -ENODEV) return ret;
    }

		for (i = k+1; i<lblocks; i++) {
			for (j = k+1; j<mblocks; j++) {
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
  // if l == m then TAG11 is the last computation
  starpu_task_wait_for_all();
	return 0;
}

static void check_result(void)
{
	unsigned i,j;
  elim(A_saved, l, m);
  unsigned ctr  = 0, ctr2 = 0, ctr3 = 0;
  for (i=0; i<l; ++i) {
    for (j=i; j<m; ++j) {
      ctr++;
      if (A[j+i*m] != A_saved[j+i*m]) {
        ctr2++;
        if (j+i*m - ctr3 != 1) {
          //printf("\n");
        }
        ctr3 = j+i*m;
        printf("not matchting: A[%d][%d] = %u =/= %u A_saved[%d][%d]\n", i,j, A[j+i*m], A_saved[j+i*m], i,j);
      }
    }
  }
  printf("%u / %u elements NOT matching\n", ctr2, ctr);
  printf("-------------------------------------------------------\n");
}

static void init_matrix(unsigned l_init, unsigned m_init)
{
	/* allocate matrix */
	starpu_malloc((void **)&A, (size_t)l*m*sizeof(TYPE));
  printf("size of A = %d\n",sizeof(A)/sizeof(TYPE));
	STARPU_ASSERT(A);

  assert(l_init <= l);
  assert(m_init <= m);

  srand(time(NULL));

	/* initialize matrix content */
	unsigned i,j;
#if RANDOM_MAT == 1
	for (j = 0; j < l_init; j++)
  {
		for (i = 0; i < m_init; i++)
		{
      A[i+j*m]  = rand() % prime;
		}
		for (i = m_init; i < m; i++)
		{
#if ZEROFILL
      if (j != i)
        A[i+j*m]  = 0;
      else
        A[i+j*m]  = 1;
#else
      A[i+j*m]  = rand() % prime;
#endif
		}
	}
	for (j = l_init; j < l; j++)
	{
		for (i = 0; i < m; i++)
		{
#if ZEROFILL
      if (j != i)
        A[i+j*m]  = 0;
      else
        A[i+j*m]  = 1;
#else
      A[i+j*m]  = rand() % prime;
#endif
		}
	}
#else
	for (j = 0; j < l; j++)
  {
		for (i = 0; i < m; i++)
		{
      if (i != j+1)
        A[i+j*m] = (i-m+l+j-17) % prime;
      else
        A[i+j*m] = 0;
    }
  }
#endif
}

static void save_matrix(void)
{
	A_saved = malloc((size_t)l*m*sizeof(TYPE));
	STARPU_ASSERT(A_saved);

	memcpy(A_saved, A, (size_t)l*m*sizeof(TYPE));
}

int lu_decomposition(TYPE *matA, unsigned l, unsigned m, unsigned tile_size)
{
  unsigned boundary   = (l>m) ? m : l;
  neg_inv_piv         = (TYPE *)malloc(boundary * sizeof(TYPE));
  // adjust boundary for working with blocks/tiles
  int number_threads  = starpu_worker_get_count();
  struct timeval start, stop;
  clock_t cStart, cStop;
 
	starpu_data_handle_t dataA;

	/* monitor and partition the A matrix into blocks :
	 * one block is now determined by 2 unsigned (i,j) */
	starpu_matrix_data_register(&dataA, STARPU_MAIN_RAM, (uintptr_t)matA, m, m, l, sizeof(TYPE));

  boundary   = (lblocks>mblocks) ? mblocks : lblocks;
	/* We already enforce deps by hand */
	starpu_data_set_sequential_consistency_flag(dataA, 0);

	struct starpu_data_filter f =
	{
		.filter_func = starpu_matrix_filter_vertical_block,
		.nchildren = lblocks
	};

	struct starpu_data_filter f2 =
	{
		.filter_func = starpu_matrix_filter_block,
		.nchildren = mblocks
	};

	starpu_data_map_filters(dataA, 2, &f, &f2);

  gettimeofday(&start, NULL);
  cStart  = clock();

	int ret = dw_codelet_facto_v3(dataA, boundary);

  gettimeofday(&stop, NULL);
  cStop = clock();

  double flops = (2.0f*l_init*m_init*boundary)/3.0f/3.0f;
  //double flops = (2.0f*l*m*boundary)/3.0f;
  //flops = countGEPFlops(l, m);

  float epsilon = 0.0000000001;
  double realtime = ((stop.tv_sec - start.tv_sec) * 1e6 + 
                    (stop.tv_usec - start.tv_usec)) / 1e6;
  double cputime  = (double)((cStop - cStart)) / CLOCKS_PER_SEC;
  char buffer[50];
  // get digits before decimal point of cputime (the longest number) and setw
  // with it: digits + 1 (point) + 4 (precision) 
  int digits = sprintf(buffer,"%.0f",cputime);
  
  double ratio = cputime/realtime;
  printf("=======================================================\n");
  printf("Method: StarPU - tiled Gaussian Elimination\n");
  printf("-------------------------------------------------------\n");
  printf("Field characteristic: %d\n", prime);
#if MODULUS == 0
  printf("modulus computations: no\n");
#endif
#if MODULUS == 1 && DELAYED_MODULUS == 0
  printf("modulus computations: yes, direct\n");
#endif
#if MODULUS == 1 && DELAYED_MODULUS == 1
  printf("modulus computations: yes, delayed\n");
#endif
  printf("-------------------------------------------------------\n");
  printf("Tile size:            %d\n", tile_size);
  printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
  printf("Matrix sizes\n");
  printf("> user input:         %d x %d\n", l_init, m_init);  
  printf("> generated:          %d x %d\n", l, m);  
  printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
  printf("# row blocks:         %d\n",lblocks);
  printf("# column blocks:      %d\n",mblocks);
  printf("-------------------------------------------------------\n");
  printf("#Threads:             %d\n", number_threads);
  printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
  printf("Real time:            %.3f sec\n", realtime);
  printf("CPU time:             %.3f sec\n", cputime);
  if (cputime > epsilon)
    printf("CPU/real time:        %.2f\n", ratio);
  printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
  printf("GFLOPS/sec:           %.2f\n", flops /1000.0f/1000.0f/realtime);
  printf("-------------------------------------------------------\n");

	/* gather all the data */
	//starpu_data_unpartition(dataA, STARPU_MAIN_RAM);
	//starpu_data_unregister(dataA);

	return ret;
}

int main(int argc, char *argv[]) {
	int ret;

	int done = parse_args(argc, argv);
  if (done)
    return 0;

#ifdef STARPU_QUICK_CHECK
	size /= 4;
	nblocks /= 4;
#endif

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_cublas_init();
  /***********************************
   * calculate correct dimensions for
   * tiling
   */
  //printf("boundary %u\n",boundary);
  l_init  = l;
  m_init  = m;

  lblocks = l / tile_size;
  mblocks = m / tile_size;

  if (l % tile_size > 0) {
    lblocks++;
    l +=  tile_size - (l % tile_size);
  }
  if (m % tile_size > 0) {
    mblocks++;
    m +=  tile_size - (m % tile_size);
  }

	init_matrix(l_init, m_init);

	unsigned *ipiv = NULL;
	if (check)
		save_matrix();

	if (display)
    display_matrix(A, l, m, m, "A");

	if (bound)
		starpu_bound_start(bounddeps, boundprio);

	if (profile)
		starpu_profiling_status_set(STARPU_PROFILING_ENABLE);


	ret = lu_decomposition(A, l, m, tile_size);
	if (display)
    display_matrix(A, l, m, m, "A");

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
    printf("-------------------------------------------------------\n");
    printf("Checking result\n");

		check_result();
	}

  starpu_free(A);

	starpu_cublas_shutdown();
	starpu_shutdown();

	printf("Shutting down\n");
  printf("=======================================================\n");

	if (ret == -ENODEV) return 77; else return 0;
  // compute FLOPS:
  // assume addition and multiplication in the mult kernel are 2 operations
  // done A.nRows() * B.nRows() * B.nCols()
}
