#include <math.h>
#include <float.h>
#include <getopt.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>

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
static unsigned thrds       = 1;
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



/*****************************************************************
 * class definitions
 ****************************************************************/

void *getri(TYPE *a_, TYPE offset_) {
  TYPE *a     = a_;
  TYPE offset = offset_;
#if DEBUG
  printf("getri %p -- %u << >>\n",this,offset);
  printf("\n -A -- GETRI IN - \n");
  for (int i=0; i<l; ++i) {
    for (int j=0; j<m; ++j) {
      printf("%u\t",A[j+i*m]);
    }
    printf("\n");
  }
#endif
  for (int i = 0; i < tile_size-1; ++i) {  
#if MODULUS == 1 && DELAYED_MODULUS == 1
    a[i+i*m] %=  prime;
#endif
    // compute inverse
    neg_inv_piv[i+offset*tile_size] = negInverseModP(a[i+i*m], prime);
#if MODULUS == 1 && DELAYED_MODULUS == 1
    for (int j = i+1; j < tile_size; ++j) {
      a[j+i*m] %=  prime;
    }
#endif
    for (int j = i+1; j < tile_size; ++j) {
      // multiply by corresponding coeff
      a[i+j*m] *= neg_inv_piv[i+offset*tile_size];
#if MODULUS == 1
      a[i+j*m] %=  prime;
#endif
      for (int k = i+1; k < tile_size; ++k) {
        a[k+j*m] +=  (a[k+i*m] * a[i+j*m]);
// don't do this if delayed modulus. we take care of this in the next round of
// the outer for loop going over i
#if MODULUS == 1 && DELAYED_MODULUS == 0
        a[k+j*m] %=  prime;
#endif
      }
    }
  }
// if we delay modulus this last element on the diagonal is not reduced w.r.t.
// prime in the above for loop going over i till
#if MODULUS == 1 && DELAYED_MODULUS == 1
  a[(tile_size-1)+(tile_size-1)*m] %=  prime;
  neg_inv_piv[tile_size-1+offset*tile_size] = negInverseModP(a[(tile_size-1)+(tile_size-1)*m], prime);
#endif

  return NULL;
}

// GESSM ------------------------------
void *gessm(TYPE *a_, TYPE *b_, TYPE offset_) {
  TYPE *a     = a_;
  TYPE *b     = b_;
  TYPE offset = offset_;

#if DEBUG
  printf("gessm %p -- %u\n",this,offset);
#endif

  for (int i = 0; i < tile_size-1; ++i) {  
    // reduce entries in this line mod prime
    // no other task will work on them anymore
#if MODULUS == 1 && DELAYED_MODULUS == 1
    for (int k = 0; k < tile_size; ++k) {
      b[k+i*m] %=  prime;
    }
#endif
    for (int j = i+1; j < tile_size; ++j) {  
      for (int k = 0; k < tile_size; ++k) {
        b[k+j*m] +=  (b[k+i*m] * a[i+j*m]);
#if MODULUS == 1 && DELAYED_MODULUS == 0
        b[k+j*m] %=  prime;
#endif
      }
    }
  }
  // if we delay modulus this last element on the diagonal is not reduced w.r.t.
  // prime in the above for loop going over i till
#if MODULUS == 1 && DELAYED_MODULUS == 1
  for (int k = 0; k < tile_size; ++k) {
    b[k+(tile_size-1)*m] %=  prime;
  }
#endif
  /*
  printf("\nb -- done\n");
  for (int i=0; i<tile_size; ++i) {
    for (int j=0; j<tile_size; ++j) {
      printf("%u\t",b[j+i*m]);
    }
    printf("\n");
  }
  */

  return NULL;
}

// TRSTI ------------------------------
void *trsti(TYPE *a_, TYPE *b_, TYPE offset_) {
  TYPE *a     = a_;
  TYPE *b     = b_;
  TYPE offset = offset_;

#if DEBUG
  printf("trsti %p -- %u\n",this,offset);
  printf("\n -A -- TRSTI IN - \n");
  for (int i=0; i<l; ++i) {
    for (int j=0; j<m; ++j) {
      printf("%u\t",A[j+i*m]);
    }
    printf("\n");
  }
#endif

  for (int i = 0; i < tile_size; ++i) {
    // compute inverse
    for (int j = 0; j < tile_size; ++j) {
      // multiply by corresponding coeff
      b[i+j*m] *=  neg_inv_piv[i+offset*tile_size];
#if MODULUS == 1
      b[i+j*m] %=  prime;
#endif
      for (int k = i+1; k < tile_size; ++k) {
        b[k+j*m] +=  (a[k+i*m] * b[i+j*m]);
#if MODULUS == 1 && DELAYED_MODULUS == 0
        b[k+j*m] %=  prime;
#endif
      }
    }
  }  
  return NULL;
}

// SSSSM ------------------------------
void *ssssm(TYPE *a_, TYPE *b_, TYPE *c_, TYPE offset_, TYPE offset_a_, TYPE offset_b_) {
  TYPE *a        = a_;
  TYPE *b        = b_;
  TYPE *c        = c_;
  TYPE offset    = offset_;
  TYPE offset_a  = offset_a_;
  TYPE offset_b  = offset_b_;

#if DEBUG
  printf("ssssm %p -- %u | %u | %u -- thread \n",this,offset,offset_a,offset_b);
#endif
  // compute inverse
  for (int j = 0; j < tile_size; ++j) {
    for (int i = 0; i < tile_size; ++i) {  
      // multiply by corresponding coeff
      for (int k = 0; k < tile_size; ++k) {
#if TRANSPOSE
        c[k+j*m] +=  (a[j+i*m] * b[k+i*m]) ;
#else
        c[k+j*m] +=  (a[i+j*m] * b[k+i*m]) ;
#endif

/*
 * This is the only place where we do not need to compute an immediate modulus
 * If we do not compute it here, we need to add some modulus computations in
 * GETRI, GESSM and TRSTI (macro DELAYED_MODULUS)
 */
#if MODULUS == 1 && DELAYED_MODULUS == 0
        c[k+j*m] %=  prime;
#endif
      }
    }
  }  
  /*
  printf("\n -A- \n");
  for (int i=0; i<l; ++i) {
    for (int j=0; j<m; ++j) {
      printf("%u\t",A[j+i*m]);
    }
    printf("\n");
  }
  */
  return NULL;
}

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
    //printf("inv %u\n",inv);
    for (j = i+1; j < l; ++j) {
      mult = cc[i+j*m] * inv;
      mult %= prime;
      for (k = i+1; k < m; ++k) {
        cc[k+j*m]  += cc[k+i*m] * mult;
        cc[k+j*m]  %= prime;
      }
    }
  }
#if DEGBUG
  for (i=0; i<l; ++i) {
    for (j=0; j<m; ++j) {
      printf("%u\t",cc[j+i*m]);
    }
    printf("\n");
  }
#endif
 
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
  printf("       It uses sequential code.\n");

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

int computeGEP(TYPE *mat, TYPE boundary) {
	/* create all the DAG nodes */
	unsigned i,j,k,ld;

  // generate all tasks and
  // precompute the dependency graph:
  //
  // In the following "A --> B" means that B depends on A and once task A is
  // computed the reference count of B is decremented. Once the reference count
  // of B is 0 it is spawned.
  //
  // GETRI --> GESSM
  // GETRI --> TRSTI
  // GESSM --> SSSSM
  // TRSTI --> SSSSM
  // SSSSM --> GETRI (SSSSM on the diagonal in last round)
  // SSSSM --> TRSTI (SSSSM on top horizontal in last round)
  // SSSSM --> TRSTI (SSSSM on leftmost vertical in last round)
  // SSSSM --> SSSSM (SSSSM on inner SSSSM part / not an SSSSM of 
  //                  the above 3 ones)
  //

  // let tasks run

  for (i=0; i<boundary; ++i) {
    getri(&mat[tile_size*(i+i*m)],i);
    for (j=i+1; j<lblocks; ++j)
      trsti(&mat[tile_size*(i+i*m)], &mat[tile_size*(i+j*m)],i);
    for (j=i+1; j<mblocks; ++j)
      gessm(&mat[tile_size*(i+i*m)], &mat[tile_size*(j+i*m)],i);
    for (j=i+1; j<lblocks; ++j)
      for (k=i+1; k<mblocks; ++k)
        ssssm(&mat[tile_size*(i+j*m)], &mat[tile_size*(k+i*m)],
                &mat[tile_size*(k+j*m)],i,j,k);
  }

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
	A = (TYPE*) malloc((size_t)l*m*sizeof(TYPE));

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
	A_saved = (TYPE*) malloc((size_t)l*m*sizeof(TYPE));

	memcpy(A_saved, A, (size_t)l*m*sizeof(TYPE));
  int i, j;
}

int lu_decomposition(TYPE *matA, unsigned l, unsigned m, unsigned tile_size)
{
  unsigned boundary   = (l>m) ? m : l;
  neg_inv_piv         = (TYPE *)malloc(boundary * sizeof(TYPE));
  // adjust boundary for working with blocks/tiles
  struct timeval start, stop;
  clock_t cStart, cStop;
 

  boundary   = (lblocks>mblocks) ? mblocks : lblocks;
  gettimeofday(&start, NULL);
  cStart  = clock();
  int i,j;
	int ret = computeGEP(matA, boundary);

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
  printf("Method: Sequential - tiled Gaussian Elimination\n");
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
  printf("# Threads:            %d\n", thrds);
  printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
  printf("Real time:            %.3f sec\n", realtime);
  printf("CPU time:             %.3f sec\n", cputime);
  if (cputime > epsilon)
    printf("CPU/real time:        %.2f\n", ratio);
  printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
  printf("GFLOPS/sec:           %.2f\n", flops /1000.0f/1000.0f/realtime);
  printf("-------------------------------------------------------\n");

	return ret;
}

int main(int argc, char *argv[]) {
	int ret;

	int done = parse_args(argc, argv);
  if (done)
    return 0;

#ifdef TBB_QUICK_CHECK
	size /= 4;
	nblocks /= 4;
#endif

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

	ret = lu_decomposition(A, l, m, tile_size);
	if (display)
    display_matrix(A, l, m, m, "A");

	if (check)
	{
    printf("-------------------------------------------------------\n");
    printf("Checking result\n");

		check_result();
	}

  free(A);

	printf("Shutting down\n");
  printf("=======================================================\n");

  return 0;

  // compute FLOPS:
  // assume addition and multiplication in the mult kernel are 2 operations
  // done A.nRows() * B.nRows() * B.nCols()
}
