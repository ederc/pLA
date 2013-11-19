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

using namespace tbb;

class GETRI;
class GESSM;
class TRSTI;
class SSSSM;

#include "getri-tbb.h"
#include "gessm-tbb.h"
#include "trsti-tbb.h"
#include "ssssm-tbb.h"


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

// GETRI ------------------------------
GETRI::GETRI(TYPE *a_, TYPE offset_) {
  a       = a_;
  offset  = offset_;
  gessm_succ  = new GESSM*[mblocks-offset-1];
  trsti_succ  = new TRSTI*[lblocks-offset-1];
}
task* GETRI::execute() {
  __TBB_ASSERT(ref_count()==0, NULL);
#if DEBUG
  printf("getri -- %u\n",offset);
#endif
  for (int i = 0; i < tile_size-1; ++i) {  
#if MODULUS == 1 && DELAYED_MODULUS == 1
    a[i+i*m] %=  prime;
#endif
    // compute inverse
    neg_inv_piv[i+offset] = negInverseModP(a[i+i*m], prime);
#if MODULUS == 1 && DELAYED_MODULUS == 1
    for (int j = i+1; j < tile_size; ++j) {
      a[j+i*m] %=  prime;
    }
#endif
    for (int j = i+1; j < tile_size; ++j) {
      // multiply by corresponding coeff
      a[i+j*m] *= neg_inv_piv[i+offset];
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
  neg_inv_piv[tile_size-1+offset] = negInverseModP(a[(tile_size-1)+(tile_size-1)*m], prime);
#endif

  // clean up reference counters of successors
  for (int i=0; i<mblocks-offset-1; ++i) {
    if (GESSM *t = gessm_succ[i])
      if (t->decrement_ref_count()==0) {
#if DEBUG
        printf("GETRI - spawn GESSM succ%d -- %p\n",i,t);
#endif
        spawn(*t);
      } else {
#if DEBUG
        printf("GETRI - GESSM succ ref %d\n",t->ref_count());
#endif
      }
  }
  for (int i=0; i<lblocks-offset-1; ++i)
    if (TRSTI *t = trsti_succ[i])
      if (t->decrement_ref_count()==0) {
#if DEBUG
        printf("GETRI - spawn TRSTI succ%d -- %p\n",i,t);
#endif
        spawn(*t);
      } else {
#if DEBUG
        printf("GETRI - TRSTI succ ref %d\n",t->ref_count());
#endif
      } 
  return NULL;
}

// GESSM ------------------------------
GESSM::GESSM(TYPE *a_, TYPE *b_, TYPE offset_) {
  a           = a_;
  b           = b_;
  offset      = offset_;
  ssssm_succ  = new SSSSM*[lblocks-offset-1];
}

task* GESSM::execute() {
  __TBB_ASSERT(ref_count()==0, NULL);

#if DEBUG
  printf("gessm -- %u\n",offset);
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

  // clean up reference counters of successors
  for (int i=0; i<lblocks-offset-1; ++i)
    if (SSSSM *t = ssssm_succ[i])
      if (t->decrement_ref_count()==0) {
#if DEBUG
        printf("GESSM - spawn SSSSM succ%d\n",i);
#endif
        spawn(*t);
      } else {
#if DEBUG
        printf("GESSM - SSSSM succ ref %d -- %p\n",t->ref_count(),t);
#endif
      } 
  return NULL;
}

// TRSTI ------------------------------
TRSTI::TRSTI(TYPE *a_, TYPE *b_, TYPE offset_) {
  a           = a_;
  b           = b_;
  offset      = offset_;
  ssssm_succ  = new SSSSM*[mblocks-offset-1];
}

task* TRSTI::execute() {
  __TBB_ASSERT(ref_count()==0, NULL);

#if DEBUG
  printf("trsti -- %u\n",offset);
#endif

  for (int i = 0; i < tile_size; ++i) {
    // compute inverse
    for (int j = 0; j < tile_size; ++j) {
      // multiply by corresponding coeff
      b[i+j*m] *=  neg_inv_piv[i+offset];
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
#if TRANSPOSE
  // transpose b
  TYPE temp  = 0;
  for (int i=0; i<tile_size; ++i) {
    for (int j=i+1; j<tile_size; ++j) {
      temp          = b[i+j*m];
      b[i+j*m] = b[j+i*m];
      b[j+i*m] = temp;
    }
  }
#endif

  // clean up reference counters of successors
  for (int i=0; i<mblocks-offset-1; ++i)
    if (SSSSM *t = ssssm_succ[i])
      if (t->decrement_ref_count()==0) {
#if DEBUG
        printf("TRSTI - spawn SSSSM succ%d\n",i);
#endif
        spawn(*t);
      } else {
#if DEBUG
        printf("TRSTI - SSSSM succ ref %d -- %p\n",t->ref_count(),t);
#endif
      } 
  return NULL;
}

// SSSSM ------------------------------
SSSSM::SSSSM(TYPE *a_, TYPE *b_, TYPE *c_, TYPE offset_, TYPE offset_a_, TYPE offset_b_) {
  a         = a_;
  b         = b_;
  c         = c_;
  offset    = offset_;
  offset_a  = offset_a_;
  offset_b  = offset_b_;
};

task* SSSSM::execute() {
  __TBB_ASSERT(ref_count()==0, NULL);

#if DEBUG
  printf("ssssm -- %u | %u | %u\n",offset,offset_a,offset_b);
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

  if (GETRI *t = getri_succ) {
    if (t->decrement_ref_count()==0) {
#if DEBUG
      printf("spawn GETRI -- %p\n",t);
#endif
      spawn(*t);
    } else {
#if DEBUG
      printf("SSSSM - GETRI succ ref %d -- %p\n",t->ref_count(),t);
#endif
    } 
  }
  if (GESSM *t = gessm_succ) {
    if (t->decrement_ref_count()==0) {
#if DEBUG
      printf("spawn GESSM\n");
#endif
      spawn(*t);
    } else {
#if DEBUG
      printf("SSSSM - GESSM succ ref %d\n",t->ref_count());
#endif
    } 
  }
  if (TRSTI *t = trsti_succ) {
    if (t->decrement_ref_count()==0) {
#if DEBUG
      printf("spawn TRSTI\n");
#endif
      spawn(*t);
    } else {
#if DEBUG
      printf("SSSSM - TRSTI succ ref %d\n",t->ref_count());
#endif
    } 
  }
  if (SSSSM *t = ssssm_succ) {
    if (t->decrement_ref_count()==0) {
#if DEBUG
      printf("spawn SSSSM\n");
#endif
      spawn(*t);
    } else {
#if DEBUG
      printf("SSSSM - SSSSM succ ref %d -- %p\n",t->ref_count(), t);
#endif
    } 
  }
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
  printf("       It uses Intel TBB.\n");

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
  printf("       -t        number of threads\n");
  printf("                 default: 1\n");

  exit(exval);
}

static int parse_args(int argc, char **argv)
{
	int i, opt, ret = 0;
  if(argc == 1) {
  }

  while((opt = getopt(argc, argv, "hl:m:b:p:t:cd")) != -1) {
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
      case 't':
        thrds = atoi(strdup(optarg));
        break;
    }
  }
  return ret;
}

int computeGEP(TYPE *mat, TYPE boundary) {
	/* create all the DAG nodes */
	unsigned i,j,k,ld, gessm_ctr = 0, trsti_ctr = 0, ssssm_ctr = 0;
  unsigned sum_getri = 0, sum_gessm = 0, sum_trsti = 0, sum_ssssm = 0;
  unsigned ssssm_back = 0;
  // stores the sssm index where to start for the big outer loop on the pivot
  // blocks in order to track the dependencies of getri, gessm and trsti of
  // (one round) older sssm.
  unsigned getri_old_idx = 0, gessm_old_idx = 0, trsti_old_idx = 0, ssssm_old_idx = 0;
  unsigned ssssm_start_idx = 0;

  // set threads to be used in Intel TBB spawn process
  if (thrds<1)
    thrds = task_scheduler_init::default_num_threads();
  task_scheduler_init init(thrds);

  for (i=0; i<boundary; ++i) {
    sum_getri +=  1;
    sum_gessm +=  mblocks-1-i;
    sum_trsti +=  lblocks-1-i;
    sum_ssssm +=  (mblocks-1-i)*(lblocks-1-i);
  }
  GETRI *first_task;
  GETRI *getri_last_task;
  GESSM *gessm_last_task;
  TRSTI *trsti_last_task;

  GETRI **getri_tasks = new GETRI *[sum_getri];
  GESSM **gessm_tasks = new GESSM *[sum_gessm];
  TRSTI **trsti_tasks = new TRSTI *[sum_trsti];
  SSSSM **ssssm_tasks = new SSSSM *[sum_ssssm];

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
  for (i=0; i<boundary; ++i) {
    getri_tasks[i] = new(task::allocate_root()) GETRI(&mat[tile_size*(i+i*m)],i);
    if (i!=0) {
#if DEBUG
      printf("getri count %p -- %d <-- ssssm %d\n",getri_tasks[i],i,ssssm_start_idx);
#endif
      ssssm_tasks[ssssm_start_idx]->getri_succ = getri_tasks[i];
      getri_tasks[i]->set_ref_count(1);
    } else {
      getri_tasks[i]->set_ref_count(0);
    }
    for (j=i+1; j<mblocks; ++j) {
      gessm_tasks[gessm_ctr] = new(task::allocate_root()) 
        GESSM(&mat[tile_size*(i+i*m)], &mat[tile_size*(j+i*m)],i);
      getri_tasks[i]->gessm_succ[j-i-1] = gessm_tasks[gessm_ctr];
#if DEBUG
      printf("gessm count %p -- %d <-- getri %d\n",gessm_tasks[gessm_ctr],gessm_ctr,i);
#endif
      if (i!=0) {
        ssssm_tasks[ssssm_start_idx+j-i]->gessm_succ = gessm_tasks[gessm_ctr];
        gessm_tasks[gessm_ctr]->set_ref_count(2);
      } else {
        gessm_tasks[gessm_ctr]->set_ref_count(1);
      }
      gessm_ctr++;
    }
    for (j=i+1; j<lblocks; ++j) {
      trsti_tasks[trsti_ctr] = new(task::allocate_root()) 
        TRSTI(&mat[tile_size*(i+i*m)], &mat[tile_size*(i+j*m)],i);
#if DEBUG
      printf("trsti count %p -- %d <-- getri %d\n",trsti_tasks[trsti_ctr],trsti_ctr,i);
#endif
      getri_tasks[i]->trsti_succ[j-i-1] = trsti_tasks[trsti_ctr];
      if (i!=0) {
        ssssm_tasks[ssssm_start_idx+(mblocks-i)*(j-i)]->trsti_succ = trsti_tasks[trsti_ctr];
        trsti_tasks[trsti_ctr]->set_ref_count(2);
      } else {
        trsti_tasks[trsti_ctr]->set_ref_count(1);
      }
      trsti_ctr++;
    }
    ld  = mblocks-1-i;
    for (j=i+1; j<lblocks; ++j) {
      for (k=i+1; k<mblocks; ++k) {
        ssssm_tasks[ssssm_ctr] = new(task::allocate_root()) 
          SSSSM(&mat[tile_size*(i+j*m)], &mat[tile_size*(k+i*m)],
                &mat[tile_size*(k+j*m)],i,j,k);
        trsti_tasks[trsti_old_idx+j-i-1]->ssssm_succ[k-i-1] = ssssm_tasks[ssssm_ctr];
        gessm_tasks[gessm_old_idx+k-i-1]->ssssm_succ[j-i-1] = ssssm_tasks[ssssm_ctr];
#if DEBUG
        printf("ssssm count %p -- %d <-- trsti %d\n",ssssm_tasks[ssssm_ctr],ssssm_ctr,trsti_old_idx+j-i-1);
        printf("ssssm count %p -- %d <-- gessm %d\n\n",ssssm_tasks[ssssm_ctr],ssssm_ctr,gessm_old_idx+k-i-1);
#endif
        if (i>0) {
          ssssm_back  = ssssm_ctr - (mblocks-i)*(lblocks-i-1) + (j-i);
#if DEBUG
          printf("ssssm count %p -- %d <-- back %d\n",ssssm_tasks[ssssm_ctr],ssssm_ctr,ssssm_back);
#endif
          ssssm_tasks[ssssm_back]->ssssm_succ  = ssssm_tasks[ssssm_ctr];
          ssssm_tasks[ssssm_ctr]->set_ref_count(3);
        } else {
          ssssm_tasks[ssssm_ctr]->set_ref_count(2);
        }
        ssssm_ctr++;
      }
    }
    // recompute sssm_start_idx (recursively by old index)
    if (i>0) {
      ssssm_start_idx = ssssm_old_idx;
    }
    gessm_old_idx = gessm_ctr;
    trsti_old_idx = trsti_ctr;
    ssssm_old_idx = ssssm_ctr;
  }

  first_task  = getri_tasks[0];

  // let tasks run
  if (mblocks>lblocks) {
    gessm_tasks[sum_gessm-1]->increment_ref_count();
    gessm_tasks[sum_gessm-1]->spawn_and_wait_for_all(*first_task);
    gessm_tasks[sum_gessm-1]->execute();
    task::destroy(*gessm_tasks[sum_gessm-1]);
  }
  if (mblocks<lblocks) {
    trsti_tasks[sum_trsti-1]->increment_ref_count();
    trsti_tasks[sum_trsti-1]->spawn_and_wait_for_all(*first_task);
    trsti_tasks[sum_trsti-1]->execute();
    task::destroy(*trsti_tasks[sum_trsti-1]);
  }
  if (mblocks==lblocks) {
    getri_tasks[sum_getri-1]->increment_ref_count();
    getri_tasks[sum_getri-1]->spawn_and_wait_for_all(*first_task);
    getri_tasks[sum_getri-1]->execute();
    task::destroy(*getri_tasks[sum_getri-1]);
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
  printf("Method: Intel TBB - tiled Gaussian Elimination\n");
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

	if (ret == -ENODEV) return 77; else return 0;
  // compute FLOPS:
  // assume addition and multiplication in the mult kernel are 2 operations
  // done A.nRows() * B.nRows() * B.nCols()
}
