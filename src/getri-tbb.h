#ifndef GETRI_TBB
#define GETRI_TBB

#include "gessm-tbb.h"
#include "trsti-tbb.h"

class GETRI : public task {
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
public:
  TYPE *a;
  TYPE offset;
  // successors, either GESSM or TRSTI
  GESSM **gessm_succ;
  TRSTI **trsti_succ;

  GETRI(TYPE *a_, TYPE offset_);
  task* execute();
};

#endif
