#ifndef TRSTI_TBB
#define TRSTI_TBB

class SSSSM;
#include "ssssm-tbb.h"

class TRSTI : public task {
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
public:
  TYPE *a;
  TYPE *b;
  TYPE offset;
  // successors, all SSSSM
  SSSSM **ssssm_succ;

  TRSTI(TYPE *a_, TYPE *b_, TYPE offset_);

  task* execute();
};
#endif
