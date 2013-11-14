#ifndef GESSM_TBB
#define GESSM_TBB

#include "ssssm-tbb.h"

class GESSM : public task {
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
public:
  TYPE *a;
  TYPE *b;
  TYPE offset;
  // successors, all SSSSM
  SSSSM **ssssm_succ;

  GESSM(TYPE *a_, TYPE *b_, TYPE offset);

  task* execute();
};
#endif
