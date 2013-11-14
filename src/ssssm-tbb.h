#ifndef SSSSM_TBB
#define SSSSM_TBB

class GETRI;
class GESSM;
class TRSTI;
#include "getri-tbb.h"
#include "gessm-tbb.h"
#include "trsti-tbb.h"

class SSSSM : public task {
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
public:
  TYPE *a;
  TYPE *b;
  TYPE *c;
  TYPE offset;
  TYPE offset_a;
  TYPE offset_b;
  // successors, GETRI, GESSM, TRSTI
  // depending on offset_a and offset_b
  GETRI *getri_succ;
  GESSM *gessm_succ;
  TRSTI *trsti_succ;

  SSSSM(TYPE *a_, TYPE *b_, TYPE *c_, TYPE offset_, TYPE offset_a_, TYPE offset_b_);

  task* execute();
};
#endif
