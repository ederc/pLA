/**
 * \file   mat-elim-tools.h
 * \author Christian Eder ( christian.eder@inria.fr )
 * \date   March 2013
 * \brief  Source file for GEP tools.
 *         This file is part of F4RT, licensed under the GNU General
 *         Public License version 3. See COPYING for more information.
 */


unsigned int bitlog(unsigned int x) {
  unsigned int b;
  unsigned int res;

  if (x <=  8 ) { /* Shorten computation for small numbers */
    res = 2 * x;
  } else {
    b = 15; /* Find the highest non zero bit in the input argument */
    while ((b > 2) && ((unsigned int)x > 0)) {
      --b;
      x <<= 1;
    }
    x &= 0x7000;
    x >>= 12;

    res = x + 8 * (b - 1);
  }

  return res;
}


double countGEPFlops(unsigned int m, unsigned int n) {
  unsigned int boundary = m > n ? n : m;
  double logp = (double) bitlog(65521);
  double res = 0;
  unsigned int i;
  for (i = 1; i <= boundary; ++i) {
    //res +=  (2*(n-i)+1)*(m-i);
    res +=  (2*(n-i)+1+logp)*(m-i) + logp * (n-i) * (m-i);
  }
  return res;
}


unsigned long negInverseModP(unsigned long a, unsigned long prime) {
  // we do two turns of the extended Euclidian algorithm per
  // loop. Usually the sign of x changes each time through the loop,
  // but we avoid that by representing every other x as its negative,
  // which is the value minusLastX. This way no negative values show
  // up.
  unsigned long b           = prime;
  unsigned long minusLastX  = 0;
  unsigned long x           = 1;
  while (1) {
    // 1st turn
    if (a == 1)
      break;
    const unsigned long firstQuot  =   b / a;
    b                             -=  firstQuot * a;
    minusLastX                    +=  firstQuot * x;

    // 2nd turn
    if (b == 1) {
      x = prime - minusLastX;
      break;
    }
    const unsigned long secondQuot =   a / b;
    a                             -=  secondQuot * b;
    x                             +=  secondQuot * minusLastX;
  }
  return prime - x;
}
