#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"
// file:///opt/intel/sw_dev_tools/documentation_2020/en/compiler_c/iss2020/get_started_lc.htm
// source opt/intel/sw_dev_tools/compilers_and_libraries_2020.1.219/linux/bin/compilervars.sh intel64
// gcc -o blas blas.c -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
#define N 5
void main()
{
  int n, inca = 1, incb = 1, i;
  double a[N], b[N], c;
  //void cblas_zdotc_sub();
  n = N;
  for( i = 0; i < n; i++ ){
    a[i] = (double)2;
    b[i] = (double)(1);
  }
  c=cblas_ddot(n, a, inca, b, incb);
  printf( "The complex dot product is: ( %6.2f)\n", c );
}
