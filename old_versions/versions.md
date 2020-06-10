# Baum-Welch
## old_versions
## Optimization history
- std:
    * [like Wikipedia](https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm)
- stb:
    * Stable version
- op1: 
	* Using scalar replacement
- op2 (cop):
	* transpose alpha, beta and gamma array to improve access pattern
- op3.0:
	* Rewrite and fuse backward and update step without basic optimizations
- op3:
	* Like 3.0 but the update of the transitionMatrix and emissionMatrix is in the forward step
- op3.1:
	* Like 3 but with scalar replacement
- bla:
    * added BLAS instuctions to 3.1
- op3.2:
	* precomputing indicator function is not worth it
- op3.3:
	* tranpose transitionMatrix to get better access pattern in forward step. Is not worth it.
- op3.4:
	* using blocking for computation of emission Matrix (blocksize 4x4) 
- op3.5:
	* like 3.4 but with blocking in forward step
- op 3.6:
	* like 3.1 but precomputing a*b for update step and reduced number of divisions
- reo:
    * inlined of 3.6
- op3.7:
	* like 3.1 but with blocking in forward step (uses another blocking than 3.5)
- url1.0:
	* unrolled version of opt3.1
- url1.1:
	* url1.1 with scalar replacement
- url1.2:
	* unrolled version of opt3.6. Difference to url1.1 is transposition of transitionMatrix before and after forward step to get better access pattern, precomputing a*b before the update step and computing more efficient the divisions of gamma.
- url1.3:
	* like url1.2 but without transposition of transition Matrix. Still precomputing a*b and more efficient computation of division with gamma.
- url1.4:
	* like url1.3 but without precomputing a*b. Also this is like url1.1 but with better computation of the divisions of gamma.
- vec1.2:
	* like url1.2 but with vectorized update step, final scaling and finishing criteria. To use vectorized finishing criteria you have to compile with icc and the number of observations (T) has to be divisible by 4.
- vec-simple:
	* like vec1.2 but with vectorized intial step and forward step (e.g. fully vectorized).
- vec-op:
	* like vec-simple but better usage of vectorization. Especially for reductions and transpose.
- vec-op2:
	* like vec-op but ct vector 4 times smaller.
- vec:
	* final vectorized version. Like vec-op2 but with global one vector.
- vec-op4x8
	* like vec but the inner loops are unrolled to stepsize 8.
- vec-op8x4
	* like vec but the outer loops are unrolled to stepsize 8.
- vec-op8x8
	* like vec but the inner and outer loops are unrolled to stepsize 8.

### Intel Math Kernel (BLAS)
- download mkl and icc from [here](https://dynamicinstaller.intel.com/system-studio/download)
- get mkl to path with ~~~source /opt/intel/sw_dev_tools/compilers_and_libraries_2020.1.219/linux/bin/compilervars.sh intel64~~~
- get linking part from [here](https://software.intel.com/content/www/us/en/develop/articles/intel-mkl-link-line-advisor.html)
    - add this to other files e.g. ~~~gcc -o blas bw-bla.c io.c bw-tested.c -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl~~~


## Compile +  run C code
- version <a href="https://www.codecogs.com/eqnedit.php?latex=\in" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\in" title="\in" /></a> {stb, cop, reo, vec, bla}
- make version 
- ./version $seed $hiddenState $differentObservable $T

