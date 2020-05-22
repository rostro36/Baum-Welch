# Baum-Welch

## Compile + run
- gcc $flag -o run $file io.c bw-tested.c tested.h -lm
- ./run $seed $hiddenState $differentObservable $T

## Run suite of different params on same file
- bash suite.sh (you may change some parameters by changing the first line of the file)

pipe the output into "model_output.txt"
- bash suite.sh > ../output_measures/$filename

## Run suite with N and different files
- bash N.sh (you may change some parameters by changing the first line of the file)
- pipe the output into $filename
- bash N.sh > ../output_measures/$filename

## Plotting
One of the first parameters is the filename from the suite.
- python3 plotting.py if you previously ran suite.sh > model_output.txt
- python3 N-plotting.py if you previously ran N.sh > N.txt
It stores a plot with the current time as name in the same directory.
and for the normal plots also the roofline plot.

## Valgrind
- sudo apt-get install [valgrind](https://valgrind.org/docs/manual/manual.html), [kcachegrind](https://kcachegrind.github.io/html/Home.html)
From inside code folder:
- gcc $flags -g -o run $file io.c bw-tested.c tested.h -lm
- ./run $params
- valgrind --log-file=../valgrind/$abbrevation-mem ./run $params 
    - checks for memory leaks
- valgrind --tool=cachegrind --cachegrind-out-file=../valgrind/$abbrevation-cache  --branch-sim=yes  ./run $params 
    - generate statistics
- kcachegrind ../valgrind/$abbrevation-cache
    - read-out statistics for cache misses/branch predictions etc.

## Intel Math Kernel (BLAS)
- download mkl and icc from [here](https://dynamicinstaller.intel.com/system-studio/download)
- get mkl to path with ~~~source /opt/intel/sw_dev_tools/compilers_and_libraries_2020.1.219/linux/bin/compilervars.sh intel64~~~
- get linking part from [here](https://software.intel.com/content/www/us/en/develop/articles/intel-mkl-link-line-advisor.html)
    - add this to other files e.g. ~~~gcc -o blas bw-bla.c io.c bw-tested.c -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl~~~

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
- reo2:
    * inlined of 3.6
- op3.7:
	* like 3.1 but with blocking in forward step (uses another blocking than 3.5)
- url1.0:
	* unrolled version of opt1.1
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
- vec1.2.1:
    * like vec1.2 but with vectorized intial step and forward step

## Usage baum-welch.r
- open with RStudio
- Go the session -> Set Working Directory -> to source file location
- load package 'HMM' 
- comment out PI update in baum-welch-stable.c to get same result 
- also comment out PI update in bw-tested.c to make sure we don't get an error
- make sure you write initial matrices to init folder (use write_init(...))
- make sure you write result matrices to result folder (use write_result(...))

## Additional Documents

[Overleaf](https://www.overleaf.com/2741931356ngjpcjmswxff): 

- main.tex		overview of baum-welch algorithm and possible optimizations
- slides.tex		slides for meeting with supervisors
- flops.tex		cost analysis of stable and default baum-welch algorithm
- memory_accesses.tex	mememory access analysis for stable baum-welch algorithm
