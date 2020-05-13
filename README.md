# Baum-Welch

## Compile + run
- gcc $flag -o run baum-welch-stable.c io.c bw-tested.c tested.h -lm
- ./run $seed $hiddenState $differentObservable $T

## Run suite of different params on same file
- bash suite.sh (you may change some parameters by changing the first line of the file)

pipe the output into "model_output.txt"
- bash suite.sh > model_output.txt

## Run suite with N and different files
- bash N.sh (you may change some parameters by changing the first line of the file)
- pipe the output into "model_output.txt"
- bash N.sh > N.txt

## Plotting
- python3 plotting.py if you previously ran suite.sh > model_output.txt
- python3 N-plotting.py if you previously ran N.sh > N.txt
It stores a plot with the current time as name in the same directory.
and for the normal plots also the roofline plot.

## Valgrind
- sudo apt-get install [valgrind](https://valgrind.org/docs/manual/manual.html), [kcachegrind](https://kcachegrind.github.io/html/Home.html)
- gcc $flags -g -o run $file io.c bw-tested.c tested.h -lm
- valgrind ./run $params -
    - checks for memory leaks
- valgrind --tool=cachegrind --cachegrind-out-file=cachegrindfile --branch-sim=yes  ./run $params 
    - generate statistics
- kcachegrind cachegrindfile
    - read-out statistics for cache misses/branch predictions etc.


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
