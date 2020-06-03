#!/bin/bash
mkl_root=("/opt/intel/mkl")
blas_flags=("-DMKL_ILP64 -m64 -I$MKLROOT/include")
blas_libs=("-Wl,--start-group $MKLROOT/lib/intel64/libmkl_intel_ilp64.a $MKLROOT/lib/intel64/libmkl_sequential.a $MKLROOT/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -ldl")
files=( "bla" )
compilers=("i" )
flags=( "-O2" )
seeds=( 36 )
differenObservables=( 8 16 32 64 128 256 512 1024 )
hiddenStates=( 8 64 128 )
Ts=( 32 512 1028 )
now=`date +%m-%d.%H:%M:%S`
for file in "${files[@]}"
do
    for compiler in "${compilers[@]}"
    do
        for flag in "${flags[@]}"
        do
                "$compiler"cc $flag $blas_flags -o time "bw-$file.c" io.c bw-tested.c util.c $blas_libs -lm
            for seed in "${seeds[@]}"
            do
                arraylength=${#hiddenStates[@]}
                for ((place=0; place<${arraylength}; place++));
                do
                    T=${Ts[place]}
                    hiddenState=${hiddenStates[place]}
                    for differentObservable in "${differenObservables[@]}"
                    do
                        echo "DAS SEI UESI PARAMETER" "FLAG" $compiler$flag "SEED" $seed "HIDDENSTATE" $hiddenState "DIFFERENTOBSERVABLES" $differentObservable "T" $T >> "../output_measures/$file-$now-time-do.txt"
                        ./time $seed $hiddenState $differentObservable $T >> "../output_measures/$file-$now-time-do.txt"
                        echo `date +%m-%d.%H:%M:%S`
                        echo "$file $compiler$flag $seed $differentObservable $hiddenState $T"
                    done
                done
            done
        done
    done
done
hiddenStates=( 8 16 32 64 128 256 512 1024)
differentObservables=( 8 64 128 )
Ts=( 32 512 1028 )
for file in "${files[@]}"
    do
    for compiler in "${compilers[@]}"
    do
        for flag in "${flags[@]}"
        do
                "$compiler"cc $flag $blas_flags -o time "bw-$file.c" io.c bw-tested.c util.c $blas_libs -lm
            for seed in "${seeds[@]}"
            do
                arraylength=${#hiddenStates[@]}
                for ((place=0; place<${arraylength}; place++));
                do
                    T=${Ts[place]}
                    differentObservable=${differentObservables[place]}
                    for hiddenState in "${hiddenStates[@]}"
                    do
                        echo "DAS SEI UESI PARAMETER" "FLAG" $compiler$flag "SEED" $seed "HIDDENSTATE" $hiddenState "DIFFERENTOBSERVABLES" $differentObservable "T" $T >> "../output_measures/$file-$now-time-hs.txt"
                        ./time $seed $hiddenState $differentObservable $T >> "../output_measures/$file-$now-time-hs.txt"
                        echo `date +%m-%d.%H:%M:%S`
                        echo "$file $compiler$flag $seed $differentObservable $hiddenState $T"
                    done
                done
            done
        done
    done
done
hiddenStates=( 8 64 128 )
differentObservables=( 8 64 128 )
Ts=( 32 64 128 256 512 1024 2048 4096 )
for file in "${files[@]}"
    do
    for compiler in "${compilers[@]}"
    do
        for flag in "${flags[@]}"
        do
                "$compiler"cc $flag $blas_flags -o time "bw-$file.c" io.c bw-tested.c util.c $blas_libs -lm
            for seed in "${seeds[@]}"
            do
                arraylength=${#hiddenStates[@]}
                for ((place=0; place<${arraylength}; place++));
                do
                    hiddenState=${hiddenStates[place]}
                    differentObservable=${differentObservables[place]}
                    for T in "${Ts[@]}"
                    do
                        echo "DAS SEI UESI PARAMETER" "FLAG" $compiler$flag "SEED" $seed "HIDDENSTATE" $hiddenState "DIFFERENTOBSERVABLES" $differentObservable "T" $T >> "../output_measures/$file-$now-time-T.txt"
                        ./time $seed $hiddenState $differentObservable $T >> "../output_measures/$file-$now-time-T.txt"
                        echo `date +%m-%d.%H:%M:%S`
                        echo "$file $compiler$flag $seed $differentObservable $hiddenState $T"
                    done
                done
            done
        done
    done 
done
