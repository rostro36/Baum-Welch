#!/bin/bash

file=( "stb" "cop" "reo" "vec" )
compilers=( "g" "i" )
flags=( "-O2" )
seeds=( 36 )
hiddenStates=( 8 64 128 256 512 1024)
differentObservables=( 8 64 128 )
Ts=( 32 512 1028 )
for compiler in "${compilers[@]}"
do
    for flag in "${flags[@]}"
    do
	    "$compiler"cc $flag -o cache "bw-$file-cg.c" io.c bw-tested.c tested.h -lmm
        for seed in "${seeds[@]}"
        do
            arraylength=${#hiddenStates[@]}
            for ((place=0; place<${arraylength}; place++));
            do
                T=${Ts[place]}
                differentObservable=${differentObservables[place]}
                for hiddenState in "${hiddenStates[@]}"
                do
                    valgrind --tool=cachegrind --cachegrind-out-file=../valgrind/$file-$hiddenState-$differentObservable-$T-cache --cache-sim=yes  --branch-sim=yes ./cache $seed $hiddenState $differentObservable $T > bin.txt
                    echo "DAS SEI UESI PARAMETER" "FLAG" $compiler$flag "SEED" $seed "HIDDENSTATE" $hiddenState "DIFFERENTOBSERVABLES" $differentObservable "T" $T >> "../output_measures/$file-cache.txt"
                    pcregrep -Mo "fn=bw[\n]+([^\n\r]+)" ../valgrind/$file-$hiddenState-$differentObservable-$T-cache | grep "[0-9].*" >> "../output_measures/$file-cache.txt"
                done
            done
        done
    done
done
