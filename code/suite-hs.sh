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
	    "$compiler"cc $flag -o time "bw-$file.c" io.c bw-tested.c tested.h -lm
        for seed in "${seeds[@]}"
        do
            arraylength=${#hiddenStates[@]}
            for ((place=0; place<${arraylength}; place++));
            do
                T=${Ts[place]}
                differentObservable=${differentObservables[place]}
                for hiddenState in "${hiddenStates[@]}"
                do
                echo "DAS SEI UESI PARAMETER" "FLAG" $compiler$flag "SEED" $seed "HIDDENSTATE" $hiddenState "DIFFERENTOBSERVABLES" $differentObservable "T" $T >> "../output_measures/$file-time.txt"
                ./time $seed $hiddenState $differentObservable $T >> "../output_measures/$file-time.txt"
                done
            done
        done
    done
done
