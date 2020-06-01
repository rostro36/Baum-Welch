#!/bin/bash

file=( "stb" "cop" "reo" "vec" )
flags=( "-O2" )
compilers=( "g" "i" )
seeds=( 36 )
hiddenStates=( 8 64 128 )
differentObservables=( 8 64 128 )
Ts=( 32 128 512 2048 4096 8192 )
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
                hiddenState=${hiddenStates[place]}
                differentObservable=${differentObservables[place]}
                for T in "${Ts[@]}"
                do
                echo "DAS SEI UESI PARAMETER" "FLAG" $compiler$flag "SEED" $seed "HIDDENSTATE" $hiddenState "DIFFERENTOBSERVABLES" $differentObservable "T" $T >> "../output_measures/$file-time.txt"
                ./time $seed $hiddenState $differentObservable $T >> "../output_measures/$file-time.txt"
                done
            done
        done
    done
done 