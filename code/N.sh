#!/bin/bash

files=( "baum-welch-stable.c" "baum-welch-stable(copy).c" )
flags=( -O0 -O1 )
seeds=( 0 36 )
Ns=( 4 8 16 )
for file in "${files[@]}"
do
    for flag in "${flags[@]}"
        do
	    gcc $flag -o $file io.c bw-tested.c tested.h -lm
        for seed in "${seeds[@]}"
        do
            for N in "${Ns[@]}"
            do
                echo "DAS SEI UESI PARAMETER" "FILE" $file "FLAG" $flag "SEED" $seed "N" $N
                ./run $seed $N $N $(( N * N ))
            done
        done
    done
done
