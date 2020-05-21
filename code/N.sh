#!/bin/bash

files=( "bw-reo.c" "bw-reo-out.c" "bw-reo-prg.c" )
flags=( -O2 )
seeds=( 0 36 )
Ns=( 8 16 32 )
for file in "${files[@]}"
do
    for flag in "${flags[@]}"
        do
	    gcc $flag -o run "$file" io.c bw-tested.c tested.h -lm
        for seed in "${seeds[@]}"
        do
            for N in "${Ns[@]}"
            do
                echo "DAS SEI UESI PARAMETER" "FILE" "$file" "FLAG" "$flag" "SEED" $seed "N" $N
                ./run $seed $N $N $(( N * N ))
                valgrind --tool=cachegrind --cachegrind-out-file="../valgrind/$file-$N-cache" --cache-sim=yes --branch-sim=yes ./run $seed $N $N $(( N * N )) > bin.txt
                done
        done
    done
done
