#!/bin/bash

files=( "stb" "cop" "reo" )
compilers=( "g" "i" )
flags=( "-O2" )
seeds=( 36 )
Ns=( 4 16 64 256 512 1028 )
now=`date +%m-%d.%H:%M:%S`
for file in "${files[@]}"
do
    for compiler in "${compilers[@]}"
	do
        for flag in "${flags[@]}"
            do
	        "$compiler"cc $flag -o timing "bw-$file.c" io.c bw-tested.c tested.h -lm
            for seed in "${seeds[@]}"
            do
                for N in "${Ns[@]}"
                do
                    echo "DAS SEI UESI PARAMETER" "FILE" "$file" "FLAG" "$compiler$flag" "SEED" $seed "N" $N >> "../output_measures/$now-time.txt"
                    ./timing $seed $N $N $(( N * N )) >> "../output_measures/$now-time.txt"
                done
            done
        done
    done
done
