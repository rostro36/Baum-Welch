#!/bin/bash

files=( "stb" "cop" "reo" "vec" )
compilers=( "g" "i" )
flags=( "-O2" )
seeds=( 36 )
Ns=( 4 16 64 256 1028 4096 16384 65536 262144 1048576 4194304 16777216 67108864 )
now=`date +%m-%d.%H:%M:%S`
for file in "${files[@]}"
do
    for compiler in "${compilers[@]}"
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
