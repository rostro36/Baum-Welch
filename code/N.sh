#!/bin/bash

files=( "stb" "cop" "reo" "vec" )
flags=( "-O2" )
seeds=( 36 )
Ns=( 4 16 64 256 1028 4096 16384 65536 262144 1048576 4194304 16777216 67108864 )
now=`date +%m-%d.%H:%M:%S`
for file in "${files[@]}"
do
    for flag in "${flags[@]}"
        do
	    gcc $flag -o timing "bw-$file.c" io.c bw-tested.c tested.h -lm
        gcc $flag -o cache "bw-$file-cg.c" io.c bw-tested.c tested.h -lm
        for seed in "${seeds[@]}"
        do
            for N in "${Ns[@]}"
            do
                echo "DAS SEI UESI PARAMETER" "FILE" "$file" "FLAG" "$flag" "SEED" $seed "N" $N >> "../output_measures/$now-time.txt"
                ./timing $seed $N $N $(( N * N )) >> "../output_measures/$now-time.txt"
                valgrind --tool=cachegrind --cachegrind-out-file="../valgrind/$file-$N-cache" --cache-sim=yes --branch-sim=yes ./cache $seed $N $N $(( N * N )) > bin.txt
                echo "DAS SEI UESI PARAMETER" "FILE" "$file" "FLAG" "$flag" "SEED" $seed "N" $N >> "../output_measures/$now-cache.txt"
                pcregrep -Mo "fn=bw[\n]+([^\n\r]+)" ../valgrind/$file-$N-cache | grep "[0-9].*" >> "../output_measures/$now-cache.txt"
                done
        done
    done
done
