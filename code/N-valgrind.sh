#!/bin/bash

files=( "stb" "cop" "reo" "vec" )
compilers=( "g" "i" )
flags=( "-O2" )
seeds=( 36 )
Ns=( 8 128  256  )
now=`date +%m-%d.%H:%M:%S`
for file in "${files[@]}"
do
    for compiler in "${compilers[@]}"
    do
        for flag in "${flags[@]}"
            do
	        "$compiler"cc $flag -o cache "bw-$file-cg.c" io.c bw-tested.c util.c -lm
            for seed in "${seeds[@]}"
            do
                for N in "${Ns[@]}"
                do
                    valgrind --tool=cachegrind --cachegrind-out-file="../valgrind/$now-$file-$N-cache" --cache-sim=yes --branch-sim=yes ./cache $seed $N $N $(( N * N )) > bin.txt
                    echo "DAS SEI UESI PARAMETER" "FILE" "$file" "FLAG" "$compiler$flag" "SEED" $seed "N" $N >> "../output_measures/$now-cache.txt"
                    pcregrep -Mo "fn=bw.*[\n]+([^\n\r]+)" ../valgrind/$now-$file-$N-cache | grep "[0-9].*" >> "../output_measures/$now-cache.txt"
                    echo `date +%m-%d.%H:%M:%S`
                    echo "$file $compiler$flag $seed $N"
                done
            done
        done
    done
done
rm -f cache
