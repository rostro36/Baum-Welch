#!/bin/bash

files=( "stb" "cop" "reo" )
compilers=( "g" "i" )
flags=( "-O1" )
seeds=( 36 )
Ns=( 8 32 64  )
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
                    valgrind --tool=cachegrind --cachegrind-out-file="../valgrind_comp2/$now-$file-$N-cache" --cache-sim=yes --branch-sim=yes ./cache $seed $N $N $(( N * N )) > bin.txt
                    echo "DAS SEI UESI PARAMETER" "FILE" "$file" "FLAG" "$compiler$flag" "SEED" $seed "N" $N >> "../output_measures_comp2/$now-cache.txt"
                    pcregrep -Mo "fn=bw.*[\n]+([^\n\r]+)" ../valgrind/$now-$file-$N-cache | grep "[0-9].*" >> "../output_measures_comp2/$now-cache.txt"
                    echo `date +%m-%d.%H:%M:%S`
                    echo "$file $compiler$flag $seed $N"
                done
            done
        done
    done
done
rm -f cache
files=( "vec" )
compilers=( "g" "i" )
flags=( "-O1 -mfma" )
seeds=( 36 )
Ns=( 8 32 64  )
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
                    valgrind --tool=cachegrind --cachegrind-out-file="../valgrind_comp2/$now-$file-$N-cache" --cache-sim=yes --branch-sim=yes ./cache $seed $N $N $(( N * N )) > bin.txt
                    echo "DAS SEI UESI PARAMETER" "FILE" "$file" "FLAG" "$compiler$flag" "SEED" $seed "N" $N >> "../output_measures_comp2/$now-cache.txt"
                    pcregrep -Mo "fn=bw.*[\n]+([^\n\r]+)" ../valgrind/$now-$file-$N-cache | grep "[0-9].*" >> "../output_measures_comp2/$now-cache.txt"
                    echo `date +%m-%d.%H:%M:%S`
                    echo "$file $compiler$flag $seed $N"
                done
            done
        done
    done
done
rm -f cache
