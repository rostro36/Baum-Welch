#!/bin/bash

files=( "vec-op" )
compilers=( "i" "g" )
flags=( "-O2 -mfma" )
seeds=( 36 )
Ns=( 4 16 32 64 84 104 128 )
now=`date +%m-%d.%H:%M:%S`
for file in "${files[@]}"
do
    for compiler in "${compilers[@]}"
	do
        for flag in "${flags[@]}"
            do
	        "$compiler"cc $flag -o timing "../old_versions/bw-$file.c" io.c bw-tested.c util.c -lm
            for seed in "${seeds[@]}"
            do
                for N in "${Ns[@]}"
                do                
                    ./timing $seed $N $N $(( N * N ))
                    echo "DAS SEI UESI PARAMETER" "FILE" "$file" "FLAG" "$compiler$flag" "SEED" $seed "N" $N >> "../output_measures/$now-umdhmm-time.txt"
                    ../umdhmm/esthmm -I ../umdhmm/model.hmm ../umdhmm/sequence.seq >> "../output_measures/$now-umdhmm-time.txt"
                    echo `date +%m-%d.%H:%M:%S`
                    echo "$file $compiler$flag $seed $N"
                done
            done
        done
    done
done
rm -f timing
