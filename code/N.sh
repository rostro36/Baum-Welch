#!/bin/bash

files=( "stb" "cop" "reo")
compilers=( "i" "g" )
flags=( "-O2" )
seeds=( 36 )
Ns=( 4 16 32 64 84 104 128)
now=`date +%m-%d.%H:%M:%S`
for file in "${files[@]}"
do
    for compiler in "${compilers[@]}"
	do
        for flag in "${flags[@]}"
            do
	        "$compiler"cc $flag -o timing "bw-$file.c" io.c bw-tested.c util.c -lm
            for seed in "${seeds[@]}"
            do
                for N in "${Ns[@]}"
                do
                    echo "DAS SEI UESI PARAMETER" "FILE" "$file" "FLAG" "$compiler$flag" "SEED" $seed "N" $N >> "../output_measures_comp2/$now-time.txt"
                    ./timing $seed $N $N $(( N * N )) >> "../output_measures_comp2/$now-time.txt"
                    echo `date +%m-%d.%H:%M:%S`
                    echo "$file $compiler$flag $seed $N"
                done
            done
        done
    done
done
rm -f timing
files=( "vec")
compilers=( "i" "g" )
flags=( "-O2 -mfma" )
seeds=( 36 )
Ns=( 4 16 32 64 84 104 128)
for file in "${files[@]}"
do
    for compiler in "${compilers[@]}"
	do
        for flag in "${flags[@]}"
            do
	        "$compiler"cc $flag -o timing "bw-$file.c" io.c bw-tested.c util.c -lm
            for seed in "${seeds[@]}"
            do
                for N in "${Ns[@]}"
                do
                    echo "DAS SEI UESI PARAMETER" "FILE" "$file" "FLAG" "$compiler$flag" "SEED" $seed "N" $N >> "../output_measures_comp2/$now-time.txt"
                    ./timing $seed $N $N $(( N * N )) >> "../output_measures_comp2/$now-time.txt"
                    echo `date +%m-%d.%H:%M:%S`
                    echo "$file $compiler$flag $seed $N"
                done
            done
        done
    done
done
rm -f timing
files=( "stb" "cop" "reo")
compilers=( "i" "g" )
flags=( "-O3 -march=native" )
seeds=( 36 )
Ns=( 4 16 32 64 84 104 128)
now=`date +%m-%d.%H:%M:%S`
for file in "${files[@]}"
do
    for compiler in "${compilers[@]}"
	do
        for flag in "${flags[@]}"
            do
	        "$compiler"cc $flag -o timing "bw-$file.c" io.c bw-tested.c util.c -lm
            for seed in "${seeds[@]}"
            do
                for N in "${Ns[@]}"
                do
                    echo "DAS SEI UESI PARAMETER" "FILE" "$file" "FLAG" "$compiler$flag" "SEED" $seed "N" $N >> "../output_measures_comp2/$now-time_flags.txt"
                    ./timing $seed $N $N $(( N * N )) >> "../output_measures_comp2/$now-time_flags.txt"
                    echo `date +%m-%d.%H:%M:%S`
                    echo "$file $compiler$flag $seed $N"
                done
            done
        done
    done
done
rm -f timing
files=( "vec")
compilers=( "i" "g" )
flags=( "-O3 -march=native -mfma" )
seeds=( 36 )
Ns=( 4 16 32 64 84 104 128)
for file in "${files[@]}"
do
    for compiler in "${compilers[@]}"
	do
        for flag in "${flags[@]}"
            do
	        "$compiler"cc $flag -o timing "bw-$file.c" io.c bw-tested.c util.c -lm
            for seed in "${seeds[@]}"
            do
                for N in "${Ns[@]}"
                do
                    echo "DAS SEI UESI PARAMETER" "FILE" "$file" "FLAG" "$compiler$flag" "SEED" $seed "N" $N >> "../output_measures_comp2/$now-time_flags.txt"
                    ./timing $seed $N $N $(( N * N )) >> "../output_measures_comp2/$now-time_flags.txt"
                    echo `date +%m-%d.%H:%M:%S`
                    echo "$file $compiler$flag $seed $N"
                done
            done
        done
    done
done
rm -f timing
files=( "stb" "cop" "reo")
compilers=( "i" "g" )
flags=( "-O2 -fno-tree-vectorize" )
seeds=( 36 )
Ns=( 4 16 32 64 84 104 128)
now=`date +%m-%d.%H:%M:%S`
for file in "${files[@]}"
do
    for compiler in "${compilers[@]}"
	do
        for flag in "${flags[@]}"
            do
	        "$compiler"cc $flag -o timing "bw-$file.c" io.c bw-tested.c util.c -lm
            for seed in "${seeds[@]}"
            do
                for N in "${Ns[@]}"
                do
                    echo "DAS SEI UESI PARAMETER" "FILE" "$file" "FLAG" "$compiler$flag" "SEED" $seed "N" $N >> "../output_measures_comp2/$now-time_novec.txt"
                    ./timing $seed $N $N $(( N * N )) >> "../output_measures_comp2/$now-time_novec.txt"
                    echo `date +%m-%d.%H:%M:%S`
                    echo "$file $compiler$flag $seed $N"
                done
            done
        done
    done
done
rm -f timing
files=( "vec")
compilers=( "i" "g" )
flags=( "-O2 -mfma" )
seeds=( 36 )
Ns=( 4 16 32 64 84 104 128)
for file in "${files[@]}"
do
    for compiler in "${compilers[@]}"
	do
        for flag in "${flags[@]}"
            do
	        "$compiler"cc $flag -o timing "bw-$file.c" io.c bw-tested.c util.c -lm
            for seed in "${seeds[@]}"
            do
                for N in "${Ns[@]}"
                do
                    echo "DAS SEI UESI PARAMETER" "FILE" "$file" "FLAG" "$compiler$flag" "SEED" $seed "N" $N >> "../output_measures_comp2/$now-time_novec.txt"
                    ./timing $seed $N $N $(( N * N )) >> "../output_measures_comp2/$now-time_novec.txt"
                    echo `date +%m-%d.%H:%M:%S`
                    echo "$file $compiler$flag $seed $N"
                done
            done
        done
    done
done
rm -f timing
