#!/bin/bash

file="bw-cop.c"
flags=( -O0 -O2 )
seeds=( 0 36 )
hiddenStates=( 4 8 )
differentObservables=( 4 8 )
Ts=( 16 32 )
for flag in "${flags[@]}"
do
	gcc $flag -o time "$file" io.c bw-tested.c tested.h -lm
    gcc $flag -o cache "$file" io.c bw-tested.c tested.h -lm
    for seed in "${seeds[@]}"
    do
        for hiddenState in "${hiddenStates[@]}"
        do
            for differentObservable in "${differentObservables[@]}"
            do
                for T in "${Ts[@]}"
                do
                echo "DAS SEI UESI PARAMETER" "FLAG" $flag "SEED" $seed "HIDDENSTATE" $hiddenState "DIFFERENTOBSERVABLES" $differentObservable "T" $T >> "../output_measures/$file-time.txt"
                ./time $seed $hiddenState $differentObservable $T >> "../output_measures/$file-time.txt"
                valgrind --tool=cachegrind --cachegrind-out-file=../valgrind/$file-$hiddenState-$differentObservable-$T-cache --cache-sim=yes  --branch-sim=yes ./cache $seed $hiddenState $differentObservable $T > bin.txt
                echo "DAS SEI UESI PARAMETER" "FLAG" $flag "SEED" $seed "HIDDENSTATE" $hiddenState "DIFFERENTOBSERVABLES" $differentObservable "T" $T >> "../output_measures/$file-cache.txt"
                pcregrep -Mo "fn=bw[\n]+([^\n\r]+)" ../valgrind/$file-$hiddenState-$differentObservable-$T-cache | grep "[0-9].*" >> "../output_measures/$file-cache.txt"
                done
            done
        done
    done
done
