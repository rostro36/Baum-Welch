#!/bin/bash

file="vec-op"
flags=( "-O2" )
seeds=( 36 )
differentObservables=( 32 )
hiddenStates=( 32 )
Ts=( 32 64 128 )
for flag in "${flags[@]}"
do
    gcc $flag -mfma -o time "bw-$file.c" io.c bw-tested.c tested.h -lm
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
                echo "DAS SEI UESI PARAMETER" "FLAG" $flag "SEED" $seed "HIDDENSTATE" $hiddenState "DIFFERENTOBSERVABLES" $differentObservable "T" $T >> "../output_measures/$file-umdhmm.txt"
                ../umdhmm/esthmm -I ../umdhmm/model.hmm ../umdhmm/sequence.seq >> "../output_measures/$file-umdhmm.txt"
                done
            done
        done
    done
done
