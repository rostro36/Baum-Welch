#!/bin/bash

flags=( -O0 -O2 )
seeds=( 0 36 )
hiddenStates=( 4 8 16 62 64 )
differentObservables=( 4 8 )
Ts=( 1028 2056 )
for flag in "${flags[@]}"
do
	gcc $flag -o run baum-welch-stable.c io.c bw-tested.c tested.h -lm
    for seed in "${seeds[@]}"
    do
        for hiddenState in "${hiddenStates[@]}"
        do
            for differentObservable in "${differentObservables[@]}"
            do
                for T in "${Ts[@]}"
                do
                echo "DAS SEI UESI PARAMETER" "FLAG" $flag "SEED" $seed "HIDDENSTATE" $hiddenState "DIFFERENTOBSERVABLES" $differentObservable "T" $T
                ./run $seed $hiddenState $differentObservable $T
                done
            done
        done
    done
done
