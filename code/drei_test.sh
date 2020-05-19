#!/bin/bash

seeds=( 0 36 )
hiddenStates=( 4 8 16 62 64 )
differentObservables=( 4 8 )
Ts=( 1028 2056 )
for seed in "${seeds[@]}"
do
    for hiddenState in "${hiddenStates[@]}"
    do
        for differentObservable in "${differentObservables[@]}"
        do
            for T in "${Ts[@]}"
            do
            echo "DAS SEI UESI PARAMETER" "FLAG" $flag "SEED" $seed "HIDDENSTATE" $hiddenState "DIFFERENTOBSERVABLES" $differentObservable "T" $T
            ./null $seed $hiddenState $differentObservable $T
            ./eins $seed $hiddenState $differentObservable $T
            ./zwei $seed $hiddenState $differentObservable $T
            ./drei $seed $hiddenState $differentObservable $T
            ./vier $seed $hiddenState $differentObservable $T
            ./fünf $seed $hiddenState $differentObservable $T
            done
        done
    done
done
