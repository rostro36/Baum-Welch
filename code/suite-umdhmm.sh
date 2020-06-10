#!/bin/bash

files=( "vec-op" )
compilers=( "i" "g" )
flags=( "-O2 -mfma" )
seeds=( 36 )
differenObservables=( 8 16 32 64 128 256 512 1024 )
hiddenStates=( 8 64 128 )
Ts=( 64 256 512 )
now=`date +%m-%d.%H:%M:%S`
for file in "${files[@]}"
do
    for compiler in "${compilers[@]}"
    do
        for flag in "${flags[@]}"
        do
	        "$compiler"cc $flag -o time "bw-$file.c" io.c bw-tested.c util.c -lm
            for seed in "${seeds[@]}"
            do
                arraylength=${#hiddenStates[@]}
                for ((place=0; place<${arraylength}; place++));
                do
                    T=${Ts[place]}
                    hiddenState=${hiddenStates[place]}
                    for differentObservable in "${differenObservables[@]}"
                    do
                        ./time $seed $hiddenState $differentObservable $T
                        echo "DAS SEI UESI PARAMETER" "FLAG" $compiler$flag "SEED" $seed "HIDDENSTATE" $hiddenState "DIFFERENTOBSERVABLES" $differentObservable "T" $T >> "../output_measures/umdhmm-do-$now-time.txt"
                        /umdhmm/esthmm -I /umdhmm/model.hmm /umdhmm/sequence.seq >> "../output_measures/umdhmm-do-$now-time.txt"
                        echo `date +%m-%d.%H:%M:%S`
                        echo "$file $compiler$flag $seed $differentObservable $hiddenState $T"
                    done
                done
            done
        done
    done
done
hiddenStates=( 8 16 32 64 128 256 512 1024)
differentObservables=( 8 64 128 )
Ts=( 64 256 512)
for file in "${files[@]}"
    do
    for compiler in "${compilers[@]}"
    do
        for flag in "${flags[@]}"
        do
	        "$compiler"cc $flag -o time "bw-$file.c" io.c bw-tested.c util.c -lm
            for seed in "${seeds[@]}"
            do
                arraylength=${#differentObservables[@]}
                for ((place=0; place<${arraylength}; place++));
                do
                    T=${Ts[place]}
                    differentObservable=${differentObservables[place]}
                    for hiddenState in "${hiddenStates[@]}"
                    do
                        ./time $seed $hiddenState $differentObservable $T
                        echo "DAS SEI UESI PARAMETER" "FLAG" $compiler$flag "SEED" $seed "HIDDENSTATE" $hiddenState "DIFFERENTOBSERVABLES" $differentObservable "T" $T >> "../output_measures/umdhmm-hs-$now-time.txt"
                        /umdhmm/esthmm -I /umdhmm/model.hmm /umdhmm/sequence.seq >> "../output_measures/umdhmm-hs-$now-time.txt"
                        echo `date +%m-%d.%H:%M:%S`
                        echo "$file $compiler$flag $seed $differentObservable $hiddenState $T"
                    done
                done
            done
        done
    done
done
hiddenStates=( 8 64 128 )
differentObservables=( 8 64 128 )
Ts=( 1024 2048 4096 8192 16384 32768)
for file in "${files[@]}"
    do
    for compiler in "${compilers[@]}"
    do
        for flag in "${flags[@]}"
        do
	        "$compiler"cc $flag -o time "bw-$file.c" io.c bw-tested.c util.c -lm
            for seed in "${seeds[@]}"
            do
                arraylength=${#hiddenStates[@]}
                for ((place=0; place<${arraylength}; place++));
                do
                    hiddenState=${hiddenStates[place]}
                    differentObservable=${differentObservables[place]}
                    for T in "${Ts[@]}"
                    do
                        ./time $seed $hiddenState $differentObservable $T
                        echo "DAS SEI UESI PARAMETER" "FLAG" $compiler$flag "SEED" $seed "HIDDENSTATE" $hiddenState "DIFFERENTOBSERVABLES" $differentObservable "T" $T >> "../output_measures/umdhmm-T-$now-time.txt"
                        /umdhmm/esthmm -I /umdhmm/model.hmm /umdhmm/sequence.seq >> "../output_measures/umdhmm-T-$now-time.txt"
                        echo `date +%m-%d.%H:%M:%S`
                        echo "$file $compiler$flag $seed $differentObservable $hiddenState $T"
                    done
                done
            done
        done
    done 
done
rm -f time
