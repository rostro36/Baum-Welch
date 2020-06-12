#!/bin/bash


files=( "stb" "cop" "reo" )
flags=( "-O2" )
compilers=( "g" "i" )
seeds=( 36 )
hiddenStates=( 8 64 128 )
differentObservables=( 8 64 128 )
Ts=( 1024 1368 1704 2048 2728 3416 4096 5456 6832 8192 10924 13652 16384 21844 27308 32768)
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
                    hiddenState=${hiddenStates[place]}
                    differentObservable=${differentObservables[place]}
                    for T in "${Ts[@]}"
                    do
                        echo "DAS SEI UESI PARAMETER" "FLAG" $compiler$flag "SEED" $seed "HIDDENSTATE" $hiddenState "DIFFERENTOBSERVABLES" $differentObservable "T" $T >> "../output_measures/$file-T-$now-time.txt"
                        ./time $seed $hiddenState $differentObservable $T >> "../output_measures/$file-T-$now-time.txt"
                        echo `date +%m-%d.%H:%M:%S`
                        echo "$file $compiler$flag $seed $differentObservable $hiddenState $T"
                    done
                done
            done
        done
    done 
done
rm -f time
files=( "vec" )
flags=( "-O2 -mfma" )
compilers=( "g" "i" )
seeds=( 36 )
hiddenStates=( 8 64 128 )
differentObservables=( 8 64 128 )
Ts=( 1024 1368 1704 2048 2728 3416 4096 5456 6832 8192 10924 13652 16384 21844 27308 32768 )
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
                    hiddenState=${hiddenStates[place]}
                    differentObservable=${differentObservables[place]}
                    for T in "${Ts[@]}"
                    do
                        echo "DAS SEI UESI PARAMETER" "FLAG" $compiler$flag "SEED" $seed "HIDDENSTATE" $hiddenState "DIFFERENTOBSERVABLES" $differentObservable "T" $T >> "../output_measures/$file-T-$now-time.txt"
                        ./time $seed $hiddenState $differentObservable $T >> "../output_measures/$file-T-$now-time.txt"
                        echo `date +%m-%d.%H:%M:%S`
                        echo "$file $compiler$flag $seed $differentObservable $hiddenState $T"
                    done
                done
            done
        done
    done 
done
rm -f time
