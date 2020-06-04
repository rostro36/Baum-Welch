#!/bin/bash

mkl_root="/opt/intel/mkl"
blas_flags="-DMKL_ILP64 -m64 -I$mkl_root/include"
blas_libs="-Wl,--start-group $mkl_root/lib/intel64/libmkl_intel_ilp64.a $mkl_root/lib/intel64/libmkl_sequential.a $mkl_root/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -ldl"
files=( "bla" )
compilers=( "g" "i" )
flags=( "-O2" )
seeds=( 36 )
Ns=( 8 32 64  )
now=`date +%m-%d.%H:%M:%S`
for file in "${files[@]}"
do
    for compiler in "${compilers[@]}"
    do
        for flag in "${flags[@]}"
            do
                "$compiler"cc $flag $blas_flags -o cache "bw-$file.c" io.c bw-tested.c util.c $blas_libs -lm
            for seed in "${seeds[@]}"
            do
                for N in "${Ns[@]}"
                do
                    valgrind --tool=cachegrind --cachegrind-out-file="../valgrind/$now-$file-$N-cache" --cache-sim=yes --branch-sim=yes ./cache $seed $N $N $(( N * N )) > bin.txt
                    echo "DAS SEI UESI PARAMETER" "FILE" "$file" "FLAG" "$compiler$flag" "SEED" $seed "N" $N >> "../output_measures/$now-cache-bla.txt"
                    pcregrep -Mo "fn=bw.*[\n]+([^\n\r]+)" ../valgrind/$now-$file-$N-cache | grep "[0-9].*" >> "../output_measures/$now-cache-bla.txt"
                    echo `date +%m-%d.%H:%M:%S`
                    echo "$file $compiler$flag $seed $N"
                done
            done
        done
    done
done
rm -f cache
