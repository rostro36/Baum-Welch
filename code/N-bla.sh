#!/bin/bash
mkl_root=("/opt/intel/mkl")
blas_flags=("-DMKL_ILP64 -m64 -I$MKLROOT/include")
blas_libs=("-Wl,--start-group $MKLROOT/lib/intel64/libmkl_intel_ilp64.a $MKLROOT/lib/intel64/libmkl_sequential.a $MKLROOT/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -ldl")
files=( "bla" )
compilers=( "i" )
flags=( "-O2" )
seeds=( 36 )
Ns=( 4 16 32 64 84 104 128 )
now=`date +%m-%d.%H:%M:%S`
for file in "${files[@]}"
do
    for compiler in "${compilers[@]}"
	do
        for flag in "${flags[@]}"
            do
	        "$compiler"cc $flag $blas_flags -o timing "bw-$file.c" io.c bw-tested.c util.c $blas_libs -lm
            for seed in "${seeds[@]}"
            do
                for N in "${Ns[@]}"
                do
                    echo "DAS SEI UESI PARAMETER" "FILE" "$file" "FLAG" "$compiler$flag" "SEED" $seed "N" $N >> "../output_measures/$now-bla-time-N.txt"
                    ./timing $seed $N $N $(( N * N )) >> "../output_measures/$now-bla-time-N.txt"
                    echo `date +%m-%d.%H:%M:%S`
                    echo "$file $compiler$flag $seed $N"
                done
            done
        done
    done
done
rm -f timing
