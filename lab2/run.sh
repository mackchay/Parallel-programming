#!/bin/sh
#PBS -l walltime=00:01:30
#PBS -l select=1:ncpus=1:ompthreads=1
#PBS -o out-lab2.txt
#PBS -e err-lab2.txt


cd $PBS_O_WORKDIR
echo "OMP_NUM_THREADS = $OMP_NUM_THREADS"
echo
./lab2
