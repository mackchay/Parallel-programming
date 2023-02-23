#!/bin/bash

#PBS -l select=2:ncpus=8:mpiprocs=8:mem=6000m,place=scatter
#PBS -l walltime=00:00:45
#PBS -m n
#PBS -o out-lab1.txt
#PBS -e err-lab1.txt
#!/bin/bash

#PBS -l select=2:ncpus=8:mpiprocs=8:mem=6000m,place=scatter
#PBS -l walltime=00:00:45
#PBS -m n
#PBS -o out-lab1.txt
#PBS -e err-lab1.txt

MPI_NP=$(wc -l $PBS_NODEFILE | awk '{ print $1 }')

cd $PBS_O_WORKDIR

echo "Node file path: $PBS_NODEFILE"
echo "Node file contents:"
cat $PBS_NODEFILE

echo "Using mpirun at `which mpirun`"
echo "Running $MPI_NP MPI processes"

mpirun -machinefile $PBS_NODEFILE -np $MPI_NP ./lab1

MPI_NP=$(wc -l $PBS_NODEFILE | awk '{ print $1 }')

cd $PBS_O_WORKDIR

echo "Node file path: $PBS_NODEFILE"
echo "Node file contents:"
cat $PBS_NODEFILE

echo "Using mpirun at `which mpirun`"
echo "Running $MPI_NP MPI processes"

mpirun -machinefile $PBS_NODEFILE -np $MPI_NP ./lab1
