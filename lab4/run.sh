#PBS -l select=1:ncpus=1:mpiprocs=1:mem=8000m,place=scatter
#PBS -l walltime=00:01:20
#PBS -m n
#PBS -o out-lab4.txt
#PBS -e err-lab4.txt

MPI_NP=$(wc -l $PBS_NODEFILE | awk '{ print $1 }')

cd $PBS_O_WORKDIR

echo "Node file path: $PBS_NODEFILE"
echo "Node file contents:"
cat $PBS_NODEFILE

echo "Using mpirun at `which mpirun`"
echo "Running $MPI_NP MPI processes"

mpirun -machinefile $PBS_NODEFILE -np $MPI_NP ./lab1
