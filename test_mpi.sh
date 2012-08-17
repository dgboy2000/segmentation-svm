#!bin/bash

# shell
#PBS -S /bin/bash

# name of the job
#PBS -N testmpi

# output
#PBS -o log_test_mpi.txt

# standard error output
#PBS -j oe

# max execution time
#PBS -l walltime=00:01:00

# ressources blocs to allocate
#PBS -l select=1:ncpus=5:mem=1gb

# queueName
#PBS -q iceq

python test_mpi.py

## command to run
# qsub -pe MPI 5 test_mpi.sh