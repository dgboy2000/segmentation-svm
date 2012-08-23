#!/bin/bash

# shell
#PBS -S /bin/bash

# name of the job
#PBS -N learnbatch

# output
#PBS -o log_learn_svm_batch.txt

# standard error output
#PBS -j oe

# max execution time
#PBS -l walltime=00:19:00

# resources blocs to allocate
# mpiprocs is num. of train img + 1
#PBS -l select=1:ncpus=12:mpiprocs=12:mem=60gb

# queueName
#PBS -q iceq

cd $HOME/segmentation-svm/
mpirun -np 32 python learn_svm_batch.py

