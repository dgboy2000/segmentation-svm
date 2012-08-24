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
#PBS -l walltime=03:59:00

# resources blocs to allocate
# mpiprocs is num. of train img + 1
#PBS -l select=11:ncpus=12:mpiprocs=3:mem=22gb

# queueName
#PBS -q iceq

cd $HOME/segmentation-svm/
mpirun -np 33 python learn_svm_batch.py

