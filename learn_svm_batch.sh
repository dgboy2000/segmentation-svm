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
#PBS -l walltime=02:00:00

# ressources blocs to allocate
#PBS -l select=1:ncpus=1:mem=12gb

# queueName
#PBS -q iceq

cd $HOME/segmentation-svm/

python learn_svm_batch.py

