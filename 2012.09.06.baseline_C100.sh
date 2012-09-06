#!/bin/bash

# shell
#PBS -S /bin/bash

# name of the job
#PBS -N learnbatch

# standard error output
#PBS -j oe

# queueName
#PBS -q iceq

# output
#PBS -o log_2012.09.06.baseline_C100.txt

## Params for icepar156q
#PBS -l select=11:ncpus=12:mpiprocs=3:mem=22gb
#PBS -l walltime=03:59:00
NP=33

cd $HOME/segmentation-svm/

#command:
mpirun -np $NP python learn_svm_batch.py --parallel -C 100 --folder 2012.09.06.baseline_C100
