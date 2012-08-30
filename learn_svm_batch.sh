#!/bin/bash

# shell
#PBS -S /bin/bash

# name of the job
#PBS -N learnbatch

# output
#PBS -o log_learn_svm_batch.txt

# standard error output
#PBS -j oe

## max execution time
##PBS -l walltime=03:59:00

## resources blocs to allocate
## mpiprocs is num. of train img + 1
##PBS -l select=11:ncpus=12:mpiprocs=3:mem=22gb

## Params for icemem48gbq
##PBS -l select=1:ncpus=12:mpiprocs=6:mem=44gb
##PBS -l walltime=23:59:00
#NP=6
#
## Params for icemem72gbq
##PBS -l select=1:ncpus=12:mpiprocs=7:mem=68gb
##PBS -l walltime=23:59:00
# NP=7
#
## Params for icepar156q
#PBS -l select=11:ncpus=12:mpiprocs=3:mem=22gb
#PBS -l walltime=03:59:00
NP=33
#
## Params for icetestq
##PBS -l select=1:ncpus=12:mpiprocs=6:mem=22gb
##PBS -l walltime=00:19:59
# NP=6

# queueName
#PBS -q iceq

cd $HOME/segmentation-svm/
mpirun -np $NP python learn_svm_batch.py --parallel
#mpirun -np $NP python learn_svm_batch.py --parallel --loss laplacian
#mpirun -np $NP python learn_svm_batch.py --parallel --latent

