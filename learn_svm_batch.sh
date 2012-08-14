#!/bin/bash

# shell
#PBS -S /bin/bash

# name of the job
#PBS -N segtest

# output
#PBS -o segmentation-test.txt

# standard error output
#PBS -j oe

# max execution time
#PBS -l walltime=02:00:00

# ressources blocs to allocate
#PBS -l select=1:ncpus=1:mem=2gb

# queueName
#PBS -q iceq

cd /home/baudinpy/segmentation-svm/

echo `which python`
echo `/home/goodmand/epd-7.3-2-rh5-x86_64/bin/python --version`
/home/goodmand/epd-7.3-2-rh5-x86_64/bin/python -i ./segmentation_batch.py

