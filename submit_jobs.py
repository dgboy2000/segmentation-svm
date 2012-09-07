import os
import sys

header = \
'''#!/bin/bash

# shell
#PBS -S /bin/bash

# name of the job
#PBS -N learnbatch

# standard error output
#PBS -j oe

# queueName
#PBS -q iceq
'''

## resources blocs to allocate
## mpiprocs is num. of train img + 1
##PBS -l select=11:ncpus=12:mpiprocs=3:mem=22gb

def icemem48gbq():
    return '''
# Params for icemem48gbq
#PBS -l select=1:ncpus=12:mpiprocs=6:mem=44gb
#PBS -l walltime=23:59:00
NP=6
'''

def icemem72gbq():
    return '''
# Params for icemem72gbq
#PBS -l select=1:ncpus=12:mpiprocs=7:mem=68gb
#PBS -l walltime=23:59:00
NP=7
'''

def icepar156q():
    return '''
## Params for icepar156q
#PBS -l select=11:ncpus=12:mpiprocs=3:mem=22gb
#PBS -l walltime=03:59:00
NP=33
'''

def icetestq():
    return '''
# Params for icetestq
#PBS -l select=1:ncpus=12:mpiprocs=6:mem=22gb
#PBS -l walltime=00:19:59
NP=6
'''

def output(job_name):
     return '''
# output
#PBS -o log_{}.txt
'''.format(job_name)


def folder():
    return '\ncd $HOME/segmentation-svm/\n'

def make_job(job_name, command, queue='icepar156q'):
    job_file = job_name + '.sh'

    f = open(job_file, 'w')
    f.write(header)
    f.write(output(job_name))
    f.write(eval(queue)())
    f.write(folder())
    f.write('\n#command:\n')
    f.write(command)
    f.write(' --folder {}'.format(job_name))
    f.write('\n')
    f.close()
    
    os.system('qsub -k oe {}'.format(job_file))

if __name__=='__main__':

     ## jobs
     make_job(
         '2012.09.07.segmentation_all',
         'python segmentation_batch.py',
         queue='icemem48gbq',
         )

     #make_job(
     #    '2012.09.07.baseline_C10',
     #    'mpirun -np $NP python learn_svm_batch.py --parallel -C 10',
     #    )
 
     #make_job(
     #    '2012.09.07.baseline_C10_laplacian',
     #    'mpirun -np $NP python learn_svm_batch.py --parallel --loss laplacian -C 10',
     #    )
     
     #make_job(
     #    '2012.09.06.baseline_C10',
     #    'mpirun -np $NP python learn_svm_batch.py --parallel -C 10',
     #    )

     #make_job(
     #    '2012.09.06.baseline_C100',
     #    'mpirun -np $NP python learn_svm_batch.py --parallel -C 100',
     #    )


