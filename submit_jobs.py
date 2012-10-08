import os
import sys

header = \
'''#!/bin/bash

# shell
#PBS -S /bin/bash

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

def icepar156q_test():
    return '''
## Params for icepar156q
#PBS -l select=1:ncpus=12:mpiprocs=3:mem=22gb
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

def set_jobname(name):
    return '''
# name of the job
#PBS -N learnbatch
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
    f.write(set_jobname(job_name))
    f.write(output(job_name))
    f.write(eval(queue)())
    f.write(folder())
    f.write('\n#command:\n')

    f.write(command)
    f.write(' --folder {}'.format(job_name))
    f.write(' --script {}'.format(job_file))
    f.write('\n')
    f.close()
    
    email = 'pierre-yves.baudin@ecp.fr'
    os.system('qsub -k oe -m ae -M {} {}'.format(email,job_file))

if __name__=='__main__':
    
     # jobs

     #make_job(
     #    '2012.10.07.latent_LAInone',
     #    'mpirun -np $NP python learn_svm_batch.py --parallel --latent --one_iter --loss none',
     #    )


     make_job(
        '2012.10.07.test_latent_LAInone',
        'mpirun -np $NP python learn_svm_batch.py --parallel --latent --one_iter --loss none --minimal -t 1',
        queue='icetestq',
        )

    #make_job(
    #     '2012.09.28.pca_test',
    #     'mpirun -np $NP python batch_rwpca.py -s',
    #     queue='icemem48gbq'
    #     )

   
     # make_job(
     #     '2012.09.28.test_latent_LAInone_24h',
     #     'mpirun -np $NP python learn_svm_batch.py --latent --one_iter --loss none --minimal -t 1',
     #     queue='icemem48gbq'
     #     )


     #make_job(
     #    '2012.09.28.test_latent_LAInone2',
     #    'mpirun -np $NP python learn_svm_batch.py --parallel --latent --one_iter --loss none --minimal -t 1',
     #    queue='icetestq',
     #    )


     # make_job(
     #     '2012.09.26.test_latent_LAInone',
     #     'mpirun -np $NP python learn_svm_batch.py --latent --one_iter --loss none --minimal -t 1',
     #     queue='icemem48gbq'
     #     )


     # make_job(
     #     '2012.09.25.segmentation_variance_allm',
     #     'python segmentation_batch.py -s --basis allmuscles',
     #     queue='icemem48gbq',
     #     )


     #make_job(
     #    '2012.09.25.segmentation_variance',
     #    'python segmentation_batch.py -s',
     #    queue='icemem48gbq',
     #    )


     #make_job(
     #    '2012.09.24.latent_LAInone',
     #    'mpirun -np $NP python learn_svm_batch.py --parallel --latent --one_iter --loss none',
     #    )

     #make_job(
     #    '2012.09.24.latent_LAISDloss',
     #    'mpirun -np $NP python learn_svm_batch.py --parallel --latent --one_iter',
     #    )


     # make_job(
     #     '2012.09.20.latent_LAISDloss',
     #     'mpirun -np $NP python learn_svm_batch.py --parallel --latent --one_iter',
     #     )

     # make_job(
     #     '2012.09.20.latent_LAInone',
     #     'mpirun -np $NP python learn_svm_batch.py --parallel --latent --one_iter --loss none',
     #     )
 
     #make_job(
     #    '2012.09.20.test_latent_LAInone',
     #    'mpirun -np $NP python learn_svm_batch.py --parallel --latent --one_iter --loss none --minimal',
     #    queue='icetestq'
     #    )

     #make_job(
     #    '2012.09.17.segmentation_allmuscles',
     #    'python segmentation_batch.py --basis allmuscles',
     #    queue='icemem48gbq',
     #    )

     #make_job(
     #    '2012.09.13.baseline_approx_loss1e4',
     #    'mpirun -np $NP python learn_svm_batch.py --parallel --switch_loss',
     #    )
    

     # make_job(
     #     '2012.09.12.test_baseline',
     #     'mpirun -np $NP python learn_svm_batch.py --parallel --minimal -t 1 --switch_loss',
     #     queue='icetestq',
     #     )

     #make_job(
     #    '2012.09.11.baseline_C10_laplacian_loss1e4',
     #    'mpirun -np $NP python learn_svm_batch.py --parallel -C 10 --loss laplacian',
     #    )
    
     #make_job(
     #    '2012.09.14.latent_LAInone',
     #    'mpirun -np $NP python learn_svm_batch.py --parallel --latent --one_iter --loss none',
     #    )


     #make_job(
     #    '2012.09.13.latent',
     #    'mpirun -np $NP python learn_svm_batch.py --parallel --latent --one_iter',
     #    )

     #make_job(
     #    '2012.09.13.test_latent',
     #    'mpirun -np $NP python learn_svm_batch.py --parallel --latent --minimal --one_iter',
     #    queue='icetestq',
     #    )

     #make_job(
     #    '2012.09.11.latent_C10_loss1e4',
     #    'mpirun -np $NP python learn_svm_batch.py --parallel -C 10 --latent',
     #    )

     ## weighted loss (100)
     #make_job(
     #    '2012.09.10.baseline_C10_loss1e4',
     #    'mpirun -np $NP python learn_svm_batch.py --parallel -C 10',
     #    )

     #make_job(
     #    '2012.09.10.baseline_C10_laplacian_loss1e4',
     #    'mpirun -np $NP python learn_svm_batch.py --parallel -C 10 --loss laplacian',
     #    )
     
     #make_job(
     #    '2012.09.10.segmentation_all',
     #    'python segmentation_batch.py',
     #    queue='icemem48gbq',
     #    )

     ## rerun baseline using ideal/true loss function (except in the inference)
     #make_job(
     #    '2012.09.10.baseline_C10',
     #    'mpirun -np $NP python learn_svm_batch.py --parallel -C 10',
     #    )

     #make_job(
     #    '2012.09.10.baseline_C10_laplacian',
     #    'mpirun -np $NP python learn_svm_batch.py --parallel -C 10 --loss laplacian',
     #    )

     #make_job(
     #    '2012.09.07.segmentation_all',
     #    'python segmentation_batch.py',
     #    queue='icemem48gbq',
     #    )

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


