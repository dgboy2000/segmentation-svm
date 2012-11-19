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
#PBS -l select=1:ncpus=12:mpiprocs=1:mem=44gb
#PBS -l walltime=23:59:00
NP=1
'''

def icemem72gbq():
    return '''
# Params for icemem72gbq
#PBS -l select=1:ncpus=12:mpiprocs=1:mem=68gb
#PBS -l walltime=23:59:00
NP=1
'''

def icepar156q():
    return '''
## Params for icepar156q
#PBS -l select=12:ncpus=12:mpiprocs=4:mem=24000000kb
#PBS -l walltime=03:59:59
NP=48

export MPI_OPENMP_INTEROP=1
export OMP_NUM_THREADS=12
export MPI_DSM_CPULIST=0,3,6,9:allhosts
#export MPI_DSM_VERBOSE=1
'''


def icetestq():
    return '''
# Params for icetestq
#PBS -l select=4:ncpus=12:mpiprocs=4:mem=24000000kb
#PBS -l walltime=00:19:59
NP=16

export MPI_OPENMP_INTEROP=1
export OMP_NUM_THREADS=12
export MPI_DSM_CPULIST=0,3,6,9:allhosts
#export MPI_DSM_VERBOSE=1
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


    f.write('\nmodule purge\n')
    f.write('module load sgi-mpt/2.01\n')
    f.write('module load intel-compiler/12.1\n')
    f.write('module load intel-mkl/10.3\n')
    f.write('module load python/2.7.3\n')
    f.write('module load git/1.7.5.1\n')

    f.write('type python\n')

    f.write(command)
    f.write(' --folder {}'.format(job_name))
    f.write(' --script {}'.format(job_file))
    f.write('\n')
    f.close()
    
    email = 'pierre-yves.baudin@ecp.fr'
    os.system('qsub -k oe -m ae -M {} {}'.format(email,job_file))

if __name__=='__main__':


    # jobs
    
    make_job(
        '2012.11.19.test_pca_syn',
        'python batch_rwpca.py -s --no-mpi',
        queue='icetestq')
    
    make_job(
        '2012.11.19.pca_syn',
        'python batch_rwpca.py -s --no-mpi',
        queue='icemem48gbq')
        

    #make_job(
    #    '2012.11.07.test_Lsdloss_x1000_C{}'.format(100),
    #    'mpiexec_mpt -n $NP python learn_svm_batch.py ' \
    #        '--parallel --crop 5 -t 0 '\
    #        '--loss squareddiff --loss_factor 1000 '\
    #        '--latent --one_iter --approx_aci '\
    #        '-C {} '.format(100),
    #    queue='icetestq')
   
    #C = [0.01, 0.1, 1, 10, 100, 1e3, 1e4, 1e5]
    #for c in C:
    #    pass
    #    make_job(
    #        '2012.11.06.exp_latent_DACI_crop10_Lsdloss_x1000_C{}'.format(c),
    #        'mpiexec_mpt -n $NP python learn_svm_batch.py ' \
    #            '--parallel --crop 7 '\
    #            '--loss squareddiff --loss_factor 1000 '\
    #            '--latent --approx_aci --one_iter '\
    #            '-C {} '.format(c))
    #
    #    #for cprime in C:
    #    for cprime in [1e2]:
    #        make_job(
    #            '2012.11.06.exp_latent_DACI_crop10_Lnone_x1000_Cp{}_C{}'.format(cprime, c),
    #            'mpiexec_mpt -n $NP python learn_svm_batch.py ' \
    #                '--parallel --crop 7 '\
    #                '--loss none --loss_factor 1000 '\
    #                '--latent --approx_aci --one_iter '\
    #                ' --Cprime {} -C {} '.format(cprime, c))
 
    #    make_job(
    #        '2012.11.06.exp_baseline_Lsdloss_x1000_C{}'.format(c),
    #        'mpiexec_mpt -n $NP python learn_svm_batch.py ' \
    #            '--parallel '\
    #            '--loss squareddiff --loss_factor 1000 '\
    #            '-C {} '.format(c))

    #    #for cprime in C:
    #    for cprime in [1e2]:
    #        make_job(
    #            '2012.11.06.exp_baseline_Lnone_x1000_Cp{}_C{}'.format(cprime, c),
    #            'mpiexec_mpt -n $NP python learn_svm_batch.py '\
    #                '--parallel '\
    #                '--loss none --loss_factor 1000 '\
    #                '--Cprime {} -C {} '.format(cprime, c))




    #make_job(
    #    '2012.10.25.exp_handtuned_entropy',
    #    'python segmentation_batch.py -s',
    #    queue='icemem48gbq',
    #    )

    #C = [0.1, 1, 10, 100, 1000, 10000, 100000]
    #for c in C:
        #make_job(
        #   '2012.10.25.test_latent_lsdloss1000_t192_C{}'.format(c),
        #   'mpirun -np $NP python learn_svm_batch.py --parallel --one_iter --loss squareddiff --loss_factor 1000 --latent -t 192 --crop 5 --approx_aci -C {}'.format(c),
        #   queue='icetestq',
        #   )
        #make_job(
        #   '2012.10.25.test_latent_llaploss1000_t192_C{}'.format(c),
        #   'mpirun -np $NP python learn_svm_batch.py --parallel --one_iter --loss laplacian --loss_factor 1000 --latent -t 192 --crop 5 --approx_aci -C {}'.format(c),
        #   queue='icetestq',
        #   )
    #    make_job(
    #       '2012.10.23.test_latent_lsd_t192_C{}'.format(c),
    #       'mpirun -np $NP python learn_svm_batch.py --parallel --one_iter --loss anchor --latent -t 192 --crop 5 --approx_aci -C {}'.format(c),
    #       queue='icetestq',    
    #   )
    #    make_job(
    #       '2012.10.23.test_latent_lnone_t192_C{}'.format(c),
    #       'mpirun -np $NP python learn_svm_batch.py --parallel --one_iter --loss none --latent -t 192 --crop 5 --approx_aci -C {}'.format(c),
    #       queue='icetestq',
    #       )
    #    make_job(
    #       '2012.10.23.test_latent_lnone_rescale_t192_C{}'.format(c),
    #       'mpirun -np $NP python learn_svm_batch.py --parallel --one_iter --loss none --latent -t 192 --crop 5 --approx_aci --scale_only -C {}'.format(c),
    #       queue='icetestq',
    #       )
    
    #make_job(
    #    '2012.10.23.pca_syn',
    #    'python batch_rwpca.py -s',
    #    queue='icemem72gbq'
    #    )

    #make_job(
    #    '2012.10.25.latent_Lnone_approxACI_mosek',
    #    'mpirun -np $NP python learn_svm_batch.py --parallel --one_iter --loss none --latent --crop 5 --approx_aci -C 10000',
    #    )


    #make_job(
    #    '2012.10.24.latent_Lnone_logb_exact',
    #    'mpirun -np $NP python learn_svm_batch.py --parallel --one_iter --loss none --latent --crop 5 -C 10000 --use_mosek False',
    #    )

   # make_job(
   #    '2012.10.22.test_latent_LAInone',
   #    'mpirun -np $NP python learn_svm_batch.py --parallel --one_iter --loss none --latent --crop 5 --approx_aci -C 10000',
   #    )

   
    #C = [0.1, 1, 10, 100, 1000, 10000, 100000]
    #for c in C:
    #    make_job(
    #       '2012.10.22.test2_latent_t192_C{}'.format(c),
    #       'mpirun -np $NP python learn_svm_batch.py --parallel --one_iter --loss none --latent -t 192 --crop 5 --approx_aci -C {}'.format(c),
    #       queue='icetestq',
    #       )

   
    #make_job(
    #   '2012.10.22.test_latent_LAInone',
    #   'mpirun -np $NP python learn_svm_batch.py --parallel --one_iter --loss none --latent -t 192 --crop 5 --approx_aci -C 10000',
    #   queue='icetestq',
    #   )

    #  make_job(
    #      '2012.10.17.autoseeds',
    #      'python autoseeds.py -s --basis allmuscles',
    #      queue='icetestq',
    #      )
    #  

     # make_job(
     #     '2012.10.17.latent_LAInone_approx_aci',
     #     'mpirun -np $NP python learn_svm_batch.py --parallel --latent --one_iter --loss none --crop 5 --approx_aci',
     #     )


     # make_job(
     #     '2012.10.17.pca_syn',
     #     'python batch_rwpca.py -s',
     #     queue='icemem72gbq'
     #     )

   # make_job(
   #      '2012.10.11.latent_LAInone_approx_aci',
   #      'mpirun -np $NP python learn_svm_batch.py --parallel --latent --one_iter --loss none --crop 5 --approx_aci',
   #      )

   #  make_job(
   #      '2012.10.11.latent_LAInone',
   #      'mpirun -np $NP python learn_svm_batch.py --parallel --latent --one_iter --loss none --crop 5',
   #      )


     #make_job(
     #    '2012.10.10.pca_syn',
     #    'python batch_rwpca.py -s',
     #    queue='icemem48gbq'
     #    )


     #make_job(
     #    '2012.10.09.latent_LAInone_approx_aci',
     #    'mpirun -np $NP python learn_svm_batch.py --parallel --latent --one_iter --loss none --crop 5 --approx_aci',
     #    )

    #  make_job(
    #      '2012.10.09.latent_LAInone',
    #      'mpirun -np $NP python learn_svm_batch.py --parallel --latent --one_iter --loss none --crop 5',
    #      )


     # make_job(
     #    '2012.10.07.test_latent_LAInone',
     #    'mpirun -np $NP python learn_svm_batch.py --parallel --latent --one_iter --loss none --minimal -t 1 --crop 7',
     #    queue='icetestq',
     #    )

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


