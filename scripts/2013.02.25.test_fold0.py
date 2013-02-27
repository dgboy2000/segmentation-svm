import os
import sys
import subprocess

file_script = os.path.abspath('{}.sh/'.format(__file__))
if os.path.exists(file_script):
    os.remove(file_script)
print file_script
 
def make_job(name, outdir, commands):
    dir_local = os.path.abspath(os.path.dirname(__file__))
    dir_output = os.path.expanduser('{}/{}'.format(outdir, name))
    if not os.path.isdir(dir_output): os.makedirs(dir_output)
   
    print 'dir local: {}'.format(dir_local)
    print 'dir output: {}'.format(dir_output)

    ## make script
    command = ' '.join([str(a) for a in commands])
    command += ' --folder {}'.format(name)
    print 'command {}'.format(command)

    f = open(file_script, 'a')
    f.write('{}\n\n'.format(command))
    f.close()


NP = 8

folds = [0]
C = [1e-1, 1e0, 1e1, 1e2]
Cp = [1e-3, 1e-2, 1e0, 1e2, 1e6]

for fold in folds:
    for c in C:
        for cprime in Cp:
            make_job(
                '2013.02.25.exp_latent_DDACI_crop2_Lnone_x1000_Cp{}_C{}'.format(cprime, c),
                '~/svmdata/segmentation_out/learning/',
                ['mpirun', '-np', NP, 'python', 'learn_svm_batch.py ',
                    '--parallel', '--crop', 2, '--fold', fold, 
                    '--loss none', '--loss_factor', 1000,
                    '--latent', '--one_iter', '--duald_niter', 10, '--duald_gamma', 1e1,
                    '--Cprime', cprime, '-C', c],
                 )
            make_job(
                '2013.02.25.exp_baseline_crop2_Lnone_x1000_Cp{}_C{}'.format(cprime, c),
                '~/svmdata/segmentation_out/learning/',
                ['mpirun', '-np', NP, 'python', 'learn_svm_batch.py ',
                    '--parallel', '--crop', 2, '--fold', fold, 
                    '--loss none', '--loss_factor', 1000, 
                    '--Cprime', cprime, '-C', c],
                )
 

