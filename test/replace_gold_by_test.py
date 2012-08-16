#!python

import os
import shutil

if __name__=='__main__':
    dir_inf = 'learning/inference/01/'
    dir_svm = 'learning/svm/01/'
    
    ans = raw_input('Are you sure (y/n) ?')
    if ans in ['y']:
        
        shutil.move(dir_inf + 'dice.test.txt',dir_inf + 'dice.gold.txt')
        shutil.move(dir_inf + 'y.test.npy'   ,dir_inf + 'y.gold.npy')
        shutil.move(dir_inf + 'sol.test.hdr' ,dir_inf + 'sol.gold.hdr')
        shutil.move(dir_inf + 'sol.test.img' ,dir_inf + 'sol.gold.img')
        
        shutil.move(dir_svm + 'w.test.txt',dir_svm + 'w.gold.txt')
        shutil.move(dir_svm + 'xi.test.txt',dir_svm + 'xi.gold.txt')