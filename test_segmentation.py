import numpy as np
from rwsegment import rwmean_svm as rw
reload(rw)

from rwsegment import ioanalyze

import config
reload(config)

# rw.logger.setLogLevel()

dir_reg = config.dir_reg

## load test image
test = '01/'
file_im = dir_reg + test + 'gray.hdr'
im = ioanalyze.load(file_im)


## import prior and mask
file_prior = test + 'prior.npz'
file_mask = test + 'mask.hdr'
prior = np.load(file_prior)
mask  = ioanalyze.load(file_mask)

seeds = (-1)*mask

## normalize im
im = im/np.std(im)

## run segmentation
labelset = np.asarray([0,13,14,15,16])
print 'segment'
y = rw.segment_mean_prior(
    im, 
    prior, 
    seeds=seeds,
    beta=50,
    lmbda=1e-2,
    labelset=labelset,
    rtol=1e-6,
    maxiter=1e3,
    per_label=True,
    optim_solver='scipy',
    )
y = y.reshape((-1,len(labelset)),order='F')    
## check values in y
print 'min y =', np.min(y)
print 'max y =', np.max(y)
sumy = np.sum(y,axis=1)
print 'min sumy =', np.min(sumy), ', max sum y =', np.max(sumy)


sol = labelset[np.argmax(y,axis=1)].reshape(im.shape)
ioanalyze.save('testsol.hdr',sol.astype(int))




