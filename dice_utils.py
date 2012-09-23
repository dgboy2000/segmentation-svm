import os
import numpy as np
from rwsegment import io_analyze

def batch_compute_dice(folder_list, labelset=[13,14,15,16]):
    import config
    dir_gt = config.dir_reg
    lbset = np.asarray(labelset, dtype=int)
    for folder in folder_list:
        for vol in config.vols:
            if not os.path.isfile(folder + '/' + vol + '/sol.hdr'):
                continue
            print 'computing dice for segmentation: {}'.format(folder + '/' + vol + '/sol.hdr')
            sol = io_analyze.load(folder + '/' + vol + '/sol.hdr').astype(int)
            gt  = io_analyze.load(dir_gt + '/' + vol + '/seg.hdr').astype(int)
            d_slice  = compute_dice_per_slice(sol, gt, labelset=labelset)
            d_labels = compute_dice_coef(sol, gt, labelset=labelset)
            #print d_slice, d_labels
            np.savetxt(folder + '/' + vol + '/dice_labels.txt', np.c_[d_labels.keys(), d_labels.values()], fmt='%d %f')
            np.savetxt(folder + '/' + vol + '/dice_slices.txt', np.c_[d_slice.keys(), d_slice.values()], fmt='%d %f')
            


def compute_dice_per_slice(seg1, seg2, labelset=None, axis=0):
    if labelset is None:
        lbset = np.union(np.unique(seg1), np.unique(seg2))
    else:
        lbset = np.asarray(labelset, dtype=int)

    seg1.flat[~np.in1d(seg1, lbset)] = -1
    seg2.flat[~np.in1d(seg2, lbset)] = -2

    nslice = seg1.shape[axis]
    s = [slice(None) for d in range(seg1.ndim)]
    dcoefs = {}
    for i in range(nslice):
        s[axis] = i
        s1 = seg1[s]
        s2 = seg2[s]
        n1 = np.sum(s1>=0)
        n2 = np.sum(s2>=0)
        if n1==0 and n2==0:
            continue
        d = 2.*np.sum(s1==s2)/float(n1 + n2)
        dcoefs[i] = d

    return dcoefs


def compute_dice_coef(seg1, seg2, labelset=None):
    if labelset is None:
        lbset = np.union(np.unique(seg1), np.unique(seg2))
    else:
        lbset = np.asarray(labelset, dtype=int)
    
    seg1.flat[~np.in1d(seg1, lbset)] = -1
    seg2.flat[~np.in1d(seg2, lbset)] = -1
    
    dicecoef = {}
    for label in lbset:
        l1 = (seg1==label)
        l2 = (seg2==label)
        d = 2*np.sum(l1&l2)/(1e-9 + np.sum(l1) + np.sum(l2))
        dicecoef[label] = d
    return dicecoef


if __name__=='__main__':
    import config
    dir_seg = config.dir_work + 'segmentation/2012.09.10.segmentation_all/' 
    folder_list = [
        dir_seg + 'constant1e-2',
        dir_seg + 'entropy1e-2',
        dir_seg + 'entropy1e-1',
        dir_seg + 'entropy1e-2_intensity1e-2',
        ]
    batch_compute_dice(folder_list, labelset=[13,14,15,16])
