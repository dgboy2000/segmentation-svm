import numpy as np
from scipy import sparse


def ideal_loss(z,y,mask=None):
    nlabel = len(z)
    npix   = len(z[0])
    if mask is None:
        mask = 1.
        nvar = npix
    else:
        nvar = np.sum(mask[0])
    
    biny = np.argmax(y,axis=0)==np.c_[np.arange(nlabel)]
    loss = 1 - np.sum(mask*biny*z)/float(nvar)
    return loss
    
##------------------------------------------------------------------------------
def anchor_loss(z, y, mask=None):
    nlabel = len(z)
    npix   = len(z[0])
    if mask is None:
        mask = 1.
        nvar = npix
    else:
        nvar = np.sum(mask[0])
    
    anchor, anchor_weight = compute_loss_anchor(z,mask=mask)
    
    loss = 1 - anchor_weight*np.sum(mask*(anchor-y)**2)
    return loss

def compute_loss_anchor(z, mask=None):
    nlabel = len(z)
    npix   = len(z[0])
    if mask is None:
        nvar = npix
    else:
        nvar = np.sum(mask[0])
    
    ztilde = (1.0 - z)/(nlabel - 1.0)
    
    anchor = ztilde
    anchor_weight = (nlabel - 1.)/float(nlabel*nvar)
    return anchor, anchor_weight

##------------------------------------------------------------------------------    
def laplacian_loss(z, y, mask=None):
    L = compute_loss_laplacian(z,mask=mask)
    yy = np.asmatrix(np.asarray(y).ravel()).T
    loss = 1. + float(yy.T*L*yy)
    return loss

def compute_loss_laplacian(ground_truth, mask=None):
    ''' mask and ground_truth have shape: (nlabel x npixel)'''
    
    size = ground_truth[0].size
    if mask is None:
        gt = ground_truth
        npix = size
    else:
        npix = np.sum(mask[0])
        gt = ground_truth*mask

    nlabel = len(ground_truth)

    weight = 1.0/float((nlabel-1)*npix)

    A_blocks = []
    for l2 in range(nlabel):
        A_blocks_row = []
        for l11 in range(l2):
            A_blocks_row.append(sparse.coo_matrix((size,size)))
        for l12 in range(l2,nlabel):
            A_blocks_row.append(
                sparse.spdiags(1.0*np.logical_xor(gt[l12],gt[l2]),0,size,size))
        A_blocks.append(A_blocks_row)
    A_loss = sparse.bmat(A_blocks)

        
    A_loss = A_loss + A_loss.T
    D_loss = np.asarray(A_loss.sum(axis=0)).ravel()
    L_loss = sparse.spdiags(D_loss,0,*A_loss.shape) - A_loss

    return -weight*L_loss.tocsr()
    