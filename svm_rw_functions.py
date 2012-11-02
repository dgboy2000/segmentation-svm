import numpy as np

class LaplacianWeights(object):
     def __init__(self, nlabel, functions, weights=None):
         self.functions = functions
         self.nlabel = nlabel
         if weights is None:
            self.weights = \
                [f['default'] for s in range(nlabel) for f in functions]
         else: self.weights = weights

     def __call__(self, im, i, j):
         nlabel = self.nlabel
         tot_wij = [0 for s in range(nlabel)]
         for k, lfunc in enumerate(self.lfuncs):
             wij = lfunc(im, i, j)
             w = self.weights[k*nlabel, k*nlabel + nlabel]
             tot_wij = [tot_wij[s] + w[s] * wij[s] for s in range(self.nlabel)]
         return tot_wij

class AnchorReslice(object):
     def __init__(self, imshape, model, **kwargs):
         self.model = model
         islices = kwargs.pop('islices', slice(None))
         self.indices = np.arange(np.prod(imshape)).reshape(imshape)[islices]

     def get_anchor_and_weights(self, i, D, **kwargs):
         i2 = self.indices[i]
         return self.model.get_anchor_and_weights(i2,D,**kwargs) 
        
## combine all prior models
class MetaAnchorApi(object):
    def __init__(self, models, w):
        self.models = models
        self.w = w
 
    def get_anchor_and_weights(self, i, D, **kwargs):
        nlabel = len(D)
        tot_anchor  = [0 for s in range(nlabel)]
        tot_weights = [0 for s in range(nlabel)]
        
        ## prior models
        for k, model in enumerate(self.models):
            anchor, weights = api.get_anchor_and_weights(i, D, **kwargs)
            w = self.w[nlabel * k : nlabel * (k+1)]
            tot_anchor  = [tot_anchor[s]  + w[s] * anchor[s]  for s in range(nlabel)]
            tot_weights = [tot_weights[s] + w[s] * weights[s] for s in range(nlabel)]
        tot_anchor /= (tot_weights + 1e-10)
        return tot_anchor, tot_weights 


