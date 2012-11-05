import numpy as np

class LaplacianWeights(object):
    def __init__(self, nlabel, functions, weights=None):
        self.functions = functions
        self.nlabel = nlabel
        if weights is None:
            self.weights = \
                [f['default'] for f in functions for s in range(nlabel)]
        else: self.weights = weights

    def __call__(self, im, i, j):
        nlabel = self.nlabel
        tot_wij = [0 for s in range(nlabel)]
        for k, lfunc in enumerate(self.functions):
            wij = lfunc['func'](im, i, j)
            if len(wij)==1:
                wij = [wij[0] for s in range(nlabel)]
            w = self.weights[nlabel * k : nlabel * (k+1)]
            #import ipdb; ipdb.set_trace()
            tot_wij = [tot_wij[s] + w[s] * wij[s] for s in range(nlabel)]
        return tot_wij

def reslice_models(shape, models, islices=None):
    new_models = [dict(m) for m in models]
    for m in new_models:
        m['api'] = AnchorReslice(shape, m['api'], islices=islices)
    return new_models
        
class AnchorReslice(object):
    def __init__(self, imshape, model, **kwargs):
        self.model = model
        islices = kwargs.pop('islices', slice(None))
        self.indices = np.arange(np.prod(imshape)).reshape(imshape)
        self.indices = self.indices[islices].ravel()

    def get_anchor_and_weights(self, i, D, **kwargs):
        i2 = self.indices[i]
        return self.model.get_anchor_and_weights(i,D,indices=i2,**kwargs) 
        
## combine all prior models
class MetaAnchorApi(object):
    def __init__(self, nlabel, models, weights=None):
        self.models = models
        self.weights = weights
        self.nlabel = nlabel
        if weights is None:
            self.weights = \
                [m['default'] for m in models for s in range(nlabel)]
                
    def get_anchor_and_weights(self, i, D, **kwargs):
        nlabel = len(D)
        tot_anchor  = [0 for s in range(nlabel)]
        tot_weights = [0 for s in range(nlabel)]
        
        ## prior models
        for k, model in enumerate(self.models):
            api = model['api']
            anchor, weights = api.get_anchor_and_weights(i, D, **kwargs)
            w = self.weights[nlabel * k : nlabel * (k+1)]
            tot_anchor = [tot_anchor[s]   + w[s] * anchor[s]  for s in range(nlabel)]
            tot_weights = [tot_weights[s] + w[s] * weights[s] for s in range(nlabel)]
        tot_anchor = [tot_anchor[s]/(tot_weights[s] + 1e-10) for s in range(nlabel)]
        return tot_anchor, tot_weights 


