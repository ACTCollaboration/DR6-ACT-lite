import numpy as np
from cobaya.likelihoods.base_classes import PlanckPlikLite

class PlanckActCut(PlanckPlikLite):
    def init_params(self, ini):
        super().init_params(ini)
        
        ix = 0
        for i, (xy, lmax) in enumerate(zip(ini.list('use_cl'), ini.int_list('lmax_cuts'))):
            idx = self.used_bins[i]
            
            to_cut = idx[self.blmin[idx] >= lmax] + ix
            
            self.cov[to_cut,:] = 0.0
            self.cov[:,to_cut] = 0.0
            self.cov[to_cut,to_cut] = 1e10
            
            ix += len(idx)
        
        self.invcov = np.linalg.inv(self.cov)
