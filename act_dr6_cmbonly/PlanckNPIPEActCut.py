import numpy as np
from cobaya.likelihoods.planck_NPIPE_highl_CamSpec.planck_NPIPE_CamSpec_python import Planck2020CamSpecPython


class PlanckNPIPEActCut(Planck2020CamSpecPython):
    """
    A simple Planck likelihood that cuts the data above a certain ell value.
    This is intended to be used to combine Planck with ACT DR6, including
    the Planck low-ell data.

    Author: Hidde T. Jense
    """
    def init_params(self, ini):
        super().init_params(ini)

        ix = 0
        for i, (xy, lmax) in enumerate(zip(ini.list('use_cl'),
                                           ini.int_list('lmax_cuts'))):
            if self.ell_ranges[i] is None: continue

            idx = np.arange(self.used_sizes[i])

            to_cut = idx[np.asarray(self.ell_ranges[i]) >= lmax] + ix

            self.cov[to_cut, :] = 0.0
            self.cov[:, to_cut] = 0.0
            self.cov[to_cut, to_cut] = 1e10

            ix += len(idx)

        self.covinv = np.linalg.inv(self.cov)
