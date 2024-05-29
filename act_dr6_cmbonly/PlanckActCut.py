import numpy as np
from cobaya.likelihoods.base_classes import PlanckPlikLite


class PlanckActCut(PlanckPlikLite):
    """
    A simple Planck likelihood that cuts the data above a certain ell value.
    This is intended to be used to combine Planck with ACT DR6, including
    the Planck low-ell data.

    Author: Hidde T. Jense
    """
    def init_params(self, ini):
        super().init_params(ini)

        ix = 0
        uses = {}
        for i, (xy, lmin, lmax) in enumerate(zip(ini.list('use_cl'),
                                                 ini.int_list('lmin_cuts'),
                                                 ini.int_list('lmax_cuts'))):
            idx = self.used_bins[i]

            mask = np.logical_or(self.blmin[idx] < lmin, self.blmax[idx] > lmax)
            to_cut = idx[mask] + ix

            self.cov[to_cut, :] = 0.0
            self.cov[:, to_cut] = 0.0
            self.cov[to_cut, to_cut] = 1e10

            self.log.info(f"Removing bins {to_cut} in {xy.upper()}.")

            ix += len(idx)
            if len(idx) > len(to_cut):
                uses[xy] = len(idx) - len(to_cut)

        self.invcov = np.linalg.inv(self.cov)

        self.log.info(f"Using a total of {ix} bins.")
        self.log.info("Breakdown:")
        for i, k in uses.items():
            self.log.info(f"\t{i.upper()}: {k}")
