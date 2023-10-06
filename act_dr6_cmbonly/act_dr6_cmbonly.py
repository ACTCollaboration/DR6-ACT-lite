from cobaya.likelihood import Likelihood
from cobaya.conventions import data_path
import os
import numpy as np
from typing import Optional

class ACTDR6CMBonly(Likelihood):
    """
    Likelihood for ACT DR6 foreground-marginalized (cmb-only).
    Author: Hidde T. Jense
    """
    file_base_name: str = "act_dr6_cmb"

    input_file: str
    data_folder: str = "ACTDR6CMBonly"
    ell_cuts: Optional[dict] = None
    lmax_theory: Optional[int] = None

    def initialize(self):
        data_file_path = os.path.join(self.packages_path, data_path)
        self.data_folder = os.path.join(data_file_path, self.data_folder)

        import sacc
        input_filename = os.path.join(self.data_folder, self.input_file)
        self.log.debug(f"Searching for data in {input_filename}.")

        input_file = sacc.Sacc.load_fits(input_filename)

        self.log.debug("Found SACC data file.")

        pol_dt = { "t" : "0", "e" : "e", "b" : "b" }

        self.ell_cuts = self.ell_cuts or { }
        self.lmax_theory = self.lmax_theory or -1

        self.spec_meta = []
        self.cull = []
        idx_max = 0

        for pol in [ "TT", "TE", "EE" ]:
            p1, p2 = pol.lower()
            t1, t2 = pol_dt[p1], pol_dt[p2]
            dt = f"cl_{t1}{t2}"

            tracers = input_file.get_tracer_combinations(dt)
            self.log.debug(pol)
            self.log.debug(tracers)

            for tr1, tr2 in tracers:
                lmin, lmax = self.ell_cuts.get(pol, (-np.inf, np.inf))
                ls, mu, ind = input_file.get_ell_cl(dt, tr1, tr2, return_ind = True)
                mask = np.logical_and(ls >= lmin, ls <= lmax)
                if np.any(mask == False):
                    self.log.debug(f"Cutting {pol} data to the range [{lmin}-{lmax}].")
                    self.cull.append( ind[~mask] )

                self.spec_meta.append({
                    "data_type" : dt,
                    "tracer1" : tr1,
                    "tracer2" : tr2,
                    "pol" : pol.lower(),
                    "ell" : ls[mask],
                    "spec" : mu[mask],
                    "idx" : ind[mask],
                    "window" : input_file.get_bandpower_windows(ind[mask])
                })

                idx_max = max(idx_max, max(ind))
                self.lmax_theory = max(self.lmax_theory, int(ls[mask].max()))

        self.data_vec = np.zeros((idx_max+1,))
        for m in self.spec_meta:
            self.data_vec[m["idx"]] = m["spec"]

        self.covmat = input_file.covariance.covmat
        for culls in self.cull:
            self.covmat[culls,:] = 0.0
            self.covmat[:,culls] = 0.0
            self.covmat[culls,culls] = 1e10

        self.inv_cov = np.linalg.inv(self.covmat)
        self.logp_const = np.log(2.0 * np.pi) * -0.5 * len(self.data_vec)
        self.logp_const -= 0.5 * np.linalg.slogdet(self.covmat)[1]

        self.log.debug(f"log(P) = {self.logp_const}")
        self.log.debug(f"len(data vec) = {len(self.data_vec)}")

    def get_requirements(self):
        return { "Cl" : { k : self.lmax_theory + 1 for k in [ "TT", "TE", "EE" ] } }

    def loglike(self, cl):
        ps_vec = np.zeros_like(self.data_vec)

        for m in self.spec_meta:
            idx = m["idx"]
            win = m["window"].weight.T
            ls = m["window"].values
            pol = m["pol"]
            dat = cl[pol][ls]

            ps_vec[idx] = win @ dat

        delta = self.data_vec - ps_vec

        logp = -0.5 * (delta @ self.inv_cov @ delta)
        self.log.debug(f"Chisqr = {-2*logp:.3f}")
        return self.logp_const + logp

    def logp(self, **param_values):
        cl = self.theory.get_Cl(ell_factor = True)
        return self.loglike(cl)

