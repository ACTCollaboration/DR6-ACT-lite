import jax.numpy as np
import os
import sacc
from typing import Sequence


class ACTDR6jax:
    data_folder: str = "ACTDR6CMBonly"
    input_filename: str = "act_dr6_cmb_sacc.fits"
    polarizations: Sequence[str] = ["tt", "te", "ee"]
    tt_lmax: int = 9000
    ell_cuts: dict = {
        "TT": [600, 6500],
        "TE": [600, 6500],
        "EE": [500, 6500]
    }

    def __init__(self, verbose: bool = False) -> None:
        self.__verbose = verbose

    def load_data(self) -> None:
        # load the data
        if self.verbose:
            print(f"Loading data from {self.input_filename}.")

        saccfile = sacc.Sacc.load_fits(
            os.path.join(self.data_folder, self.input_filename)
        )

        idx_max = 0
        pol_dt = {"t": "0", "e": "e", "b": "b"}
        self.spec_meta = []
        self.cull = []

        for pol in self.polarizations:
            p1, p2 = pol.lower()
            t1, t2 = pol_dt[p1], pol_dt[p2]
            dt = f"cl_{t1}{t2}"

            tracers = saccfile.get_tracer_combinations(dt)

            for tr1, tr2 in tracers:
                if self.verbose:
                    print(f"{tr1}x{tr2}")

                lmin, lmax = self.ell_cuts.get(pol.upper(), (-np.inf, np.inf))

                ls, mu, ind = saccfile.get_ell_cl(dt, tr1, tr2,
                                                  return_ind=True)
                mask = np.logical_and(ls >= lmin, ls <= lmax)
                if not np.all(mask):
                    if self.verbose:
                        print(f"Cutting {pol} data to the \
                            range [{lmin}-{lmax}].")
                    self.cull.append(ind[~mask])
                window = saccfile.get_bandpower_windows(ind[mask])

                self.spec_meta.append({
                    "data_type": dt,
                    "tracer1": tr1,
                    "tracer2": tr2,
                    "pol": pol.lower(),
                    "ell": ls[mask],
                    "spec": mu[mask],
                    "idx": ind[mask],
                    "window": window
                })

                idx_max = max(idx_max, max(ind))

        self.data_vec = np.zeros((idx_max + 1,))
        self.spec_picker = np.zeros((idx_max + 1, len(self.spec_meta)))
        self.win_func = np.zeros((idx_max + 1, self.tt_lmax - 1))

        for i, m in enumerate(self.spec_meta):
            self.data_vec = self.data_vec.at[m["idx"]].set(m["spec"])
            self.spec_picker = self.spec_picker.at[m["idx"], i].set(1)

            j1, j2 = m["window"].values.min()-2, m["window"].values.max()-2
            self.win_func = self.win_func.at[m["idx"], j1:j2+1].set(
                m["window"].weight.astype(float).T
            )

        self.covmat = np.array(saccfile.covariance.covmat)

        for culls in self.cull:
            self.covmat = self.covmat.at[culls, :].set(0.0)
            self.covmat = self.covmat.at[:, culls].set(0.0)
            self.covmat = self.covmat.at[culls, culls].set(1e10)

        self.inv_cov = np.linalg.inv(self.covmat)
        self.logp_const = -0.5 * np.log(2.0 * np.pi) * len(self.data_vec)
        self.logp_const -= 0.5 * np.linalg.slogdet(self.covmat)[1]

    def logp(self, dell: np.ndarray) -> float:
        ps_vec = np.dot(self.win_func, dell[:self.tt_lmax-1])
        ps_vec = np.sum(ps_vec * self.spec_picker, axis=1)

        self.ps_vec = ps_vec

        delta = self.data_vec - ps_vec
        logp = -0.5 * (delta @ self.inv_cov @ delta)
        return self.logp_const + logp

    @property
    def verbose(self) -> bool:
        return self.__verbose
