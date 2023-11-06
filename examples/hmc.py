"""
This implementation is based on the HMC implementation made by the
Differentiable Universe Initiative, see
https://github.com/DifferentiableUniverseInitiative/jax-cosmo-paper/blob/master/notebooks/hmc.py

There are some differences (mostly a lot of stripped code for features I
did not deem necessary or easily readable).
"""

import jax
import jax.numpy as np
import tqdm
import datetime


class HMC:
    def __init__(self, logp, limits, key=jax.random.PRNGKey(0),
                 initial_point=None, verbose=False):

        self.verbose = verbose

        self.log("Computing initial gradient/hessian of logP.")
        self.logp = logp
        self.logp_grad = jax.jit(jax.grad(logp))
        self.logp_hess = jax.jit(jax.hessian(logp))

        self.log("Normalizing bounds.")
        self.limits = np.array(limits)
        if initial_point is None:
            self.x = np.array([(x1 + x2) / 2.0 for x1, x2 in limits])
        else:
            self.x = initial_point
        self.ndim = len(self.x)

        self.fisher = None
        self.covmat = None
        self.L = None
        self.Linv = None

        # Estimate the fisher matrix
        self.update_covmat()

        if self.covmat is None:
            raise ValueError("Covariance matrix is not decomposable. You \
                              might be in a saddle point. Try starting your \
                              chain from a different starting point.")

        self.leapfrog_steps = 1
        self.epsilon = 1.0

        self.trace_logP = []
        self.trace = []
        self.trace_accept = []

        self.n_accept = 0
        self.n_reject = 0

        self.key = key

        self.log("Initialized.")

    def update_covmat(self):
        self.log("Updating covmat.")
        fisher = -self.logp_hess(self.x)
        covmat = np.linalg.inv(fisher)
        L = np.linalg.cholesky(covmat)
        Linv = np.linalg.inv(L)

        if np.any(np.isnan(L)):
            self.log("Covariance matrix is not decomposable \
                      (might be in a saddle point).")
            self.log(f"Covmat diagonal: {np.diag(covmat)}")
            return

        self.fisher = fisher
        self.covmat = covmat
        self.L = L
        self.Linv = Linv

        self.log(f"Cholesky matrix = {self.L}.")

        self.x0 = self.x.copy()
        self.lower_bound_planes = []
        self.upper_bound_planes = []
        self.bound_normals = []

        for i in range(self.ndim):
            xb = self.x0.copy()
            xb = xb.at[i].set(self.limits[i, 1])
            uq = self.x2q(xb)
            xb = xb.at[i].set(self.limits[i, 0])
            lq = self.x2q(xb)

            n = np.zeros(self.ndim)
            n = n.at[i].set(1.0)
            m = self.L.T @ n
            m = m / np.linalg.norm(m)

            self.upper_bound_planes.append(uq)
            self.lower_bound_planes.append(lq)
            self.bound_normals.append(m)

        self.log("Done updating covmat.")

    def x2q(self, x):
        """ Transform a vector in parameter space to one in the space
            with diagonal mass. """
        return self.Linv @ (x - self.x0)

    def q2x(self, q):
        """ Transform a vector in the space with diagonal mass to
            parameter space. """
        return self.L @ q + self.x0

    def get_u(self, q):
        x = self.q2x(q)
        logP = self.logp(x)
        logP_grad = self.logp_grad(x)

        return -logP, -self.L.T @ logP_grad

    def advance(self, q, p):
        p = p.copy()
        q = q.copy()

        t0 = 0.0
        done = {}

        crossing = self.first_boundary_crossing(q, p, done)
        while crossing is not None:
            q, t_c, i = crossing
            t0 += t_c
            p = self.reflect(p, i)
            done[i] = crossing
            crossing = self.first_boundary_crossing(q, p, done)

        q = q + (self.epsilon - t0) * p

        return q, p

    def first_boundary_crossing(self, q, p, done):
        q_fin = q + self.epsilon * p
        x_fin = self.q2x(q_fin)

        ok = np.logical_and(x_fin > self.limits[:, 0],
                            x_fin < self.limits[:, 1])
        if np.all(ok):
            return None

        lam = np.zeros(self.ndim)
        for i in range(self.ndim):
            uq = self.upper_bound_planes[i]
            lq = self.lower_bound_planes[i]
            m = self.bound_normals[i]

            lam = lam.at[i].set((-(q - uq) @ m) / (p @ m))

            if not lam[i] > 0:
                lam = lam.at[i].set((-(q - lq) @ m) / (p @ m))

        i = lam.argmin()
        q_new = q + 0.99 * lam[i] * p

        return q_new, lam[i], int(i)

    def reflect(self, p, i):
        m = self.bound_normals[i]
        perp = (m @ p) * m
        par = p - perp
        return par - perp

    def integrate(self, q, p):
        U, gradU = self.get_u(q)

        H0 = 0.5 * (p @ p) + U

        for i in range(self.leapfrog_steps):
            p = p - 0.5 * self.epsilon * gradU
            q, p = self.advance(q, p)

            U, gradU = self.get_u(q)
            p = p - 0.5 * self.epsilon * gradU
            T = 0.5 * (p @ p)
            H = T + U

        return p, q, U, H, H0

    def random_kick(self):
        m = np.zeros(self.ndim)
        i = np.eye(self.ndim)
        self.key, subkey = jax.random.split(self.key)
        p = jax.random.multivariate_normal(subkey, m, i)
        return p

    def sample(self, n, burn_in=0, update_covmat_every=None,
               start=None, progress_bar=False):
        if start is None:
            if self.trace:
                start = self.trace[-1]
            else:
                start = self.x0.copy()
        else:
            start = np.array(start)

        print(f"Starting sampling at {start}.")

        q = self.x2q(start)
        self.n_accept = 0
        self.n_reject = 0
        n_burn_in_left = burn_in

        if progress_bar:
            pbar = tqdm.tqdm(total=n + burn_in)

        for j in range(n + burn_in):
            p = self.random_kick()

            p, q_new, U, H, H0 = self.integrate(q, p)
            log_alpha = H0 - H
            self.key, subkey = jax.random.split(self.key)
            p1 = jax.random.uniform(subkey)

            accept = log_alpha > np.log(p1)
            if accept:
                self.n_accept += 1
                q = q_new
            else:
                self.n_reject += 1

            self.x = self.q2x(q)

            if n_burn_in_left == 0:
                self.trace.append(self.x.copy())
                self.trace_logP.append(-U)
                self.trace_accept.append(accept)
            else:
                n_burn_in_left -= 1

            if update_covmat_every is not None and \
               n_burn_in_left == 0 and j > 0:
                if j % update_covmat_every == 0:
                    self.update_covmat()
                    q = self.x2q(self.x)

            if progress_bar:
                pbar.update(1)

        if update_covmat_every is not None:
            self.update_covmat()

        if progress_bar:
            pbar.close()

    def log(self, msg):
        if self.verbose:
            timestr = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-2]
            print(f"[{timestr}]: {msg}")
