import jax
import jax.numpy as np
import act_dr6_cmbonly
import cosmopower_jax
from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ
import hmc


"""
Run an example Hamiltonian Monte-Carlo (HMC) chain with the
differentiable ACT DR6 likelihood.

Author: Hidde T. Jense
"""

"""
I have taken the liberty of creating an easily-readable HMC runner.
You can inspect the code in hmc.py, if you are interested.
"""

"""
You will need to download your own networks, since the only ones provided by
cosmopower-jax are not suitable for ACT DR6. See
<https://github.com/cosmopower-organization> for some pre-generated emulators
that are accurate enough for DR6 analysis.

Unfortunately, you will also need to manually copy these files to your
cosmpower_jax installation folder, printed out below:
"""
print(f"Searching for pre-installed networks near {cosmopower_jax.__file__}.")

"""
Let's say we installed these LCDM emulators:
<https://github.com/cosmopower-organization/lcdm/tree/main/TTTEEE>

If you want different emulators, be sure to edit the code as we go along!
"""
emu_tt = CPJ(probe="custom_log", filename="TT_v1.pkl")
emu_te = CPJ(probe="custom_pca", filename="TE_v1.pkl")
emu_ee = CPJ(probe="custom_log", filename="EE_v1.pkl")

"""
Let's quickly check which parameters we need...
"""
print(f"Parameters needed: {emu_tt.parameters}.")

"""
Ok, now we can setup our parameters & ranges.
"""
parameters = ["logA", "ns", "H0", "ombh2", "omch2", "tau"]
labels = [r"$\ln(10^{10} A_s)$", r"$n_s$", r"$H_0$",
          r"$\Omega_b h^2$", r"$\Omega_c h^2$", r"$\tau$"]
ranges = [(2.5, 4.5), (0.9, 1.1), (50.0, 90.0),
          (0.01, 0.1), (0.05, 0.5), (0.0, 0.1)]
initial_point = np.array([3.0, 0.95, 66.0, 0.022, 0.12, 0.05])
T_CMB = 2.7255e6  # muK

"""
And we can initialize our likelihood.
"""
like = act_dr6_cmbonly.ACTDR6jax()
like.data_folder = "../act_dr6_cmbonly/data"
like.load_data()

"""
Now we can setup our priors and full log-posterior function. Both these
functions need the callsign f(x) = y, where x is an array and y is a float.
"""


@jax.jit
def logprior(x):
    p = 0.0
    # Tau prior
    p -= 0.5 * ((x[5] - 0.0544) / 0.0073) ** 2.0
    return p


@jax.jit
def logp(x):
    cell_tt = (T_CMB) ** 2.0 * emu_tt.predict(x)
    cell_te = (T_CMB) ** 2.0 * emu_te.predict(x)
    cell_ee = (T_CMB) ** 2.0 * emu_ee.predict(x)

    cell = np.stack([cell_tt, cell_te, cell_ee], axis=1)

    return like.logp(cell) + logprior(x)


"""
Now we can setup our chain.
The parameters are:
 - the log-posterior function,
 - the hard prior bounds on your parameters,
 - the initial point for your chain (if not given, it chooses the middle of
    your prior bounds).
 - you can add extra verbose output with verbose=True
"""
chain = hmc.HMC(logp, ranges, initial_point=initial_point, verbose=True)

"""
It can happen that your chain crashes upon initialization because it cannot
decompose the covariance matrix. If this happens, check which parameters have
a negative covariance (it should be printed in the output), and fiddle with
these parameters in your initial point - your chain started in a saddle point.
"""

"""
At this point, we can already do some fun things like a straight fisher
forecast:
"""
error_bars = np.sqrt(np.diag(chain.covmat))

print("Fisher errors:")
for i, n in enumerate(parameters):
    print(f"{n:10s} = {error_bars[i]:g}")

"""
Now we can start sampling!
For a LCDM chain, I found that 5000 samples (+1000 burn-in samples) is more
than enough for a smooth contour. Even if you are not using a GPU, this should
take ~10 minutes to run. You can tweak the update_covmat_every parameter
to change how often your runner should re-evaluate the covariance of your
chain (while not strictly necessary, it can slightly improve acceptance rates).
"""
# HMC stepsize parameters.
chain.leapfrog_steps = 5
chain.epsilon = 0.1

chain.sample(5000, burn_in=1000, update_covmat_every=500, progress_bar=True)

"""
Let's get some output!
"""
samples = np.array(chain.trace)
log_posterior = np.array(chain.trace_logP)
accept = np.array(chain.trace_accept)
data = np.concatenate((accept[..., np.newaxis], samples,
                       log_posterior[..., np.newaxis]), axis=1)

np.save("chain.npy", data)

"""
We now saved everything to our chain.npy file with the columns:
- 1. whether our chain was accepted
- 2. the raw samples of our chain
- 3. the log-posterior of each sample.
You can load these into getdist or something to make a nice plot, similar
to how you plot cobaya chains!
"""

"""
As a final step, we can check the convergence statistics of our chain.
"""
data_mean = np.mean(samples, axis=0)
data_std = np.std(samples, axis=0)

for i, n in enumerate(parameters):
    print(f"{n:10s} = {data_mean[i]:.2g} +/- {data_std[i]:.3g}")
