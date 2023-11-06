# ACT DR6 CMB-Only Likelihood
_Cobaya likelihood for the DR6 foreground-marginalized data._

Author: Hidde T. Jense

**See also [the wiki page](https://phy-wiki.princeton.edu/polwiki/pmwiki.php?n=ACTPolPowerSpectrumTelecon.DR6CMBExtract) for this code!**

## Contents

This is a [cobaya](https://cobaya.readthedocs.io/en/latest/) likelihood for the foreground-marginalized DR6 data. It will be in active, continuous development until the DR6 data release.

## Installation Instructions

The easiest way to install the code is to first clone the repository somewhere with
```
git clone git@github.com:ACTCollaboration/dr6-cmbonly.git .
```
You can install the package then as
```
pip install -e .
```
This will locally pip-install the dataset.
If you want to use the Planck dataset cut at the ACT multipole, you need to install the data with
```
cobaya-install act_dr6_cmbonly.PlanckActCut
```
You can now run the tests with
```
pytest -v --pyargs act_dr6_cmbonly
```
If the tests return without any error (i.e. with only warnings), then the code is probably correctly installed. You may get some tests which get skipped if you do not install the differentiable likelihood (see below) - you do not need to worry about this. You can then attempt to run chains with
```
cobaya-run yamls/run_act.yaml
```
(and other data combinations if you want, see the files `yamls/`).

## The data path

By default, the dataset will be stored inside and loaded from the pip dist (the directory where pip installed your ACT DR6 cmbonly likelihood). It may be necessary to obey the cobaya convention and put everything inside your packages path (this is the directory where cobaya will look for data). For this, you will have to manually move the dataset over there with
```
mkdir -p $COBAYA_PACKAGES_PATH/data/ACTDR6CMBonly
cp act_dr6_cmbonly/data/act_dr6_cmb_sacc.fits $COBAYA_PACKAGES_PATH/data/ACTDR6CMBonly/act_dr6_cmb_sacc.fits
```
If you ever get an error that the likelihood cannot locate the data, then you can add a `--debug` flag to your cobaya call, which will make the likelihood output a message for the exact path where it is trying to locate the data from.

By default, the likelihood will look for the data in either
- `<pip directory>/act_dr6_cmbonly/data/` if no cobaya packages path is given, or
- `<cobaya packages path>/data/ACTDR6CMBonly/` if a cobaya packages path is given.

## The differentiable likelihood

If you are for whatever reason interested, I created a differentiable likelihood as well. You can install this by installing the package with
```
pip install -e .[jax]
```
Which will also install the `jax` and `cosmopower-jax` prerequisites. The differentiable likelihood can then be imported with
```
import act_dr6_cmbonly
like = act_dr6_cmbonly.ACTDR6jax()
```
I provide an example of how to run a chain with the differentiable likelihood, see the `examples/run_hmc.py` file.

For the most part, I do not expect that a differentiable likelihood adds much to ACT DR6 on its own. However, should people be interested in running joint analyses with other probes that provide differentiable likelihoods, then this simple likelihood should suffice. For the most part, it is simply 100 lines of python that do the same as the cobaya likelihood, but with JAX instead of numpy (as a result, some of the data is stored a bit differently internally to make use of JAX optimizations).
