# ACT DR6 CMB-Only Likelihood
_Cobaya likelihood for the DR6 foreground-marginalized data._

Author: Hidde T. Jense

**See the ACT DR6 power spectra and likelihoods paper (Louis, La Posta, Atkins, Jense et al., 2025) for a full description.**


## Contents

This is a [cobaya](https://cobaya.readthedocs.io/en/latest/) likelihood for the foreground-marginalized DR6 data.


## Installation Instructions

The easiest way to install the code is to first clone the repository somewhere with
```
git clone git@github.com:ACTCollaboration/DR6-ACT-lite.git .
```
You can install the package then as
```
pip install -e .
```
This will locally pip-install the dataset.

At this point, you can install the likelihood data with
```
cobaya-install act_dr6_cmbonly
```
Which will download the correct data file and install it at your cobaya packages path.

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


### Manual installation: The data path

If you cannot access the data, there is a simulated test dataset stored inside and loaded from the pip dist (the directory where pip installed your ACT DR6 lite likelihood). It may be necessary to obey the cobaya convention and put everything inside your packages path (this is the directory where cobaya will look for data). For this, you will have to manually move the dataset over there with
```
mkdir -p $COBAYA_PACKAGES_PATH/data/ACTDR6CMBonly
cp act_dr6_cmbonly/data/dr6_data_cmbonly.fits $COBAYA_PACKAGES_PATH/data/ACTDR6CMBonly/dr6_data_cmbonly.fits
```
If you ever get an error that the likelihood cannot locate the data, then you can add a `--debug` flag to your cobaya call, which will make the likelihood output a message for the exact path where it is trying to locate the data from.

By default, the likelihood will look for the data in either
- `<pip directory>/act_dr6_cmbonly/data/` if no cobaya packages path is given, or
- `<cobaya packages path>/data/ACTDR6CMBonly/` if a cobaya packages path is given.

