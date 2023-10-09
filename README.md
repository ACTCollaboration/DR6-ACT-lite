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
You can now run the tests with
```
pytest -v --pyargs act_dr6_cmbonly
```
If the tests return without any error (i.e. with only warnings), then the code is probably correctly installed. You can then attempt to run chains with
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
