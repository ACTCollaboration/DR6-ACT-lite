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
You can install the requirements with
```
pip install -r requirements.txt
```
You need to (manually) install the data with
```
mkdir -p $COBAYA_PACKAGES_PATH/data/ACTDR6CMBonly
cp act_dr6_cmb_sacc.fits $COBAYA_PACKAGES_PATH/data/ACTDR6CMBonly/act_dr6_cmb_sacc.fits
```
And run the tests with
```
pytest -v
```
If the tests return without any error (i.e. with only warnings), then the code is correctly installed, and the dataset is working fine. You can then proceed to run chains with
```
cobaya-run run_act.yaml
```
(and other data combinations if you want).
