### Installing superbit_metacal

1. First create a new conda environment with python 3.9 and activate the environment

```bash
conda create -n superbit python=3.9
conda activate superbit
```

1. Install the following packages (make sure the numpy version≥ 1.24):

```bash
pip install esutil
pip install meds
pip install pyyaml
pip install astropy --upgrade
pip install fitsio
pip install piff
pip install shapely
pip install Rtree
conda install conda-forge::astromatic-swarp
conda install conda-forge::astromatic-source-extractor
```

1. Download and Install psfex

```bash
cd ..
git clone https://github.com/esheldon/psfex.git
cd psfex
# to install globally
python setup.py install
```

1. Download and install super-bit metacal and make sure you are in the process-real branch

```bash
git clone https://github.com/superbit-collaboration/superbit-metacal.git
cd superbit-metacal/
git checkout process-real
pip install -e .
```
