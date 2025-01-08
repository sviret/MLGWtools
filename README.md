## Quick Installation instruction

### 1. Create a working directory and get the package

```
mkdir WORK
cd WORK
git clone https://github.com/sviret/MLGWtools.git
export PYTHONPATH=$PYTHONPATH:$PWD/MLGWtools
```

### 2. Create a conda environment with necessary packages installed

```
conda create --name GWtools python=3.11 -y
conda activate GWtools
pip install tensorflow
pip install pycbc
pip install numcompress
pip install gwpy
```

### 3. That's it, test that template generation macro works:

```
python MLGWtools/tests/template_example.py MLGWtools/tests/template.csv
```
