from setuptools import setup, find_packages
from os import path

# copy readme to long description
this_dir = path.abspath(path.dirname(__file__))
with open(path.join(this_dir, "README.md")) as f:
    long_description = f.read()

# always required packages
required_pkgs = ['numpy',
                 'scipy',
                 'matplotlib',
                 'joblib',
                 ]

# optional required packages
# e.g. `pip install -e .[gpu]`
extras = {'psfmodels': ['psfmodels @ git+https://git@github.com/tlambert03/PSFmodels-py@master#egg=psfmodels']} # often have trouble installing this, so make optional

setup(
    name='localize_psf',
    version='0.1.0',
    description="A package for localizing diffraction limited spots in 3D microscopy data.",
    long_description=long_description,
    author='qi2lab, Peter T. Brown, Alexis Coullomb',
    packages=find_packages(include=['localize_psf']),
    python_requires='>=3.7',
    install_requires=required_pkgs,
    extras_require=extras)

#@v1.1#egg=some-pkg
