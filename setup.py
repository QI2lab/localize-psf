from setuptools import setup, find_packages
from os import path
from localize_psf import __version__

# copy readme to long description
this_dir = path.abspath(path.dirname(__file__))
with open(path.join(this_dir, "README.md")) as f:
    long_description = f.read()

# always required packages
required_pkgs = ['numpy',
                 'scipy',
                 'numba',
                 'matplotlib',
                 'joblib', # todo: only used in affine.py ransac. maybe replace with dask to simplify requirements
                 'dask',
                 'zarr'
                 ]

# optional required packages
# e.g. `pip install -e .[gpu]`
# todo: also need to include https://github.com/QI2lab/Gpufit with extras['gpu'] somehow if possible
# todo: usually better to install CuPy on your own
extras = {'gpu': ['cupy-cuda11x'],
          'psfmodels': ['psfmodels @ git+https://git@github.com/tlambert03/PSFmodels-py@main#egg=psfmodels'] # often have trouble installing this, so make optional
          }

setup(
    name='localize_psf',
    version=__version__,
    description="A package for localizing diffraction limited spots in 3D microscopy data.",
    long_description=long_description,
    author='qi2lab, Peter T. Brown',
    packages=find_packages(include=['localize_psf']),
    python_requires='>=3.9',
    install_requires=required_pkgs,
    extras_require=extras)
