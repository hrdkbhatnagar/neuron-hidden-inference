from setuptools import find_packages, setup
from glob import glob
from os.path import basename, splitext

setup(
    name='helper_functions',
    packages=find_packages(where = 'src'),
    version='0.1.1',
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')]
)