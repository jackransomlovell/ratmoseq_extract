from setuptools import setup, find_packages
import subprocess
import codecs
import sys
import os


def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])


try:
    import cv2
except ImportError:
    install('opencv-python')

try:
    import cython
except ImportError:
    install('cython')

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

setup(
    name='ratmoseq-extract',
    author='Datta Lab',
    description='To boldly go where no RAT has gone before',
    version=get_version('ratmoseq_extract/__init__.py'),
    platforms=['mac', 'unix'],
    packages=find_packages(),
    install_requires=['h5py>=3.11.0', 'tqdm>=4.64.1', 'scipy==1.9.0', 'numpy>=1.22.4', 'click==8.1.3',
                      'joblib>=1.4.2', 'cytoolz==1.0.1', 'matplotlib>=3.8.2',
                      'scikit-image>=0.19.3', 'scikit-learn>=1.6.1', 'opencv-python>=4.5.5.64',
                      'ruamel.yaml>=0.18.10', 'jupyterlab>=4.3.4', 'pandas >= 2.2.3'],
    python_requires='>=3.10',
    entry_points={'console_scripts': ['ratmoseq-extract = ratmoseq_extract.cli:cli']},
    extras_require={
        "docs": [
            "sphinx",
            "sphinx-click",
            "sphinx-rtd-theme",
        ],
    },
)
