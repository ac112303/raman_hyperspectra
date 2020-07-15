#!/usr/bin/env python

from setuptools import setup, find_packages

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

#requires = open(file_requirements).read().strip().split('\n')
    
# This setup is suitable for "python setup.py develop".

setup(name='raman_hyperspectra',
      version='0.0.70',
      description='toolbox for Raman spectra/hyperspectra processing and visualiation',
      long_description=long_description,
      long_description_content_type='text/markdown',
      package_data ={'RENISHAW_wdf':['raman_hyperspectra/data/RENISHAW_wdf/*.wdf']},
      license = 'MIT',
      classifiers = [
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Physics',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research'
        ],
      include_package_data = True,
      keywords = 'Raman, hyperspectra, data processing',
      install_requires = ['adjustText',
                          'BaselineRemoval',
                          'numpy',
                          'matplotlib',                          
                          'opencv-contrib-python',
                          'openpyxl',
                          'pandas',
                          'renishawWiRE',
                          'seaborn',
                          'scipy',
                          'sklearn',
                          'sporco',
                          'tkMagicGrid',
                          'tqdm',
                          'xarray',
                          'xlrd',],
      author= 'ArrayStream(François Bertin, Amal Chabli)',
      author_email= 'francois.bertin7@wanadoo.fr, amal.chabli@orange.fr',
      url= 'http://www.mymath.org/',
      packages=find_packages(),
      )
