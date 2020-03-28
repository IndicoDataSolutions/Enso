#!/usr/bin/env python
import os
from setuptools import setup, find_packages

REQUIREMENTS = [
    'pandas>=0.24.2',
    'seaborn>=0.8.1',
    'tqdm>=4.19.4',
    'IndicoIo>=1.1.5',
    'scikit-learn==0.22.0',
    'numpy>=1.13.1',
    'click>=6.7',
    'bs4>=0.0.1',
    'absl-py',
    # will still need to download tensorflow
    'finetune @ git+https://github.com/IndicoDataSolutions/finetune/tarball/aux_mlm'
    # 'finetune>=0.3.0'

]

setup(
    name='enso',
    version='0.1.6',
    description='Testing feature-sets and learning algorithms for transfer learning.',
    author='indico',
    author_email='engineering@indico.io',
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    dependency_links=['git+https://github.com/IndicoDataSolutions/finetune@aux_mlm']
)
