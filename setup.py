#!/usr/bin/env python
import os
from setuptools import setup, find_packages

REQUIREMENTS = [
    'pandas>=0.20.3',
    'seaborn>=0.8.1',
    'tqdm>=4.19.4',
    'IndicoIo>=1.1.5',
    'scikit-learn>=0.18.0',
    'numpy>=1.13.1',
    'click>=6.7',
    'bs4>=0.0.1',
    'finetune>=0.1.0'
]

setup(
    name='enso',
    version='0.1.6',
    description='Testing feature-sets and learning algorithms for transfer learning.',
    author='indico',
    author_email='engineering@indico.io',
    packages=find_packages(),
    install_requires=REQUIREMENTS
)