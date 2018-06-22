#!/usr/bin/env python
import os
from setuptools import setup, find_packages

REQUIREMENTS = open('requirements.txt').readlines()

setup(
    name='enso',
    version='0.1.4',
    description='Testing feature-sets and learning algorithms for transfer learning.',
    author='indico',
    author_email='engineering@indico.io',
    packages=find_packages(),
    install_requires=REQUIREMENTS
)