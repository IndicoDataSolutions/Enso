#!/usr/bin/env python
import os
from setuptools import setup, find_packages

setup(
    name='enso',
    version='0.1.0',
    description='Testing feature-sets and learning algorithms for transfer learning.',
    author='indico',
    author_email='engineering@indico.io',
    packages=find_packages(),
    install_requires=open(
        os.path.join(
            os.path.dirname(__file__),
            "requirements.txt"
        ), 'r'
    ).readlines(),
)