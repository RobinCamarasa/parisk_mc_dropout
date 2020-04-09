"""
**Author** : Robin Camarasa

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2020-03-23

**Project** : ##__NAME__##

**Code that creates a module out of the the code**

"""
import os
import sys
from subprocess import check_output

from setuptools import find_packages, setup

# Main path
root = os.path.dirname(os.path.abspath(__file__))

# Get long description
try:
    readme_path = os.path.join(root, 'README.md')
    with open(readme_path, 'r') as readme_handler:
        long_description = readme_handler.read()
except:
    long_description = 'pytest'


# Get version
try:
    command = 'git --git-dir {}/.git rev-parse HEAD'.format(
        root
    )
    with os.popen(cmd=command) as stream:
          version = stream.read()[:-1]
except:
    version='pytest'

# Get requirements
try:
    requirements_path = os.path.join(root, 'requirements.txt')
    with open(requirements_path, 'r') as requirements_handler:
        requirements = [
            dependency
            for dependency in requirements_handler.readlines()
            if not '##__NAME__##' in dependency
        ]
except:
    requirements=[]

setup(
    name='##__NAME__##',
    author='Robin Camarasa',
    version=version,
    packages=find_packages(),
    description='##__DESCRIPTION__##',
    long_description=long_description,
    install_requires=requirements,
    author_email='r.camarasa@erasmusmc.nl',
)

