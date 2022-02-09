import os

import setuptools


setuptools.setup(
    name='bunsen',
    version=os.environ.get('BUNSEN_VERSION'),
    description='A whole flaming mess of PyTorch utilities.',
    author='Raphael Tang',
    author_email='r33tang@uwaterloo.ca',
    license='MIT',
    package_data={'': ['*bunsen_internals.*so', '*bunsen_internals.*dll']},
    packages=['bunsen'],
    package_dir={'bunsen': 'bunsen/py'},
    python_requires='>=3.8'
)