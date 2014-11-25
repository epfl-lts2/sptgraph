#!/usr/bin/env python
# -*- coding: utf-8 -*-


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


readme = open('README.rst').read()
history = open('HISTORY.rst').read().replace('.. :changelog:', '')

requirements = [
    # TODO: put package requirements here
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='sptgraph',
    version='0.1.0',
    description='Spatio-Temporal Graph repo',
    long_description=readme + '\n\n' + history,
    author='Kirell Benzi',
    author_email='kirell.benzi@epfl.ch',
    url='https://github.com/kikohs/sptgraph',
    packages=[
        'sptgraph',
    ],
    package_dir={'sptgraph':
                 'sptgraph'},
    include_package_data=True,
    install_requires=requirements,
    license="GPLv2",
    zip_safe=False,
    keywords='sptgraph',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GPL v2',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
