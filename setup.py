#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 SKA South Africa
#
# This file is part of PolitsiyaKAT.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import subprocess
import os
from setuptools import setup

pkg='politsiyakat'

__version__ = "0.3.6"

build_root=os.path.dirname(__file__)

def src_pkg_dirs(pkg_name):
    mbdir = os.path.join(build_root, pkg_name)
    # Ignore
    pkg_dirs = []
    l = len(mbdir) + len(os.sep)
    exclude = ['docs', '.git', '.svn', 'CMakeFiles']
    for root, dirs, files in os.walk(mbdir, topdown=True):
        # Prune out everything we're not interested in
        # from os.walk's next yield.
        dirs[:] = [d for d in dirs if d not in exclude]

        for d in dirs:
            # OK, so everything starts with 'politsiyakat/'
            # Take everything after that ('src...') and
            # append a '/*.*' to it
            pkg_dirs.append(os.path.join(root[l:], d, '*.*'))
    print pkg_dirs
    return pkg_dirs

def define_scripts():
    #these must be relative to setup.py according to setuputils
    return [os.path.join(pkg,"scripts",script_name) for script_name in []]

setup(name=pkg,
      version=__version__,
      description='Tool to flag baselines with amplitude and phase problems',
      url='https://github.com/bennahugo/politsiyakat',
      download_url='https://github.com/bennahugo/politsiyakat/archive/0.2alpha.tar.gz',
      classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Astronomy"],
      author='Benjamin Hugo',
      author_email='bhugo@ska.ac.za',
      license='GNU GPL v3',
      packages=[pkg, pkg+"/data", pkg+"/modules", pkg+"/processing"],
      install_requires=['numpy==1.13.3',
                        'matplotlib==1.5.0',
                        'futures',
                        'python-casacore==2.1.2',
                        'scipy==1.0.0',
                        'progress'],
      package_data={pkg: src_pkg_dirs(pkg)},
      include_package_data=True,
      zip_safe=False,
      scripts=define_scripts()
)
