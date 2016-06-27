#!/usr/bin/env python

from setuptools import setup
from setuptools.command.test import test as TestCommand

class NoseTestCommand(TestCommand):

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import nose
        nose.run_exit(argv=['nosetests'])

setup(name='PRIM',
      version='0.4',
      description='''This module implements the Patient Rule Induction Method
                   (PRIM) for scenario discovery in Python.  This is a
                   standalone version of the PRIM algorithm implemented in the
                   EMA Workbench by Jan Kwakkel, which is based on the
                   sdtoolkit R package developed by RAND Corporation.  All
                   credit goes to Jan Kwakkel for developing the original code.
                   This standalone version of PRIM was created and maintained
                   by David Hadka.''',
      author='David Hadka, based on PRIM code by Jan Kwakkel (EMA Workbench)',
      author_email='dhadka@users.noreply.github.com',
      license="GNU GPL version 3",
      url='https://github.com/Project-Platypus/PRIM',
      packages=['prim'],
      install_requires=[
          'matplotlib',
          'numpy',
          'pandas',
          'mpldatacursor',
          'six',
          'scipy'],
      tests_require=['nose', 'mock'],
      cmdclass={'test': NoseTestCommand},
      classifiers=[
          'Development Status :: 4 - Beta',
          'Environment :: Console',
          'Intended Audience :: Education',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Software Development :: Libraries :: Python Modules',
          ],
     )