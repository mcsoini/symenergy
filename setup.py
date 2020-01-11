import fnmatch
from setuptools import find_packages, setup
from setuptools.command.build_py import build_py as build_py_orig

with open("description_pypi.rst", "r") as fh:
    long_description = fh.read()

setup(
    name="SymEnergy",
    version="1.0.5",
    author="SymEnergy contributors listed in AUTHORS",
    author_email="m.c.soini@posteo.de",
    description=("Lagrange multiplier based energy market toy "
                 "modeling framework"),
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/mcsoini/symenergy",
    packages=find_packages(),
    install_requires=['pandas', 'orderedset', 'sympy', 'numpy', 'wrapt',
                      'matplotlib', 'bokeh'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"],
)
