"""Setup script to install numerical_methods package."""

from setuptools import setup, find_packages
setup(
    name="NumericalMethods",
    version="0.1",
    packages=find_packages(),
    install_requires=['numpy', 'matplotlib', 'scipy']
)
