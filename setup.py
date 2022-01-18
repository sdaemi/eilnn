import os
import sys
from distutils.util import convert_path

sys.path.append(os.getcwd())

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

main_ = {}
ver_path = convert_path("eilnn/__init__.py")
with open(ver_path) as f:
    for line in f:
        if line.startswith("__version__"):
            exec(line, main_)

setup(
    name="eilnn",
    description=("A neural network package for segmenting and labelling" +
                 " battery particle images"),
    version=main_["__version__"],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
    ],
    packages=["eilnn"],
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pillow",
        "pandas",
        "shapely",
        "opencv-python",
        "scikit-image",
        "scikit-learn",
    ],
    author="Sohrab Daemi",
    author_email="sohrab.daemi.14@ucl.ac.uk",
    url="https://eilnn.readthedocs.io/en/latest/",
    project_urls={
        "Documentation": "https://eilnn.readthedocs.io/en/latest/",
        "Source": "https://github.com/sdaemi/eilnn",
        "Tracker": "https://github.com/sdaemi/eilnn/issues",
    },
)
