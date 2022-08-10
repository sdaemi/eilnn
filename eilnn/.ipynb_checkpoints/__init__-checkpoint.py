"""
# eilnn

A tool for segmenting and classifying particle data from tomography images of
battery electrodes using neural networks
"""
from .annotation_creator import *
from .utils import *
from .definitions import *
from .segment import *
from . tflog2pandas import *

__version__ = "0.0.1"
