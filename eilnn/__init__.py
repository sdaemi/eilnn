"""
# eilnn

A tool for segmenting and classifying particle data from tomography images of
battery electrodes using neural networks
"""
import annotation_creator
from .utils import load_grayscale, load_label, save_labels, view_GPU
import definitions

__version__ = "0.0.1"
