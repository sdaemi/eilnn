# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 13:39:36 2021
Collection of tools.

@author: Sohrab Daemi
"""
import os
import cv2
import numpy as np
import shutil
import matplotlib.pyplot as plt
import random

# from mrcnn.visualize import display_images


def load_grayscale(grayscale_path):
    """
    Loads grayscale images into numpy array.

    Parameters
    ----------
    grayscale_path : string
        Path to folder containing numbered grayscale slices in ascending order.
        Ensure there are no numbered folders in the directory.

    Returns
    -------
    image_stack : numpy array
        3D grayscale image stack

    """
    print("Loading images in: " + grayscale_path)

    image_stack = np.asarray(
        [
            cv2.imread(os.path.join(grayscale_path, i), 1)
            for i in os.listdir(grayscale_path)
            if str("".join(filter(str.isdigit, i)))
        ]
    )

    middle_slice = int(image_stack.shape[0] / 2)
    plt.imshow(image_stack[middle_slice])
    print("Stack Dimensions ", str(image_stack.shape))

    if len(image_stack.shape) < 3:
        raise ValueError(
            "Image stack is not 3D/has not been loaded properly - \
                ensure there are no numbered folders contained in \
                the stack folder"
        )

    return image_stack


def load_label(label_path):
    """
    Loads label stack into numpy array.

    Parameters
    ----------
    label_path : string
        Path to folder containing numbered label slices in ascending order.
        Ensure there are no numbered folders in the directory.


    Returns
    -------
    label_stack : numpy array
        3D 8bit label stack

    """
    print("Loading images in: " + label_path)

    label_stack = np.asarray(
        [
            cv2.imread(os.path.join(label_path, i), 0)
            for i in os.listdir(label_path)
            if str("".join(filter(str.isdigit, i)))
        ]
    )

    middle_slice = int(label_stack.shape[0] / 2)
    plt.imshow(label_stack[middle_slice])
    print("Stack Dimensions ", str(label_stack.shape))

    if len(label_stack.shape) < 3:
        raise ValueError(
            "Label stack is not 3D/has not been loaded properly -\
            ensure there are no numbered files contained in the stack folder"
        )
    # Turn label stack into binary (0, 1), regardless of input
    label_stack = np.where(label_stack > 0, 1, 0)

    return label_stack


def save_label(label_field, label_folder):
    """

    Save label fields to new folder
    Parameters
    ----------
    label_field : numpy array
        8bit numpy array containing label stack for export.
    label_folder : string
        Path to which label stack is saved to.

    """

    label_field_export = ((label_field * 1) * 255).astype("uint8")
    out_dir = label_folder

    if len(label_field_export.shape) < 3:
        raise ValueError("Stack for export has incorrect dimensions.")

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
        os.mkdir(out_dir)
    else:
        os.mkdir(out_dir)

    print("Saving labels in " + out_dir)

    for i in range(0, label_field.shape[0]):
        filename = "label_" + str(i) + ".tiff"
        save_path = os.path.join(out_dir, filename)
        cv2.imwrite(save_path, label_field_export[i, :, :])


def view_GPU():
    """
    Prints list of availabile GPU's (if any)

    Returns
    -------
    None.

    """
    import tensorflow as tf

    print(tf.__version__)
    from tensorflow.python.client import device_lib

    print(device_lib.list_local_devices())