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
import imgaug
from mrcnn import utils
from mrcnn import visualize
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


def check_augmentation(dataset, augmentation):
    """
    Visualises effect of image agumentation on image from dataset

    Parameters
    ----------
    dataset : Mask R-CNN image collection
        Collection of images loaded and formatted using
        Mask R-CNN load_particles() function, part of the
        ParticlesDataset class.
    augmentation : image augmentation used with imgaug package
        DESCRIPTION.

    """
    image_id = random.choice(dataset.image_ids)
    original_image = dataset.load_image(image_id)
    original_mask, class_ids = dataset.load_mask(image_id)

    original_image_shape = original_image.shape
    original_mask_shape = original_mask.shape

    original_bbox = utils.extract_bboxes(original_mask)

    MASK_AUGMENTERS = [
        "Sequential",
        "SomeOf",
        "OneOf",
        "Sometimes",
        "Fliplr",
        "Flipud",
        "CropAndPad",
        "Affine",
        "PiecewiseAffine",
    ]

    def hook(images, augmenter, parents, default):
        return augmenter.__class__.__name__ in MASK_AUGMENTERS

    det = augmentation.to_deterministic()
    augmented_image = det.augment_image(original_image)
    augmented_mask = det.augment_image(
        original_mask.astype(np.uint8),
        hooks=imgaug.HooksImages(activator=hook)
    )
    augmented_bbox = utils.extract_bboxes(augmented_mask)

    # Verify that shapes didn't change
    assert (
        augmented_image.shape == original_image_shape
    ), "Augmentation shouldn't change image size"
    assert (
        augmented_mask.shape == original_mask_shape
    ), "Augmentation shouldn't change mask size"
    # Change mask back to bool
    # Display image and instances before and after image augmentation
    visualize.display_instances(
        original_image, original_bbox,
        original_mask, class_ids, dataset.class_names
    )
    visualize.display_instances(
        augmented_image, augmented_bbox,
        augmented_mask, class_ids, dataset.class_names
    )
