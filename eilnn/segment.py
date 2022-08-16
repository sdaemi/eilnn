# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 12:26:34 2022

@author: Sohrab
"""

import os
import sys
import random
import math
import re
import time
import numpy as np
import skimage
import imgaug
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import shutil
import timeit
import eilnn

from samples import particles
from numba import cuda
import timeit
from scipy.ndimage import gaussian_filter
from skimage.morphology import (
    remove_small_holes,
    binary_opening,
    binary_dilation,
    binary_erosion,
    label,
    reconstruction,
    local_minima,
)
from skimage.segmentation import watershed
from skimage.measure import block_reduce
import eilnn
from scipy.ndimage import distance_transform_edt as edt
import cc3d


class Segment_3D:
    def __init__(self, model, image_path, marker_size=2):
        self.marker_size = marker_size
        self.model = model
        self.grayscale_data = eilnn.load_grayscale(image_path)

    def segment_2d(self, model_inference, image_stack):
        """
        Basic segmentation block that operates on individual slices.

        Parameters
        ----------
        model_inference : Mask R-CNN model
            Pre-trained Mask R-CNN model.
        image_stack : ndarray
            Grayscale image stack.

        Returns
        -------
        label_field : ndarray
            segmented binary label field.

        """
        label_field = np.zeros(
            (image_stack.shape[1], image_stack.shape[2], image_stack.shape[0]),
            dtype=bool,
        )

        for i in range(0, image_stack.shape[0]):

            r = model_inference.detect([image_stack[i, :, :, :]])
            r = r[0]

            if r["masks"].shape[2] == 0:
                continue
            elif r["masks"].shape[2] > 0:
                for j in range(0, r["masks"].shape[2]):
                    swapped_r = np.swapaxes
                    label_field[:, :, i] = label_field[:, :, i] + r["masks"][:, :, j]

        return label_field

    def segment_xz(self, model_inference, image_stack):
        """
        Rotates and resamples the image stack in the xz direction, segments and restores original orientation

        Parameters
        ----------
        model_inference : Mask R-CNN model
            Pre-trained Mask R-CNN model.
        image_stack : ndarray
            Grayscale image stack.

        Returns
        -------
        label_field : ndarray
            segmented and resampled binary label field.

        """
        stack_resampled = np.swapaxes(np.swapaxes(image_stack, 0, 2), 1, 2)
        stack_resampled = np.flip(np.flip(stack_resampled, 1), 2)
        label_field = self.segment_2d(model_inference, stack_resampled)
        label_field = np.flip(np.flip(label_field, 0), 1)
        label_field = np.moveaxis(label_field, 0, -1)
        return label_field

    def segment_yz(self, model_inference, image_stack):
        """
        Rotates and resamples the image stack in the yz direction, segments and restores original orientation

        Parameters
        ----------
        model_inference : Mask R-CNN model
            Pre-trained Mask R-CNN model.
        image_stack : ndarray
            Grayscale image stack.

        Returns
        -------
        label_field : ndarray
            segmented and resampled binary label field.

        """
        stack_resampled = np.swapaxes(image_stack, 0, 2)
        stack_resampled = np.swapaxes(stack_resampled, 1, 2)
        stack_resampled = np.flip(stack_resampled, 1)
        label_field = self.segment_2d(model_inference, stack_resampled)
        label_field = np.moveaxis(label_field, 0, -1)
        label_field = np.flip(label_field, -1)
        return label_field

    def union(self, label_field, label_xz, label_yz):
        """
        Adds original and resampled label fields together to compensate for any slice-like artefacts.

        Parameters
        ----------
        image_stack : ndarray
            Grayscale image stack.
        label_xz : ndarray
            resampled label field.
        label_yz : ndarray
            resampled label field.

        Returns
        -------
        label : ndarray
            union of resampled (3) label fields.

        """
        label = label_field + label_xz + label_yz
        label = np.moveaxis(label, -1, 0)
        label[label > 0] = 1
        return label

    def segment_3d(self, model, data):
        """
        

        Parameters
        ----------
        model : Mask R-CNN model
            Pre-trained Mask R-CNN model.
        data : ndarray
            grayscale image stack.

        Returns
        -------
        label_3d : ndarray
            binary segmentation in three directions.

        """
        label = self.segment_2d(model, data)
        label_xz = self.segment_xz(model, data)
        label_yz = self.segment_yz(model, data)
        label_3d = self.union(label, label_xz, label_yz)
        return label_3d

    def filtering(self, labels):
        """
        Filters the binary label image to remove slice artefacts
        combination of opening, guassian filters, erosion and hole removal

        Parameters
        ----------
        labels : ndarray
            3D segmented and resampled label stack.

        Returns
        -------
        label_filtered : ndarray
            filtered segmented label stack.

        """
        label_filtered = binary_opening(labels)
        label_filtered = gaussian_filter(label_filtered, sigma=0.75)
        for i in range(1, 3):
            label_filtered = binary_dilation(label_filtered)

        label_filtered = remove_small_holes(label_filtered)

        for i in range(1, 3):
            label_filtered = binary_erosion(label_filtered)

        label_filtered = binary_opening(label_filtered)
        label_filtered = remove_small_holes(label_filtered)

        for i in range(1, 6):
            label_filtered = binary_dilation(label_filtered)

        label_filtered = gaussian_filter(label_filtered, sigma=0.1)
        label_filtered = remove_small_holes(label_filtered)

        return label_filtered

    def display_filters(self, grayscale_stack, label_3d, label_filtered):
        """
        Displays the dataset before and after filtering.

        Parameters
        ----------
        grayscale_stack : ndarray
            grayscale stack over which segmented versions are overlayed.
        label_3d : ndarray
            Unfiltered label field.
        label_filtered : TYPE
            Filtered label field.

        Returns
        -------
        None.

        """
        middle_slice = int(grayscale_stack.shape[0] / 2)
        ix = np.random.randint(
            middle_slice - middle_slice * 0.1, middle_slice + middle_slice * 0.1
        )
        plt.subplot(1, 2, 1)
        plt.title("No Filter Applied")
        plt.imshow(grayscale_stack[ix, :, :], alpha=1)
        plt.imshow(label_3d[ix, :, :], alpha=0.4, cmap="Reds")

        plt.subplot(1, 2, 2)
        plt.title("After Filtering")
        plt.imshow(grayscale_stack[ix, :, :], alpha=1)
        plt.imshow(label_filtered[ix, :, :], alpha=0.4, cmap="Reds")
        plt.show()

    def extendedmin(self, img, h):
        """
        Finds the extended minima for the watershed segmentation

        Parameters
        ----------
        img : ndarray
            3D binary segmentation image stack.
        h : int
            marker size for watershed segmentation.

        Returns
        -------
        local minima


        """
        mask = img.copy()
        marker = mask + h
        hmin = reconstruction(marker, mask, method="erosion")
        return local_minima(hmin)

    def watershed_sep(self, im, marker_size=2):
        """
        watershed segmentation

        Parameters
        ----------
        im : ndarray
            3D binary segmentation image stack..
        marker_size : int, optional
            watershed marker size. The default is 2.

        Returns
        -------
        W : ndarray
            watershed segmentation.

        """
        D = -edt(im)
        print("Distance Transform Completed")
        mask = self.extendedmin(D, marker_size)
        print("Extended Minima Found")
        mlabel, N = label(mask, return_num=True)

        W = watershed(D, markers=mlabel, mask=im)
        print("Watershed Separation Completed")
        return W

    def process_segm(self):
        """
        Processes all steps required to import grayscale data and returns binary segmentation.

        Returns
        -------
        particles : ndarray
            individually  grayscale particles.
        masks : ndarray
            individually segmented particles.
        separated_stack : ndarray
            3D stack of watershed segmented data.

        """
        start = timeit.default_timer()
        print("Commencing 3D segmentation.")
        labels = self.segment_3d(self.model, self.grayscale_data)
        print("Filtering labels.")
        labels_filtered = self.filtering(labels)

        print("Label array dimensions " + str(labels.shape) + "\n")
        self.display_filters(self.grayscale_data, labels, labels_filtered)
        separated_stack = self.watershed_sep(labels_filtered, self.marker_size)

        labelled_stack = cc3d.connected_components(labels_filtered, connectivity=18)
        stats = cc3d.statistics(labelled_stack)
        bounding_boxes = stats["bounding_boxes"]
        images = []
        increment = 0
        particles = {}
        masks = {}
        for particle, box in enumerate(bounding_boxes):
            x = box[0]
            y = box[1]
            z = box[2]

            image = self.grayscale_data[
                int(x.start) - increment : int(x.stop) + increment,
                int(y.start) - increment : int(y.stop) + increment,
                int(z.start) : int(z.stop),
            ]

            labelled_image = separated_stack[
                int(x.start) - increment : int(x.stop) + increment,
                int(y.start) - increment : int(y.stop) + increment,
                int(z.start) : int(z.stop),
            ]

            particles[particle] = image
            masks[particle] = labelled_image

        stop = timeit.default_timer()
        print("Segmentation took", np.round((stop - start) / 60, 1), "minutes")

        return particles, masks, separated_stack


if __name__ == "__main__":
    root = eilnn.IMAGES_DIR
