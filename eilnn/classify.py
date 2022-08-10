# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 11:30:47 2022

@author: Sohrab
"""

import numpy as np
import eilnn
import tensorflow as tf
from tensorflow import keras
import os
import matplotlib.pyplot as plt
import matplotlib as mlt
import pandas as pd
import cv2
from skimage import filters
from scipy.ndimage.morphology import binary_dilation, binary_erosion
import shutil
import timeit
import scipy



class classifier:
    def __init__(self, model_class, particles, masks, export_path=""):
        """
        Initialise the class with the classifier model, particles and mask array,
        export path to save the data and save option.

        Parameters
        ----------
        model_class : string
            classifier model path
        particles : dict
            dictionary containing individual grayscale slices 
        masks : TYPE
            dictionary containing individual mask slices 
        export_path : TYPE, optional
            path to export individual 3-phase segmentations. The default is '', and no saving will occur.
      

        Returns
        -------
        None.

        """
        self.particles = particles
        self.masks = masks
        self.model = model_class
        self.model = keras.models.load_model(model_class)
        self.background = self.particles[0][0, 0, 0][0]
        self.export_path = export_path

    def dict_to_numpy(self, particles):
        """
        Convert dictionary pickles to numpy array

        Parameters
        ----------
        particles : dict
            dict containing particle or mask arrays.

        Returns
        -------
        particles_numpy : ndarray
            particle dictionary converted to numpy array.

        """
        particle_values = particles.values()
        particles_list = []
        for i in particles.values():
            particles_list.append(np.asarray(i))

        particles_numpy = []
        if len(particles_list[0].shape) == 4:
            for i, j in enumerate(particles_list):
                particles_numpy.append(np.asarray(j[:, :, :, 1]))
        elif len(particles_list[0].shape) == 3:
            for i, j in enumerate(particles_list):
                particles_numpy.append(np.asarray(particles_list[i][:, :, :]))

        particles_numpy = np.asarray(particles_numpy, dtype="object")
        return particles_numpy

    def remove_border(self, particles):
        """
        Remove greyscale slices where particles are truncated by the border. 
        Checks if the total area of the bordering pixels exceeds 18% per slice.

        Parameters
        ----------
        particles : ndarray
            

        Returns
        -------
        internal_particles : ndarray
            particles without bordering pixels.
        idx_remove : TYPE
            index of particles that are removed, used to remove the matching masks.

        """
        internal_particles = []
        idx_remove = []

        for idx, particle in enumerate(particles):

            middle_idx = int(particle.shape[0]) // 2
            middle_slice = particle[middle_idx, :, :]
            total_pixels_image = particle[0, :, :].size
            image_values, counts = np.unique(middle_slice, return_counts=True)
            backround_idx = np.where(image_values == self.background)
            background_pixels = counts[backround_idx]
            background_pc = background_pixels / total_pixels_image * 100

            if background_pc < 18 or not background_pc:
                internal_particles.append(particle)

            elif background_pc >= 18:
                idx_remove.append(idx)

        print(
            str(len(particles) - len(internal_particles))
            + " partial particles removed."
        )
        internal_particles = np.asarray(internal_particles, dtype="object")
        return internal_particles, idx_remove

    def remove_mask_border(self, masks, idx):
        """
        Remove masks of particles that had bordering pixels (remove_border)

        Parameters
        ----------
        masks : ndarray
            
        idx : list
            

        Returns
        -------
        masks_removed : ndarray
            masks wtihout slices with >18% out of FOV pixels.

        """
        masks_removed = [masks[i] for i in range(0, masks.shape[0]) if i not in idx]
        masks_removed = np.asarray(masks_removed, dtype='object')
        return masks_removed

    def resize_slice(self, particles, resize_row=128, resize_col=128):
        """
        Resize grayscale slices for classifier input (128 x 128 pixels)

        Parameters
        ----------
        particles : ndarray
            
        resize_row : int, optional
             The default is 128.
        resize_col : int, optional
             The default is 128.

        Returns
        -------
        particles_resized : ndarray
             resized particles.

        """
        particles_resized = []
        for particle in particles:
            particle_resized = []
            for slice_idx in range(0, particle.shape[0]):

                resized_slice = cv2.resize(
                    particle[slice_idx, :, :],
                    (resize_row, resize_col),
                    interpolation=cv2.INTER_CUBIC,
                )
                particle_resized.append(np.asarray(resized_slice))
            particles_resized.append(np.asarray(particle_resized))
        particles_resized = np.asarray(particles_resized)
        return particles_resized

    def resize_mask(self, masks, resize_row=128, resize_col=128):
        """
        Resize mask slices for classifier input (128 x 128 pixels)


        Parameters
        ----------
        masks : ndarray
            
        resize_row : int, optional
            The default is 128.
        resize_col : int, optional
            The default is 128.

        Returns
        -------
        masks_resized : ndarray
            resized masks.

        """
        masks_resized = []
        for mask in masks:
            mask_resized = []
            for slice_idx in range(0, mask.shape[0]):

                resized_slice = mask[slice_idx, :, :].copy()
                resized_slice.resize(128, 128)

                mask_resized.append(np.asarray(resized_slice))
            masks_resized.append(np.asarray(mask_resized))
        masks_resized = np.asarray(masks_resized)
        return masks_resized

    def remove_outliers(self, particles_numpy):
        """
        If slice artefacts are introduced during the CV segmentation step,
        ­these are removed by measuring the Mahalanobis distance (multivariate outlier detection method.)
        Parameters
        ----------
        particles_numpy : ndarray
            

        Returns
        -------
        particles_removed : ndarray
            Particles without slice artefacts.

        """

        y_dims = [
            particles_numpy[i].shape[1] for i in range(0, particles_numpy.shape[0])
        ]
        x_dims = [
            particles_numpy[i].shape[2] for i in range(0, particles_numpy.shape[0])
        ]
        #plt.scatter(x_dims, y_dims)
        #plt.xlabel("X axis size / px")

        #plt.ylabel("Y axis size / px")
        #plt.show()


        def mahalanobis_dist(x=None, data=None, cov=None):
            x_minus_mu = x - np.mean(data)

            if not cov:
                cov = np.cov(data.values.T)
            inv_covmat = scipy.linalg.inv(cov)
            left_term = np.dot(x_minus_mu, inv_covmat)
            mahal_dist = np.dot(left_term, x_minus_mu.T)
            return mahal_dist

        Xy = np.column_stack([x_dims, y_dims])
        Xy_pd = pd.DataFrame(Xy)

        m_dist = []
        for i, j in enumerate(Xy):
            m_dist.append(mahalanobis_dist(j, Xy_pd))
        m_dist = np.asarray(m_dist)

        outliers = np.squeeze(np.where(m_dist > 20, 0, 1))
        outliers = outliers[:, None]

        xy_removed = Xy * outliers
        xy_removed = xy_removed[~np.all(xy_removed == 0, axis=1)]

        idx = np.arange(0, outliers.shape[0])
        idx = idx[np.newaxis].T
        idx = idx * outliers
        idx = idx[~np.all(idx == 0, axis=1)]

        xy_idx = np.column_stack([xy_removed, idx])

        #plt.scatter(xy_removed[:, 0], xy_removed[:, 1])
        #plt.xlabel("X axis size / px")
        #plt.ylabel("Y axis size / px")
        #plt.show()
        particles_removed = np.take(particles_numpy, np.squeeze(idx), axis=0)
        return particles_removed

    def filter_clahe(self, particles, limit=3):
        """
        Contrast Limited AHE to improve grayscale slices for classification

        Parameters
        ----------
        particles : ndarray
            
        limit : int, optional
            The default is 3.

        Returns
        -------
        particle_filtered : ndarray

        """
        clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(4, 4))
        particle_filtered = []
        for i, particle in enumerate(particles):
            slices_filtered = []

            for idx in range(0, particle.shape[0]):
                filtered_slice = clahe.apply(particle[idx]) / (255)
                slices_filtered.append(filtered_slice)

            particle_filtered.append(np.asarray(slices_filtered))
        particle_filtered = np.asarray(particle_filtered)

        return particle_filtered

    def add_axis(self, particles):
        """
        Expand 1D ndarray to 3D ndarray for classifier input

        Parameters
        ----------
        particles : ndarray
            

        Returns
        -------
        particle_xp : ndarray
            expanded ndarray

        """

        particle_xp = []
        for i, particle in enumerate(particles):
            particle_xp.append(np.expand_dims(np.asarray(particle), axis=3))
        particle_xp = np.asarray(particle_xp)

        return particle_xp

    def remove_empty_array(self, particles, masks):
        """
        Remove any empty slices/masks that might have been introduced as part 
        of the process.

        Parameters
        ----------
        particles : ndarray
            
        masks : ndarray
            

        Returns
        -------
        full_particles : ndarray
        
        full_masks : ndarray
            

        """
        full_particles = []
        full_masks = []
        for idx, mask in enumerate(masks):

            particle = particles[idx]
            if np.unique(mask).size == 1:

                continue
            elif np.unique(mask).size > 1:
                full_particles.append(particle)
                full_masks.append(mask)

        full_particles = np.asarray(full_particles, dtype="object")
        full_masks = np.asarray(full_masks, dtype="object")

        return full_particles, full_masks

    def isolate_masks(self, masks):
        """
        Remove partial masks from other particles included by counting the
        particle with the highest number of unique pixel values.

        Parameters
        ----------
        masks : ndarray

        Returns
        -------
        masks_isolated : ndarray
            

        """
        masks_isolated = []
        for i, mask in enumerate(masks):
            zero = 0
            values, counts = np.unique(mask, return_counts=True)
            values = np.delete(values, [0])
            counts = np.delete(counts, [0])

            particle_value = values[np.argmax(counts)]
            mask = mask == particle_value

            masks_isolated.append(mask)
        masks_isolated = np.asarray(masks_isolated)
        return masks_isolated

    def mask_particle(self, particles, masks):
        """
        Binary mask the grayscale particle using the CV segmentation.

        Parameters
        ----------
        particles : ndarray
        masks : ndarray

        Returns
        -------
        particles_masked : ndarray

        """
        masks_bool = []
        for i, j in enumerate(masks):
            mask_bool = np.asarray(j, dtype="bool")
            masks_bool.append(mask_bool)
        masks_bool = np.asarray(masks_bool)

        particles_masked = []
        for i, j in enumerate(particles):
            masked_particle = j * masks_bool[i]
            particles_masked.append(masked_particle)

        particles_masked = np.asarray(particles_masked)

        return particles_masked

    def slice_particles(self, particles):
        """
        Extract 10 equidistant slices from 80% of the depth of the particle
        for classification

        Parameters
        ----------
        particles : ndarray
            

        Returns
        -------
        particles_sliced : ndarray
            particle slices

        """
        particles_sliced = []

        for particle in particles:
            particle_slices = []

            middle_slice = int(particle.shape[0]) // 2
            increments = np.linspace(-0.4, 0.4, 10)

            for increment in increments:
                increment_slice = np.round(increment * int(particle.shape[0]))
                particle_slice = particle[int(middle_slice + increment_slice), :, :]

                particle_slices.append(np.asarray(particle_slice))

            particles_sliced.append(particle_slices)

        particles_sliced = np.asarray(particles_sliced)

        return particles_sliced

    def classify(self, model, sliced_particles, view=0):
        """
        Classifies the 10 previously extracted particles. If the mean particle
        score is >=0.4, then the particle is deemed as flawed. 

        Parameters
        ----------
        model : keras model
            Previously trained classifier model.
        sliced_particles : ndarray
            
        view : int, optional
            If set to 1, all the slices and their equivalent segmentations are 
            displayed. The default is 0.

        Returns
        -------
        particles_pd : pandas dataframe
            dataframe containing classification score for each particle
        cracked_idx : list
            index of cracked particles for successive segmentation/saving step
        results_list : list
            list of results

        """
        particle_score = []
        results_list = []
        classes = []
        for i in range(sliced_particles.shape[0]):

            results = model.predict(sliced_particles[i])
            results_list.append(np.round(results))
            if view == 1:
                print("Particle " + str(i))
                for j in range(sliced_particles[i].shape[0]):
                    print(results[j])
                    plt.imshow(sliced_particles[i, j, :, :, :], cmap="gray")
                    plt.show()

            particle_score.append(np.mean(results))
            classes.append("flawed" if particle_score[i] >= 0.4 else "pristine")
        dict_results = {"Particle score": particle_score, "Particle class": classes}
        particles_pd = pd.DataFrame(dict_results)
        cracked_idx = particles_pd[particles_pd["Particle class"] == "flawed"].index

        return particles_pd, cracked_idx, results_list

    def segment_cv2(self, particles, masks, index):
        """
        Segment each individual grayscale particle with OpenCV and overlay this segmentaiton with the CV label

        Parameters
        ----------
        particles : ndarray
        masks : ndarray
        index : list
            indexes of flawed particles returned from the classification step.
            
        Returns
        -------
        particles_segm : ndarray
        masks_export : ndarray

        """
        particles_segm = []
        particles = particles[index]
        masks = masks[index]
        masks_export = []
        particle = self.filter_clahe(particles, 2)
        for idx_particle, particle in enumerate(particles):
            particle_segm = []
            mask_export = []
            particle = particle[:, :]
            for idx_slice in range(0, particle.shape[0]):

                th, binary = cv2.threshold(
                    particle[idx_slice], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE
                )

                mask_morph = binary_dilation(
                    masks[idx_particle][idx_slice, :, :], iterations=1
                )
                mask_bool = np.asarray(mask_morph, dtype="bool")
                mask_export.append(binary)
                mask_new = np.where(mask_morph > 0, 2, 1)
                particle_masked = binary * mask_bool
                particle_masked_new = np.where(particle_masked >= 1, 2, 1)
                image_new = mask_new * particle_masked_new

                image_new = image_new.astype("uint8")
                particle_segm.append(image_new)
            particles_segm.append(np.asarray(particle_segm))
            masks_export.append(np.asarray(mask_export))

        masks_export = np.asarray(masks_export, dtype="object")
        particles_segm = np.asarray(particles_segm, dtype="object")

        return particles_segm, masks_export

    def save_particles(self, particles, segmentation, export_folder):
        """
    
        Save label fields to new folder
        Parameters
        ----------
        label_field : numpy array
            8bit numpy array containing label stack for export.
        label_folder : string
            Path to which label stack is saved to.
    
        """

        out_dir = export_folder

        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
            os.mkdir(out_dir)
        else:
            os.mkdir(out_dir)

        print("Saving individual particles in " + out_dir)

        n_particles = particles.shape[0]
        for j in range(0, n_particles):
            particle = particles[j]

            particle_folder = "particle_" + str(j)
            os.mkdir(os.path.join(out_dir, particle_folder))
            os.mkdir(os.path.join(out_dir, particle_folder, "grayscale"))
            os.mkdir(os.path.join(out_dir, particle_folder, "label"))

            for i in range(0, particle.shape[0]):

                filename_segm = "label_" + str(i + 1) + ".tiff"
                filename_gray = "gray_" + str(i + 1) + ".tiff"
                save_path_segm = os.path.join(out_dir, particle_folder, "label")
                save_path_gray = os.path.join(out_dir, particle_folder, "grayscale")

                cv2.imwrite(
                    os.path.join(save_path_segm, filename_segm), (segmentation[j][i])
                )
                cv2.imwrite(os.path.join(save_path_gray, filename_gray), particle[i])

        print("Complete!")

    def classify_particles(self):
        """
        Runs all functions sequentially and classifies the input dataset

        Returns
        -------
        None.

        """

        particles_numpy = self.dict_to_numpy(self.particles)
        masks_numpy = self.dict_to_numpy(self.masks)
        particles_removed = self.remove_outliers(particles_numpy)
        masks_removed = self.remove_outliers(masks_numpy)
        particles_full, masks_full = self.remove_empty_array(
            particles_removed, masks_removed
        )
        internal_particles, idx = self.remove_border(particles_full)
        internal_masks = self.remove_mask_border(masks_full, idx)
        masks_isolated = self.isolate_masks(internal_masks)
        masked_particles = self.mask_particle(internal_particles, masks_isolated)
        particles_resized = self.resize_slice(masked_particles)
        filtered_particles = self.filter_clahe(particles_resized)
        particles_xp = self.add_axis(filtered_particles)
        sliced_particles = self.slice_particles(particles_xp)
        particles_pd, cracked_idx, results_list = self.classify(
            self.model, sliced_particles, 0
        )
        print(particles_pd["Particle class"].value_counts())
        cracked_particles = internal_particles[cracked_idx]
        particles_segm, masks_export = self.segment_cv2(
            internal_particles, masks_isolated, cracked_idx
        )
        if not self.export_path == "":
            self.save_particles(cracked_particles, particles_segm, self.export_path)
