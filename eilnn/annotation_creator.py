# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 10:06:54 2021

@author: Sohrab Daemi - EIL

Usage: import the module to generate COCO style annotations from
grayscale images and their respective 8bit label fields.

The images are divided in validation and test subsets,
with folders in model-readable formats
"""

from PIL import Image
import numpy as np
from skimage import measure, morphology
from shapely.geometry import Polygon, MultiPolygon
import json
import os
import cv2
import shutil
from sklearn.utils import shuffle
from pathlib import Path


class ImportUtils:
    def __init__(self, root_dir):

        self.root_dir = root_dir
        # self.data_subset = data_subset

    def create_sub_masks(self, mask_image):

        """
        Creates sub masks from which annotations will be generated

        Parameters
        ----------
        mask_image : RGB image (PIL)
            individual label image to generate sub_masks from.

        Returns
        -------
        sub_masks : PIL image
            sub_mask image.

        """

        width, height = mask_image.size
        sub_masks = {}

        for x in range(width):
            for y in range(height):
                pixel = mask_image.getpixel((x, y))[:3]

                if pixel != (0, 0, 0):
                    pixel_str = str(pixel)
                    sub_mask = sub_masks.get(pixel_str)
                    if sub_mask is None:
                        sub_masks[pixel_str] = Image.new(
                            "1", (width + 2, height + 2))
                    sub_masks[pixel_str].putpixel((x + 1, y + 1), 1)
        return sub_masks

    def create_sub_mask_annotation(self, sub_mask, annotation_id):

        """
        Finds contours of each individual sub-mask and saves the
        values in a dictionary which is then merged in a .json file.

        Parameters
        ----------
        sub_mask : PIL image
            sub mask image with individual particle.

        annotation_id : int
            particle identifier (first particle 1, second particle 2 etc).

        Returns
        -------
        regions_model : dict
            dictionary containing particle annotations.
        area : int
            surface area of particle, used to filter artefacts.

        """

        sub_mask = np.asarray(sub_mask)
        sub_mask = np.multiply(sub_mask, 1)
        contours = measure.find_contours(sub_mask, 0.5,
            positive_orientation="high")

        segmentations = []
        polygons = []

        for contour in contours:
            for i in range(len(contour)):
                row, col = contour[i]
                contour[i] = (col - 1, row - 1)

            poly = Polygon(contour)
            poly = poly.simplify(1, preserve_topology=True)
            polygons.append(poly)
            segmentation = np.array(poly.exterior.coords).ravel().tolist()
            segmentations.append(segmentation)

        multi_poly = MultiPolygon(polygons)
        x, y, max_x, max_y = multi_poly.bounds
        area = multi_poly.area

        # Create individual particle annotaions here

        regions_model = {
            "{}".format(annotation_id): {
                "shape_attributes": {
                    "all_points_x": [x for x in segmentation[0::2]],
                    "all_points_y": [y for y in segmentation[1::2]],
                    "name": "polygon",
                },
                "region_attributes": {"name": "particle", "type": "particle"},
            }
        }
        return regions_model, area

    def train_validation_split(self, gray_list,
                mask_list, gray_filenames, val_split):

        """
        Shuffles and divides data into train and test subsets
        for annotation creation depending on val_split parameter.

        Parameters
        ----------
        gray_list : list
            List containing all grayscale images.
        mask_list : list
            List continaing all label images.
        gray_filenames : list
            List containing all grayscale filenames (used in json file and
            when copying images).
        val_split : float
            Percent split of validation data.

        Returns
        -------
        train_vars : list
            list containing shuffled and split lists of training images,
            training labels grayscale image filenames.
        val_vars : list
            list containing shuffled and split lists of validation images,
            training labels grayscale image filenames.

        """

        train_len = int(len(mask_list) * (1 - val_split))
        gray_list_shuff, gray_names_shuff, mask_list_shuff = shuffle(
            gray_list, gray_filenames, mask_list, random_state=0
        )
        gray_list_train = gray_list_shuff[0:train_len]
        gray_names_train = gray_names_shuff[0:train_len]
        mask_list_train = mask_list_shuff[0:train_len]

        gray_list_val = gray_list_shuff[train_len + 1:]
        gray_names_val = gray_names_shuff[train_len + 1:]
        mask_list_val = mask_list_shuff[train_len + 1:]

        train_vars = [gray_list_train, mask_list_train, gray_names_train]
        val_vars = [gray_list_val, mask_list_val, gray_names_val]

        return train_vars, val_vars

    def process_annotations(self, data, data_subset):

        """
        Processes train and validation datasets split and shuffled by
        the train_validation_split. Generates sub-mask annotations
        and merges and saves them into .json file

        Parameters
        ----------
        data : list
            List containing images.
        data_subset : str
            Data subset (train or val) for saving in correct folder.

        Returns
        -------
        None.

        """

        multi_regions = []

        # Loop through each individual image and
        # generate sub mask and annotations.

        for file_id, (gray_image, mask_image, gray_filename) \
                in enumerate(zip(*data)):
            try:

                mask_image_np = np.asarray(mask_image)
                annotation_id = 1
                image_id = 1
                particle_regions = []
                mask_image_min = np.min(mask_image_np)
                # Ensures that whatever the mask image format (8 or 16bit),
                # it will be converted to binary 8bit.

                mask_image_np = np.where(mask_image_np > mask_image_min, 1, 0)
                mask_image_np = morphology.binary_erosion(mask_image_np)
                mask_image_np = morphology.remove_small_holes(
                    mask_image_np, 15000)
                mask_image_np = measure.label(mask_image_np)
                mask_image_np = (mask_image_np * 255).astype(np.uint8)

                mask_image_rgb = Image.fromarray(mask_image_np).convert("RGB")
                sub_masks = self.create_sub_masks(mask_image_rgb)
                all_area = []

                for color, sub_mask in sub_masks.items():

                    model_annotations, area = self.create_sub_mask_annotation(
                        sub_mask, annotation_id
                    )

                    # Filter smaller artefacts as these crash the model

                    if area >= 500:
                        particle_regions.append(model_annotations)
                        annotation_id += 1
                        all_area.append(area)

                    elif area < 500:
                        continue

                print(
                    "Saving {} to /data/{} folder..".
                    format(gray_filename, data_subset)
                )

                image_id += 1
                model_regions_dict = {}

            # will skip any non image files (eg. *.info) that might
            except (TypeError):
                print("Skipped non image file in folder.")
                pass

            # Merge individual particle annotations into single .json

            for region, particle_region in enumerate(particle_regions):
                model_regions_dict.update(particle_region)
                multi_region = {
                    "filename": gray_filename,
                    "regions": model_regions_dict,
                }
                cv2.imwrite(
                    os.path.join(
                        self.out_dir, data_subset, gray_filename), gray_image
                )
                multi_regions.append(multi_region)
        # except:
        # continue

        # Save annotation
        print("Merging annotations..")
        self.json_annotations = {"image_data": multi_regions[:]}
        print("Saving json..")
        json_file_name = "annotations.json"
        export_path = os.path.join(self.out_dir, data_subset, json_file_name)
        with open(export_path, "w") as outfile:
            json.dump(self.json_annotations, outfile)

    def create_annotations(self, val_split=0.2, first_im=1, step=2):

        """
        Function imports grayscale and label fields for further processing,
        creates folders for export and splits data into test and train arrays
        before creating annotations.

        Parameters
        ----------
        val_split : float, optional
            Train/test validation split. The default is 0.2.
        first_im : int, optional
            First image of stack
            (can be adjusted to skip current collector, air etc).
            The default is 1.
        step : int, optional
            The default is 5.

        Returns
        -------
        None.

        """

        # Create folders for training and validation subsets

        self.out_dir = os.path.join(self.root_dir, "data/")
        if os.path.exists(self.out_dir):
            shutil.rmtree(self.out_dir)
            os.mkdir(self.out_dir)
        else:
            os.mkdir(self.out_dir)

        self.ann_dir = os.path.join(self.root_dir, "data_ann/")
        self.ann_dir = os.path.abspath(self.ann_dir)

        os.mkdir(os.path.join(self.out_dir, "train/"))
        os.mkdir(os.path.join(self.out_dir, "val/"))
        gray_dir = os.path.join(self.ann_dir, "grayscale/")
        print(gray_dir)
        masks_dir = os.path.join(self.ann_dir, "masks/")

        # Read grayscale images and labels
        # replace with eilnn functions?
        self.gray_list = [
            cv2.imread(os.path.join(gray_dir + i), 1)
            for i in os.listdir(gray_dir)
            if str("".join(filter(str.isdigit, i)))
        ][first_im::step]

        self.gray_filenames = [
            i for i in os.listdir(gray_dir)
            if str("".join(filter(str.isdigit, i)))
        ][first_im::step]
        self.mask_list = [
            cv2.imread(os.path.join(masks_dir + i), 0)
            for i in os.listdir(masks_dir)
            if str("".join(filter(str.isdigit, i)))
        ][first_im::step]

        print(np.asarray(self.gray_list).shape)
        print("Images Loaded..")

        train_vars, val_vars = self.train_validation_split(
            self.gray_list, self.mask_list, self.gray_filenames, val_split
        )
        data = [train_vars, val_vars]
        data_subset = ["train", "val"]

        # Generate annotaions

        for n, data in enumerate(data):
            self.process_annotations(data, data_subset[n])


if __name__ == "__main__":
    cwd = Path(os.getcwd())
    cwd_parent = cwd.parent.absolute()
    val_split = 0.2
    first_im = 1
    folder = "images/test_annotations"
    image_path = os.path.join(cwd_parent, folder)
    test = ImportUtils(image_path)
    test.create_annotations(val_split, first_im)
