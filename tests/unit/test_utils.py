#
# Test utils.py
#

import eilnn
import unittest
import os
import numpy as np
import shutil


class TestUtils(unittest.TestCase):
    def test_load_grayscale(self):

        root = eilnn.IMAGES_DIR
        subset = "tests/test_gray_slices"
        fp = os.path.join(root, subset)
        image_stack = eilnn.load_grayscale(fp)

        assert len(image_stack.shape) == 4

    def test_load_grayscale_error(self):
        root = eilnn.IMAGES_DIR
        subset = "tests/test_gray_slices"
        error = "error01"
        # Create a dummy folder that will produce the error
        fp = os.path.join(root, subset, error)
        os.mkdir(fp)
        with self.assertRaises(ValueError):
            self.test_load_grayscale()
        # Remove the folder
        os.rmdir(fp)

    def test_load_label(self):
        root = eilnn.IMAGES_DIR
        subset = "tests/test_annotations"
        fp = os.path.join(root, subset)
        label_stack = eilnn.load_label(fp)
        assert len(label_stack.shape) == 3

    def test_load_label_error(self):
        root = eilnn.IMAGES_DIR
        subset = "tests/test_annotations"
        error = "error02"
        # Create a dummy folder that will produce the error
        fp = os.path.join(root, subset, error)
        os.mkdir(fp)
        with self.assertRaises(ValueError):
            self.test_load_label()
        # Remove the folder
        os.rmdir(fp)

    def test_save_label(self):
        root = eilnn.IMAGES_DIR
        subset = "tests/test_annotations"
        fp = os.path.join(root, subset)
        fp_export = os.path.join(root, "test_save")

        label_stack = eilnn.load_label(fp)
        eilnn.save_label(label_stack, fp_export)

    def test_save_label_error(self):
        # try save a 2D array
        error_data = np.empty([2, 2])
        root = eilnn.IMAGES_DIR
        subset = "tests/test_label"
        fp = os.path.join(root, subset)
        os.mkdir(fp)

        with self.assertRaises(ValueError):
            eilnn.save_label(error_data, fp)
        os.rmdir(fp)
        
    def test_pickles(self):
        test_data = np.empty([2, 2])
        root = eilnn.IMAGES_DIR
        dummy_folder = "tests/dummy_pickles"
        folder_path = os.path.join(root, dummy_folder)
        os.mkdir(folder_path)
        dummy_pickle = os.path.join(folder_path, 'dummy.txt')
        eilnn.save_pickles(test_data,dummy_pickle)
        data = eilnn.load_pickles(dummy_pickle, 1)
        shutil.rmtree(folder_path)
        

if __name__ == "__main__":
    unittest.main()
