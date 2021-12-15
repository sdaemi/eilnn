#
# Test utils.py
#

import eilnn
import unittest
import os

class TestUtils(unittest.TestCase):
    def test_load_grayscale(self):
        image_stack = eilnn.load_grayscale('test_gray_slices')
        assert len(image_stack.shape) == 4

    def test_load_grayscale_error(self):
        root = eilnn.IMAGES_DIR
        subset = 'test_gray_slices'
        error = 'error01'
        # Create a dummy folder that will produce the error
        fp = os.path.join(root, subset, error)
        os.mkdir(fp)
        with self.assertRaises(ValueError):
            self.test_load_grayscale()
        # Remove the folder
        os.rmdir(fp)
        
if __name__ == '__main__':
    unittest.main()