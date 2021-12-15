#
# Test utils.py
#

import eilnn
import unittest

class TestUtils(unittest.TestCase):
    def test_load_grayscale(self):
        image_stack = eilnn.load_grayscale('test_gray_slices')
        
        
if __name__ == '__main__':
    unittest.main()