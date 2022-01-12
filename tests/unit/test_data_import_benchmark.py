#
# Tests the ImportUtils class.
#
import eilnn
import unittest


class TestDataImportBenchmark(unittest.TestCase):
    def test_ImportUtils(self):
        root = eilnn.ROOT_DIR
        val_split = 0.2
        first_im = 1
        folder = "test_data"
        image_folder = os.path.join(root, folder)
        test = eilnn.ImportUtils(image_folder)
        test.create_annotations(val_split, first_im)


if __name__ == "__main__":
    unittest.main()
