#
# Tests the ImportUtils class.
#
import eilnn
import os
import unittest


class TestAnnotationCreator(unittest.TestCase):
    def test_ImportUtils(self):
        root = eilnn.IMAGES_DIR
        val_split = 0.2
        first_im = 1
        folder = "tests/example_annotations/data_ann"
        image_folder = os.path.join(root, folder)
        test = eilnn.ImportUtils(image_folder)
        test.create_annotations(val_split, first_im)

    def test_json_exists(self):
        root = eilnn.IMAGES_DIR
        folder_train = "tests/example_annotations/data/train"
        folder_val = "tests/example_annotations/data/val"
        json_train = os.path.join(root, folder_train, "annotations.json")
        json_val = os.path.join(root, folder_val, "annotations.json")

        if os.path.exists(json_train):
            print("Training annotations saved correctly.")
        else:
            print("Training annotations not saved.")

        if os.path.exists(json_val):
            print("Validation annotations saved correctly.")
        else:
            print("Validation annotations not saved.")


if __name__ == "__main__":
    unittest.main()
    # print(eilnn.IMAGES_DIR)
