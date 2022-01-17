#
# Tests the ImportUtils class.
#
import eilnn
import unittest


class TestDataImportBenchmark(unittest.TestCase):
    
    def test_ImportUtils(self):
        root = eilnn.IMAGES_DIR
        val_split = 0.2
        first_im = 1
        folder = "test_data"
        image_folder = os.path.join(root, folder)
        test = eilnn.ImportUtils(image_folder)
        test.create_annotations(val_split, first_im)
    
    def test_json_exists(self):
        root = eilnn.IMAGES_DIR
        folder_train = "test_annotations/data/train"
        folder_val = "test_annotations/data/val"
        json_train = os.path.join(folder_train, 'annotations.json')
        json_val = os.path.join(folder_val, 'annotations.json')
        
        if json_train.is_file():
            print('Training annotations saved correctly.')
        else:
            print('Training annotations not saved.')
        
        if json_val.is_file():
            print('Validation annotations saved correctly.')
        else:
            print('Validation annotations not saved.')


if __name__ == "__main__":
    unittest.main()
