# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 11:45:37 2021

@author: Sohrab Daemi @ EIL
"""
############# Write docstrings for each function, re-import necessary modules etc

def view_GPU():
        import tensorflow as tf
        print(tf.__version__)
        from tensorflow.python.client import device_lib
        print(device_lib.list_local_devices())
        
def check_augmentation(dataset, augmentation):
        #check agumentation
        ## view agumentation
        image_id = random.choice(dataset.image_ids)
        original_image = dataset.load_image(image_id)
        original_mask, class_ids = dataset.load_mask(image_id)
        
        original_image_shape = original_image.shape
        original_mask_shape = original_mask.shape
        
        original_bbox = utils.extract_bboxes(original_mask)
        
        MASK_AUGMENTERS = ['Sequential','SomeOf','OneOf','Sometimes','Fliplr','Flipud','CropAndPad','Affine','PiecewiseAffine'
            ]
        
        def hook(images, augmenter, parents, default):
            return augmenter.__class__.__name__ in MASK_AUGMENTERS
        
        det = augmentation.to_deterministic()
        augmented_image = det.augment_image(original_image)
        augmented_mask = det.augment_image(original_mask.astype(np.uint8), hooks=imgaug.HooksImages(activator=hook))
        augmented_bbox = utils.extract_bboxes(augmented_mask)
        
        # Verify that shapes didn't change
        assert augmented_image.shape == original_image_shape, "Augmentation shouldn't change image size"
        assert augmented_mask.shape == original_mask_shape, "Augmentation shouldn't change mask size"
        # Change mask back to bool
        
        
        # Display image and instances before and after image augmentation
        visualize.display_instances(original_image, original_bbox, original_mask, class_ids,dataset.class_names)
        visualize.display_instances(augmented_image, augmented_bbox, augmented_mask, class_ids, dataset.class_names)

def last_log(logdir):
    subsets = ['train', 'validation']
    data = pd.DataFrame()
    for subset in subsets:
        path_folder = os.listdir(logdir)[-1]
        log_name = str(os.listdir(os.path.join(logdir, path_folder, subset))[0])
        path_log = os.path.join(logdir, path_folder, subset, log_name)
        data_subset = tf2pandas.tflog2pandas(path_log)
        if subset == 'validation':
            data_subset['subset']='val'
        else:
            data_subset['subset']='train'
        data = data.append(data_subset, ignore_index=True)
    g = sns.FacetGrid(data, col='metric', hue = 'subset', aspect = 1, height = 3, ylim = [0, 1.5])
    g.map(sns.lineplot, 'step', 'value', alpha = 0.8)
    g.add_legend()
    return []

def get_last_weights(logdir):
    model_folder = os.listdir(logdir)[-1]
    weights_name = os.listdir(os.path.join(logdir, model_folder))[-3]
    return os.path.join(logdir, model_folder, weights_name)