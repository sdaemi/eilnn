# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 10:06:54 2021

@author: Sohrab  Randjbar Daemi - EIL

"""
#write docs for functions etc
from PIL import Image
import numpy as np
from skimage import measure, morphology
from shapely.geometry import Polygon, MultiPolygon
import json
import os
import itertools
import cv2
import matplotlib.pyplot as plt
import shutil
from sklearn.utils import shuffle
import pandas as pd
import random
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import imgaug
from eilnn import tflog2pandas as tf2pandas
import seaborn as sns



class ImportUtils:
    
    def __init__(self, root_dir, data_subset):
        
        self.root_dir = root_dir
        self.data_subset = data_subset

        
    def create_sub_masks(self, mask_image):
        width, height = mask_image.size
        sub_masks = {}
             
        for x in range(width):
            for y in range(height):
                pixel = mask_image.getpixel((x,y))[:3]
                
                if pixel !=(0,0,0):
                    pixel_str = str(pixel)
                    sub_mask = sub_masks.get(pixel_str)
                    if sub_mask is None:
                        sub_masks[pixel_str] = Image.new('1', (width+2,height+2))
                    sub_masks[pixel_str].putpixel((x+1,y+1),1)
        
        return sub_masks
    
    
    def create_sub_mask_annotation(self, sub_mask, image_id, annotation_id):
        # try:
         
         sub_mask = np.asarray(sub_mask)
         sub_mask = np.multiply(sub_mask,1)
         
         contours = measure.find_contours(sub_mask,0.5,positive_orientation='high')
         
         segmentations = []
         polygons = []
         for contour in contours:
             for i in range(len(contour)):
                 row, col = contour[i]
                 contour[i] = (col-1, row-1)
                 
             poly = Polygon(contour)
             poly = poly.simplify(1, preserve_topology=True)
             polygons.append(poly)
             segmentation = np.array(poly.exterior.coords).ravel().tolist()
             segmentations.append(segmentation)

             
         multi_poly = MultiPolygon(polygons)
         x, y, max_x, max_y = multi_poly.bounds
         width = max_x - x
         height = max_y - y
         bbox = (x, y, width, height)
         area = multi_poly.area         
         
         regions_model = {
             "{}".format(annotation_id):{
                 
                 "shape_attributes":{
                     "all_points_x":[x for x in segmentation[0::2]],
                     "all_points_y":[y for y in segmentation[1::2]],
                     "name": "polygon"}, 
                 "region_attributes":
                     {"name":"particle", "type":"uncracked"}}
         }

         return regions_model, area
        
        # except ValueError:
        #     pass
      
        
    def train_validation_split(self,gray_list, mask_list,gray_filenames, val_split):
        
        train_len = int(len(mask_list)*(1-val_split))
        gray_list_shuff, gray_names_shuff, mask_list_shuff = shuffle(gray_list, gray_filenames, mask_list,
                                                                    random_state = 0)
        gray_list_train = gray_list_shuff[0:train_len]
        gray_names_train = gray_names_shuff[0:train_len]
        mask_list_train = mask_list_shuff[0:train_len]
        
        gray_list_val = gray_list_shuff[train_len+1:]
        gray_names_val = gray_names_shuff[train_len+1:]
        mask_list_val = mask_list_shuff[train_len+1:]
        
        train_vars = [gray_list_train, mask_list_train, gray_names_train]
        val_vars = [gray_list_val,mask_list_val, gray_names_val]
        
        return train_vars, val_vars
    
    def process_annotations(self, data, data_subset):
        model_json_export = {}
        multi_regions = []
        for  file_id, (gray_image, mask_image, gray_filename) in \
            enumerate(zip(*data)):
                #try:
        
                    #print(gray_filename)
                 mask_image_np = np.asarray(mask_image)
                 mask_image_max = (mask_image_np).max()
                 mask_image_min = (mask_image_np).min()
                 #print(np.unique(mask_image_np))
                 #mask_image_np = (mask_image_np*255).astype(np.uint8)
                 annotation_id = 1
                 image_id = 1
                 annotations = []
                 particle_regions = []
                 mask_image_np = np.where(mask_image_np<mask_image_max, 1,0)
       
                 mask_image_np = morphology.binary_erosion(mask_image_np)
     
                 mask_image_np = morphology.remove_small_holes(mask_image_np,15000)
                 
                 mask_image_np = measure.label(mask_image_np)
                 plt.imshow(mask_image_np)
                 mask_image_np = (mask_image_np*255).astype(np.uint8)
                 
                 mask_image_rgb = Image.fromarray(mask_image_np).convert('RGB')
                 sub_masks = self.create_sub_masks(mask_image_rgb)
                 all_area = []
 
                 for color, sub_mask in sub_masks.items():
                     #try:
                     
                     model_annotations, area = self.create_sub_mask_annotation(sub_mask,image_id,annotation_id)
                     if area>=500:
                         particle_regions.append(model_annotations)
                         annotation_id += 1
                         all_area.append(area)
                     elif area<500:
                         continue
 
                     
                 all_area.sort()    
                # print(all_area)
                 image_id +=1
                 model_regions_dict={}
                 
                 print('Saving {} to /data/{} folder..'.format(gray_filename, data_subset))
                 for region,particle_region in enumerate(particle_regions):
                     model_regions_dict.update(particle_region)
                     multi_region = {"filename":gray_filename, "regions":model_regions_dict}
                     cv2.imwrite(os.path.join(self.out_dir,data_subset,gray_filename),gray_image)
                     multi_regions.append(multi_region)
                #except:
                    #continue
                
        print('Merging annotations..')
        self.json_annotations = {"image_data":multi_regions[:]}
        print('Saving json..')
        json_file_name ='annotations.json'
        export_path = os.path.join(self.out_dir, data_subset, json_file_name)
        with open(export_path, 'w') as outfile:
            json.dump(self.json_annotations, outfile)
    
    
    def create_annotations(self, val_split = 0.2, first_im = 1, step = 2):
     
        for subset in self.data_subset:
            print('Processing {}..'.format(subset))
            self.out_dir = os.path.join(self.root_dir,subset,'data/')
            if os.path.exists(self.out_dir):
                shutil.rmtree(self.out_dir)
                os.mkdir(self.out_dir)
            else:
                os.mkdir(self.out_dir)
                
            self.ann_dir = os.path.join(self.root_dir, subset, 'data_ann/')
            self.ann_dir = os.path.abspath(self.ann_dir)

            os.mkdir(os.path.join(self.out_dir,'train/'))
            os.mkdir(os.path.join(self.out_dir,'val/'))
            gray_dir = os.path.join(self.ann_dir, 'grayscale/')
            print(gray_dir)
            masks_dir = os.path.join(self.ann_dir,'masks/')
            
            ##exclude first 10 slices here
            self.gray_list = [cv2.imread(os.path.join(gray_dir+i),1) \
                              for i in os.listdir(gray_dir) if str("".join(filter(str.isdigit,i)))][first_im::step]
            self.gray_filenames =[i \
                              for i in os.listdir(gray_dir) if str("".join(filter(str.isdigit,i)))][first_im::step]    
            self.mask_list = [cv2.imread(os.path.join(masks_dir+i),0) \
                              for i in os.listdir(masks_dir) if str("".join(filter(str.isdigit,i)))][first_im::step]
            
            print(np.asarray(self.gray_list).shape)
            print('Images Loaded..')

            
            train_vars, val_vars = self.train_validation_split(self.gray_list, self.mask_list, self.gray_filenames, val_split)
            #print(train_vars[2],val_vars[2])
            data = [train_vars, val_vars]
            data_subset = ['train','val']
            
            for n, data in enumerate(data):
                self.process_annotations(data, data_subset[n])
        
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

if __name__ == "__main__":

    val_split = 0.2
    first_im = 1
    folder = ['subset_4']
    test = ImportUtils('C:/Users/Sohrab/Documents/crack/EILNet_tf2/images/', folder)
    test.create_annotations(val_split, first_im) 