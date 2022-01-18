# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 12:26:34 2022

@author: Sohrab
"""

import os
import sys
import random
import math
import re
import time
import numpy as np
import skimage
import imgaug
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import shutil
import timeit
import eilnn

from samples import particles
from numba import cuda
import timeit


class Segment_3D:
    
    def __init__(self, model, image_path, filter_amt)
        self.model = model
        self.grayscale_data = eilnn.load_grayscale(image_path)
        self.filter_amt = filter_amt
        
        
    def segment_2d(self, model_inference, image_stack):
        label_field = np.zeros((image_stack.shape[1],image_stack.shape[2],image_stack.shape[0]),dtype=bool)
        #label_field = np.zeros((image_stack.shape[0],image_stack.shape[1],image_stack.shape[2]),dtype=bool)
        
        start = timeit.default_timer()
        for i in range(0, image_stack.shape[0]):
     
            r = model_inference.detect([image_stack[i,:,:,:]])
            r = r[0]
            if r['masks'].shape[2]==0:
                continue
            elif r['masks'].shape[2]>0:
                for j in range(0,r['masks'].shape[2]):
                    label_field[:,:,i] = label_field[:,:,i]+r['masks'][:,:,j]
                    
        stop=timeit.default_timer()
        print('Segmentation took',np.round((stop-start)/60,1),'minutes')
        return label_field

    #def resample_xz(self, image_stack):
    #    stack_resampled = np.swapaxes(np.swapaxes(image_stack,0,2),1,2)
    #    stack_resampled = np.flip(np.flip(stack_resampled,1),2)
    #    return stack_resampled

    def segment_xz(self, model_inference, image_stack):
        stack_resampled = np.swapaxes(np.swapaxes(image_stack,0,2),1,2)
        stack_resampled = np.flip(np.flip(stack_resampled,1),2)
        label_field = segment(model_inference, stack_resampled)
        label_field = np.flip(np.flip(label_field,0),1)
        label_field =np.moveaxis(label_field, 0,-1)
        return label_field

    #def resample_yz(self, image_stack):
    #    stack_resampled = np.swapaxes(image_stack,0,2)
    #    stack_resampled = np.swapaxes(stack_resampled,1,2)
    #    stack_resampled = np.flip(stack_resampled,1)
    #    return stack_resampled

    def segment_yz(self, model_inference, image_stack):
        
        stack_resampled = np.swapaxes(image_stack,0,2)
        stack_resampled = np.swapaxes(stack_resampled,1,2)
        stack_resampled = np.flip(stack_resampled,1)
        label_field = segment(model_inference, stack_resampled)
        label_field = np.moveaxis(label_field,0,-1)
        label_field = np.flip(label_field,-1)
        return label_field

    def union(self, label_field, label_xz, label_yz):
        label = label_field + label_xz + label_yz
        label[label>0]=1
        return label

    def segment_3d(self, model, data):
        label = segment(model, data)
        label_xz = segment_xz(model, data)
        label_yz = segment_yz(model, data)
        label_3d = union(label, label_xz, label_yz)
        return label_3d
        
    #class to visualise images from 3D segmentation in each direction
    def filtering(self, labels):
        
        label_filtered = binary_opening(labels)
        label_filtered = gaussian_filter(label_filtered, sigma=0.75)
        for i in range(1,3):
            label_filtered = binary_dilation(label_filtered)

        label_filtered = remove_small_holes(label_filtered)
        
        for i in range(1,3):
            label_filtered = binary_erosion(label_filtered)
            #label_filtered = binary_erosion(label_filtered)

        
        label_filtered = binary_opening(label_filtered)
        label_filtered = remove_small_holes(label_filtered)
        
        for i in range(1,6):
            label_filtered = binary_dilation(label_filtered)

        label_filtered = gaussian_filter(label_filtered, sigma=0.1)
        label_filtered = remove_small_holes(label_filtered)
        
        return label_filtered
        
    def visualize_filtering(self, grayscale_stack, label_3d, label_filtered):
        #define ix
        plt.subplot(1,2,1)
        plt.imshow(image_stack[ix,:,:], alpha = 1)
        plt.imshow(label_stack[ix,:,:], alpha = 0.4, cmap='Reds')
        
        plt.subplot(1,2,2)
        plt.imshow(image_stack[ix,:,:], alpha = 1)
        plt.imshow(label_filtered[ix,:,:], alpha = 0.4, cmap='Reds')
        plt.show() 

    def extendedmin(self, img, h):
        mask = img.copy() 
        marker = mask + h  
        hmin =  morph.reconstruction(marker, mask, method='erosion')
        return morph.local_minima(hmin)

    def watershed_sep(self):
        D = -edt(im)
        print('Done Distance Transform')
        mask = extendedmin(D, 2)
        print("Done Extended Minima")
        mlabel, N = morph.label(mask, return_num=True)

        # Watershed
        W = watershed(D, markers=mlabel, mask=im).astype(float)
        # make background white
        [W==0] = np.nan        

    def process_segm(self):
        
        labels = segment_3d(self.model, self.grayscale_data)
        labels_filtered = filtering(labels)
        visualize_filtering(self.grayscale_data, labels, labels_filtered)

        #watershed
        #cc3d
        bounding_boxes = stats['bounding_boxes']
        images = []
        increment =0
        particles = {}
        for particle, box in enumerate(bounding_boxes):
            x = box[0]
            y = box[1]
            z = box[2]
     
        image = image_stack[int(x.start)-increment:int(x.stop)+increment,int(y.start)-increment:int(y.stop)+increment,int(z.start):int(z.stop)]
        #print(image.shape)
        particles[particle] = image

        