# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 10:33:22 2022

@author: Sohrab
"""
import eilnn




def segment_classify(model_segm, grayscale_dir, model_class, marker_size=8, export_path = ''):
        segment = eilnn.Segment_3D(model_segm, grayscale_dir, marker_size)
        particles, masks, watershed = segment.process_segm()

        classify = eilnn.classifier(model_class, particles, masks, export_path)
        classify.classify_particles()
        
### function