# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 10:33:22 2022

@author: Sohrab
"""
import eilnn
import os


def segment_classify(
    model_segm, grayscale_dir, model_class, marker_size=8, export_path=""
):
    segment = eilnn.Segment_3D(model_segm, grayscale_dir, marker_size)
    particles, masks, watershed = segment.process_segm()

    classify = eilnn.classifier(model_class, particles, masks, export_path)
    classify.classify_particles()
    export_path = os.path.abspath(export_path)
    watershed_path = os.path.join(export_path, "watershed/")
    if not export_path == "":

        if os.path.exists(watershed_path):
            shutil.rmtree(watershed_path)
            os.mkdir(watershed_path)
        else:
            os.mkdir(watershed_path)

        eilnn.save_label(watershed, watershed_path)
    print("Complete!")


### function
