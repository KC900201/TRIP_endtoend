# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 12:56:37 2019

@author: atsumilab
@filename: trip_combined.py
@coding: utf-8
========================
Date          Comment
========================
09142019      First revision
"""

#Import libraries
import os
import argparse

from risk_prediction.trip_trainer import TripTrainer
from estimation.dataset_generator.object_detector import ObjectDetector

#Main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dataset_maker')
    parser.add_argument('--object_detection_threshold', type=float, default=0.1)
    parser.add_argument('--gpu', type=int, default=0)
    
    dataset_file = input('Input dataset spec file (dataset_spec.txt): ')
    with open(dataset_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        if line[0] == '#':
            continue
        elif line.startswith('object_model_type'):
#            parser.add_argument('--object_model_type', choices=('yolo_v2', 'yolo_v3'), default='yolo_v3')
            parser.add_argument('--object_model_type', default=line.split(':')[1].strip().split())
        elif line.startswith('object_model_path'):
            parser.add_argument('--object_model_path', default=line.split(':')[1].strip().split())
        elif line.startswith('object_label_path'):
            parser.add_argument('--object_model_path', default=line.split(':')[1].strip().split())
        elif line.startswith('object_cfg_path'):
            parser.add_argument('--object_cfg_path', default=line.split(':')[1].strip().split())
        elif line.startswith('input_dir'):
            parser.add_argument('--input_dir', default='r' + line.split(':')[1].strip().split(), help='input directory')
        elif line.startswith('output_dir'):
            parser.add_argument('--output_dir', default='r'+ line.split(':')[1].strip().split(), help='directory where the dataset will be created')
        elif line.startswith('layer_name_list'):
            parser.add_argument('--layer_name_list', default='conv33,conv39,conv45', help='list of hidden layers name to extract features')
        elif line.startswith('save_img'):
            parser.add_argument('--save_img', type=bool, default=True, help='save_img option')
        elif line.startsiwth('video')
            parser.add_argument('--video', type=bool, default=True, help='video option')

    args = parser.parse_args()
    
    ##Continue -- 20191011
