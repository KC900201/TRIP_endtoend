# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 17:30:17 2019

@author: setsu
@filename: main_trip_prediction.py
@coding: utf-8
"""

from trip_predictor import TripPredictor
import os

#
if __name__ == '__main__':
    prediction_spec_file = input('Input prediction spec file (prediction_spec.txt): ')
    with open(prediction_spec_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        if line[0] == '#':
            continue
        elif line.startswith('ds:'):
            ds_path, ds_spec_file_name = line.split(':')[1].strip().split()
            ds_path = os.path.join(os.path.dirname(prediction_spec_file), ds_path)
        elif line.startswith('layer_name:'):
            layer_name = line.split(':')[1].strip()
        elif line.startswith('box_type:'):
            box_type = line.split(':')[1].strip()
        elif line.startswith('window_size:'):
            window_size = int(line.split(':')[1].strip())
        elif line.startswith('model_param_file:'):
            model_param_file_path = line.split(':')[1].strip()
            model_param_file_path = os.path.join(os.path.dirname(prediction_spec_file), model_param_file_path)
        elif line.startswith('plog_path'):
            plog_path = line.split(':')[1].strip()
            plog_path = os.path.join(os.path.dirname(prediction_spec_file), plog_path)
        elif line.startswith('gpu_id:'):
            gpu_id = int(line.split(':')[1].strip())
    tripPredictor = TripPredictor(ds_path, ds_spec_file_name, layer_name, box_type, window_size, model_param_file_path, plog_path, gpu_id)
    tripPredictor.predict()