# -*- coding: utf-8 -*-
"""
Created on Mon March 09 2020

@author: setsu
@filename: trip_dataset_carla.py
@coding: utf-8
@description: New src code to implement TRIP prediction module on CARLA simulator. Basically, its a replica of original src code except
              removing hard-coded inputs to be able run on sync during CARLA simulator running
========================
Date          Comment
========================
03092020      First revision
"""

import os
import math
import numpy as np
from chainer import dataset
from functools import lru_cache # 01092020

class TripDatasetCarla(dataset.DatasetMixin):
    """ A class of TRIP(Traffic Risk Prediction) dataset for CARLA
    """
    @lru_cache()
    def __init__(self, ds_path, feature, layer, layer_name, box_type=None, sp=0):
        """Constructor
           Args:
            feature (str): feature type of dataset
            layer: layer type of quintiple
            layer_name (str): specific layer used from YOLOv3 for prediction box
            box_type (str): type of prediction box used - 'tbox' or 'ebox' - (used only for feature type 'raw')
        """
        layer_names = ''
        if(',' in layer_name):
            layer_names = str(layer_name).strip().split(',')
        else:
            layer_names = str(layer_name).strip()
        
        self.ds_path = ds_path
        self.feature_type = feature
        layers = layer.split(';')
        for layer in layers:
            quintuple = layer.strip().split(',')
            quintuple = [element.strip() for element in quintuple]
            if quintuple[0] in layer_names:
                layer_dir = quintuple[1]
                self.layer_info = (quintuple[0], int(quintuple[2]), int(quintuple[3]), int(quintuple[4])) # (layer_name, height, width, channels)
        
        if self.feature_type == 'raw':
            self.box_type = box_type
        elif self.feature_type == 'tbox_processed':
            self.box_type = 'tbox'
        elif self.feature_type == 'ebox_processed':
            self.box_type = 'ebox'
        # set dataset (feature_data, box_data)
        # feature data is a list of each feature file sequence(list)
        # box data is a list of each box file sequence(list)
        self.dirs = [d for d in os.listdir(self.ds_path) if os.path.isdir(os.path.join(self.ds_path, d))]
        self.feature_data = []
        self.box_data = []
        
        for d in self.dirs:
            # add in　all feature list from different layers - 20190210
            elist = []
            # Check if more than one layer is tested
            if(type(layer_names) == str):
                flist = [os.path.join(d, layer_names, f) for f in os.listdir(os.path.join(self.ds_path, d, layer_names))]
                if(sp > 0):
                    flist = flist[sp:len(flist)]
                elist.append(flist)
            else:
                for layer in layer_names:
                    flist = [os.path.join(d, layer, f) for f in os.listdir(os.path.join(self.ds_path, d, layer))] 
                    if(sp > 0):
                        flist = flist[sp:len(flist)]
                    elist.append(flist)
            self.feature_data.append(list(zip(*elist)))      
            blist = [os.path.join(d, self.box_type, f) for f in os.listdir(os.path.join(self.ds_path, d, self.box_type))]
            self.box_data.append(tuple(blist))

    @lru_cache()
    def get_example(self, i):
        """Get the i-th example
           Args:
            i (int): The index of the example
           Returns:
            a list of a tuple of feature array and box list
        """
        sample = []
 
        for j in range(len(self.feature_data[i])):            
            f_paths = list(self.feature_data[i][j])
            for p, paths in enumerate(f_paths):
                f_path = os.path.join(self.ds_path, paths)
                b_path = os.path.join(self.ds_path, self.box_data[i][j])
                f_array = np.load(f_path)['arr_0'] # f_array.shape = (1, channel, height, width)
                # resize the width and height according to layer 
                shape = list(f_array.shape)  
                f_array.shape = tuple(shape)
                if p == 0:
                    f_arrays = f_array
                else:
                    f_arrays = np.concatenate([f_arrays, f_array], axis=0)
                    
            with open(b_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            b_list = []
            for line in lines:
                elements = line.strip().split()
                b_list.append([elements[0], float(elements[1]), float(elements[2]), float(elements[3]), float(elements[4])]) # [label, center-x, center-y, width, height]
            sample.append((tuple(f_arrays), b_list))
        return self.dirs[i], sample
    #
    def __len__(self):
        """Get the length of a dataset
           Returns:
            len (int): length of the dataset (that is, the number of video clips)
        """
        return len(self.feature_data)
    #
    def get_layer_info(self):
        """Get layer information
           Returns:
            layer_info (tuple): a tuple of layer_name, height, width, channels
        """
        return self.layer_info
    #
    def get_feature_type(self):
        """Get feature type
           Returns:
            feature_type (str): feature type 'raw', 'tbox_processed' or 'ebox_processed'
        """
        return self.feature_type
    #
    def get_box_type(self):
        """Get box type
           Returns:
            box_type (str): box type 'tbox' or 'ebox'
        """
        return  self.box_type
    #
    def get_length(self):
        """Get the length of a dataset
           Returns:
            len (int): length of the dataset (that is, the number of video clips)
        """
        return self.__len__()
    #
    def prepare_input_sequence(self, batch, roi_bg=('BG_ZERO')): # <ADD roi_bg=('BG_ZERO')/>
        """ Prepare input sequence
            Args:
             batch (list of dataset samples): a list of samples of dataset
            Returns:
             feature batch (list of arrays): a list of feature arrays
        """
        # frame (ROI) feature sequence batch
        ffs_batch = [] # a minibatch size list of ffs(frame feature sequence)
        if self.feature_type == 'raw':
            for one_batch in batch:
                ffs = [element[0] for element in one_batch[1]] # a list of array[1 x channel x h x w], size=the length of sequence
                rbs = [element[1] for element in one_batch[1]]
                roi_ffs = []
                for i in range(len(ffs)):
                    roi_ffs.append(self.extract_roi_feature(ffs[i], rbs[i], roi_bg)) # <ADD roi_bg/>
                ffs_batch.append(roi_ffs)
        else:
            for one_batch in batch:
                ffs = [element[0] for element in one_batch]
                ffs_batch.append(ffs)
        # frame (ROI) feature batch sequence
        ffb_seq = [] # a sequence length list of frame feature 

        for i in range(len(ffs_batch[0])):
            ffb_seq.append(np.concatenate([alist[i] for alist in ffs_batch]))

        return ffb_seq
    #
    def extract_roi_feature(self, feature, box, roi_bg): # <ADD roi_bg/>
        """ Extract ROI feature
            Args:
             feature (numpy array): feature array  (1, channel, height, width)
             box (list): box information
            Returns:
             extracted feature (numpy array): extracted feature array
        """
        # <MOD>
        bg = roi_bg[0] # bg ::= BG_ZERO|BG_GN(Gaussian Noise)|BG_DP(Depression)
        if bg == 'BG_ZERO':
            # zero array with the same shape
            extracted_feature = np.zeros_like(feature)
        elif bg == 'BG_GN':
            gn_mean = 0.0
            gn_std = roi_bg[1]
            #extracted_feature = np.random.normal(gn_mean, gn_std, feature.shape)
            #convert feature into np array to use shape attr - 20190226
            np_feature = np.asarray(feature)
            extracted_feature = np.random.normal(gn_mean, gn_std, np_feature.shape)
        elif bg == 'BG_DP':
            depression = roi_bg[1]
            extracted_feature = feature * depression
        # </MOD>
        # partial substitution
        #l_name, l_height, l_width, l_channels = self.layer_info        
        l_height, l_width = extracted_feature.shape[2:]
        feat_np = np.asarray(feature)
        for b in box: # [label, center-x, center-y, width, height] 
            x0 = b[1] - b[3]/2
            y0 = b[2] - b[4]/2
            x1 = b[1] + b[3]/2
            y1 = b[2] + b[4]/2
            l_x0 = math.floor(l_width * x0)
            l_y0 = math.floor(l_height * y0)
            l_x1 = math.ceil(l_width * x1)
            l_y1 = math.ceil(l_height * y1)
            #extracted_feature[:,:,l_y0:l_y1,l_x0:l_x1] = feature[:,:,l_y0:l_y1,l_x0:l_x1]
            extracted_feature[:,:,l_y0:l_y1,l_x0:l_x1] = feat_np[:,:,l_y0:l_y1,l_x0:l_x1]
        return extracted_feature
    #
#