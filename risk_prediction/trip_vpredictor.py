# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 17:35:17 2019

@author: setsu
@filename: trip_vpredictor.py
@coding: utf-8
"""

from risk_prediction.trip_predictor import TripPredictor
import chainer
from chainer import cuda
import os
import glob
import cv2
import numpy as np
import math

class TripVPredictor(TripPredictor):
    """A class of TRIP(Traffic Risk Prediction) visualized predictor
    """
    def __init__(self, ds_path, spec_file_name, layer_name, box_type, window_size, model_param_file_path, plog_path, gpu_id):
        """ Constructor
            Args:
             ds_path (str): a dataset path
             spec_file_name (str): a dataset spec file name
             layer_name (str): a layer name
             box_type (str): a type of boxes - 'tbox' or 'ebox'
             model_param_file_path (str): a model parameter file path
             gpu_id (int): GPU ID (-1 for CPU) 
        """
        super(TripVPredictor, self).__init__(ds_path, spec_file_name, layer_name, box_type, window_size, model_param_file_path, plog_path, gpu_id)
        self.video_out = False
        self.video_out_path = None
        self.video_writer = None
    #
    def vpredict(self):
        """ Visualized prediction
        """
        # open a log file
        self.open_log_file()
        # prediction
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            running = True
            for i in range(self.ds_length):
                # get and prepare input feature sequence
                sample = [self.ds.get_example(i)]
                print(sample[0][0])
                if self.plogf is not None:
                    self.plogf.write(sample[0][0]+'\n')
                input_feature_seq = self.ds.prepare_input_sequence(sample, self.roi_bg) # <ADD self.roi_bg/>

                # get a list of original image file paths 
                orig_img_list = glob.glob(os.path.join(self.ds.ds_path, self.ds.dirs[i], 'orig_img')+'\*.*')

                for t in range(0, len(input_feature_seq), self.window_size):
                    # risk calculation
                    input_feature_win = input_feature_seq[t:t+self.window_size]
                    if self.risk_type == 'seq_risk':
                        r = self.model.predict_risk(input_feature_win)
                    elif self.risk_type == 'seq_mean_risk':
                        r = self.model.predict_mean_risk(input_feature_win)
                    elif self.risk_type == 'seq_max_risk': # find maximum risk value
                        r = self.model.predict_max_risk(input_feature_win)
                    # show original and ROI images and the risk (visualization)
                    for f in range(t, t+self.window_size):
                        # read an original image
                        orig_img = cv2.imread(orig_img_list[f])
                        # prepare ROI image
                        roi_img = self.extract_roi_image(orig_img, sample[0][1][f][1])
                        # put a risk on the ROI image
                        rval = cuda.to_cpu(r.data)[0][0]
                        fcolor = (0, 255, 255)
                        fsize = 2 + 3*rval
                        fthickness = 2 + int(8*rval)
                        height, width, _ = roi_img.shape[:3]
                        cv2.putText(roi_img, 'risk:'+str('{:.3f}'.format(rval)), (25, height-25), cv2.FONT_HERSHEY_SIMPLEX, fsize, fcolor, fthickness, cv2.LINE_AA)
                        img = cv2.hconcat([orig_img, roi_img])
                        # show
                        if width > 640:
                            img = cv2.resize(img, (2*640, round(640.0/width * height)))
                        cv2.imshow('TRIP Viewer', img)
                        # video out
                        if self.video_out:
                            self.output_video(img)
                        # wait and break control
                        if cv2.waitKey(5) & 0xFF == ord('q'):
                            running = False
                            break

                    # log
                    end = min(t+self.window_size, len(input_feature_seq))
                    print(' risk of interval [{0},{1}]: {2}'.format(t, end-1, r.data[0][0]))
                    if self.plogf is not None:
                        self.plogf.write(' risk of interval [{0},{1}]: {2}\n'.format(t, end-1, r.data[0][0]))
                        self.plogf.flush()
                    
                    # break control
                    if not running:
                        break
                # break control
                if not running:
                    break
        # close a log file
        self.close_log_file()
        # release video writer
        if self.video_writer is not None:
            self.video_writer.release()
        # destroy window
        cv2.waitKey()
        cv2.destroyAllWindows()
    #
    def extract_roi_image(self, img, box):
        """ Extract ROI image
            Args:
             img (numpy array):  array
             box (list): box information
            Returns:
             extracted image (numpy array): extracted open cv image array
        """
        # zero array with the same shape
        roi_img = np.zeros_like(img)
        # partial substitution
        height, width, _ = img.shape[:3]
        for b in box: # [label, center-x, center-y, width, height] 
            x0 = b[1] - b[3]/2
            y0 = b[2] - b[4]/2
            x1 = b[1] + b[3]/2
            y1 = b[2] + b[4]/2
            l_x0 = math.floor(width * x0)
            l_y0 = math.floor(height * y0)
            l_x1 = math.ceil(width * x1)
            l_y1 = math.ceil(height * y1)
            roi_img[l_y0:l_y1,l_x0:l_x1] = img[l_y0:l_y1,l_x0:l_x1]
        return roi_img
    #
    def set_video_out(self, video_out_path):
        """ Set vodeo output
            Args:
             video_out_path (str): a video out path or None
        """
        if video_out_path is not None:
            self.video_out = True
        else:
            self.video_out = False
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        self.video_out_path = video_out_path
    #
    def output_video(self, img):
        """ Write an image to a video out
            Args:
             img (numpy array): an image
        """
        if self.video_out:
            if self.video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'XVID') # only for AVI format
                self.video_writer = cv2.VideoWriter(self.video_out_path, fourcc, 30.0, (img.shape[1], img.shape[0])) # width, height
            self.video_writer.write(img)
#