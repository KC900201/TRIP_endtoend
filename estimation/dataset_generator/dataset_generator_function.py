# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 18:48:56 2019

@author: setsu
@filename: dataset_generator_function.py
@coding: utf-8
========================
Date          Comment
========================
10102019      First revision
10112019      Amend directory
"""

import os
import numpy as np
import cv2
from chainer import serializers, Variable
import chainer.functions as F
from estimation.dataset_generator.object_detector import ObjectDetector #10112019

class DatasetGenerator(object): #Create class (10102019)
    def save_images(orig_img, bboxes, output_dir, file):
        counter=0
        for bbox in bboxes:
            # 指定した物体が存在したら…/ If there is a specified object
            #if name in object_list:
            # 画像からその物体の領域を切り取って保存
            top = int(bbox[0])
            bottom = int(bbox[2])
            left = int(bbox[1])
            right = int(bbox[3])
            #left, top = result['box'].int_left_top()
            #right, bottom = result['box'].int_right_bottom()
            filename = os.path.join(output_dir,'img',file+'_'+str(counter)+'.jpg')
            cv2.imwrite(filename, orig_img[top:bottom, left:right])
            counter+=1
    
    def save_feature(features, layer_name_list, output_dir, file):
        # 指定したレイヤーの特徴を保存 / Save the specified layer's feature
        for feature, layer_name in zip(features, layer_name_list):
            save_name=os.path.join(output_dir, layer_name, file)
            np.savez(save_name, feature)
    
    def save_ebox(bboxes, labels, layer_ids, img_h, img_w, output_dir, file):
        with open(os.path.join(output_dir, 'ebox', file),'w') as w:
            for bbox, label, layer_id in zip(bboxes, labels, layer_ids):
                # 指定した物体が存在したら… / If there is a specified object
                #if name in object_list:
                # ラベルと相対座標をファイルに書き込む / Write the label and relative coordinates to a file
                width = bbox[3] - bbox[1]
                height = bbox[2] - bbox[0]
                center_x = bbox[1] + width/2
                center_y = bbox[0] + height/2
     
                wlist=[str(label),
                        str(center_x/img_w),
                        str(center_y/img_h),
                        str(width/img_w),
                        str(height/img_h),
                        str(layer_id)]
                w.write(' '.join(wlist)+'\n')
    
    def save_specfile(output_dir, features):
        filename = os.path.join(os.path.dirname(output_dir), 'ds_spec.txt')
        with open(filename, 'w') as w:
            w.write('feature: raw\nlayer: ')
            for i, layer_name in enumerate(img_features.keys()):
                lc,lh,lw=features[layer_name].data.shape[1:]
                wlist=[layer_name,layer_name,str(lh),str(lw),str(lc)]
                w.write(','.join(wlist))
                if(i!=len(img_features)-1): w.write('; ')
    
    def copy_file(outpath):
        copy_filelist = ['tbox']
        path = 'F:\AtsumiLabMDS-2\Trip\Dashcam\ds2' #defined path
    
        files = os.listdir(path)
        datasets = []
        for file in files:
            if file[0]!="X":
                datasets.append(file)
    
        for dataset in datasets:
            video_path = os.path.join(path, dataset)
            out_video_path = os.path.join(outpath, dataset)
            videos = os.listdir(video_path)
            for video in videos:
                feature_path = os.path.join(video_path, video)
                out_feature_path = os.path.join(out_video_path, video)
                if not video[-4:] =='.txt':
                    feature = os.listdir(feature_path)
                    print(os.path.join(feature_path, feature[8]), os.path.join(out_feature_path, feature[8]))
                    shutil.copytree(os.path.join(feature_path, feature[8]), os.path.join(out_feature_path, feature[8]))
