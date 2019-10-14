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
10102019      Coding for estimation part
10112019      Coding for risk prediction part, add in path for object detection threshold and gpu
10132019      Use pathlib to fix Unix and Windows path directory input
10142019      Input risk prediction input in beginning of main function
"""

#Import libraries
import os
import cv2
import argparse
import numpy as np
from pathlib import Path, WindowsPath #10132019

from risk_prediction.trip_trainer import TripTrainer
from estimation.dataset_generator.dataset_generator_function import DatasetGenerator
from estimation.dataset_generator.object_detector import ObjectDetector

#Main function
if __name__ == '__main__':
    # Initialize parameters
    parser = argparse.ArgumentParser(description='dataset_maker')
    
    spec_file = input('Input spec file (endtoend_spec.txt): ')
#    dataset_file = "C:/Users/Ng Kwong Cheong/OneDrive/Documents/Soka University/Research Plan/Source Code/TRIP/TRIP_endtoend/spec files/dataset_spec.txt"
#    print("Filepath: " + spec_file)
    with open(spec_file, "r", encoding='utf-8') as f: 
        lines = f.readlines()
    for line in lines:
        if line[0] == '#':
            continue
        # Dataset generator part
        elif line.startswith('object_model_type'):
#            parser.add_argument('--object_model_type', choices=('yolo_v2', 'yolo_v3'), default='yolo_v3')
            parser.add_argument('--object_model_type', default=line.split(':')[1].strip().split())
        elif line.startswith('object_model_path'):
            parser.add_argument('--object_model_path', default=line.split(':')[1].strip().split())
        elif line.startswith('object_label_path'):
            parser.add_argument('--object_label_path', default=line.split(':')[1].strip().split())
        elif line.startswith('object_cfg_path'):
            parser.add_argument('--object_cfg_path', default=line.split(':')[1].strip().split())
        elif line.startswith('object_detection_threshold'): #10112019
            parser.add_argument('--object_detection_threshold', type=float, default=float(line.split(':')[1]))
        elif line.startswith('gpu'): #10112019
            parser.add_argument('--gpu', type=int, default=int(line.split(':')[1]))
        elif line.startswith('input_dir'):
            parser.add_argument('--input_dir', default=line.split(':')[1], help='input directory')
        elif line.startswith('output_dir'):
            parser.add_argument('--output_dir', default=line.split(':')[1], help='directory where the dataset will be created')
        elif line.startswith('layer_name_list'):
            parser.add_argument('--layer_name_list', default=line.split(':')[1].strip().split(), help='list of hidden layers name to extract features')
        elif line.startswith('save_img'):
            parser.add_argument('--save_img', type=bool, default=bool(line.split(':')[1]), help='save_img option')
        elif line.startswith('video'):
            parser.add_argument('--video', type=bool, default=bool(line.split(':')[1]), help='video option')
        # End dataset generator part
        # Risk prediction part - 10142019
        elif line.startswith('train_ds1:'):
            train_ds_path1, train_spec_file_name1, train_risk1 = line.split(':')[1].strip().split()
            train_ds_path1 = os.path.join(os.path.dirname(spec_file), train_ds_path1)
            train_risk1 = int(train_risk1)
        elif line.startswith('train_ds2:'):
            train_ds_path2, train_spec_file_name2, train_risk2 = line.split(':')[1].strip().split()
            train_ds_path2 = os.path.join(os.path.dirname(spec_file), train_ds_path2)
            train_risk2 = int(train_risk2)
        elif line.startswith('test_ds1:'):
            test_ds_path1, test_spec_file_name1, test_risk1 = line.split(':')[1].strip().split()
            test_ds_path1 = os.path.join(os.path.dirname(spec_file), test_ds_path1)
            test_risk1 = int(test_risk1)
        elif line.startswith('test_ds2:'):
            test_ds_path2, test_spec_file_name2, test_risk2 = line.split(':')[1].strip().split()
            test_ds_path2 = os.path.join(os.path.dirname(spec_file), test_ds_path2)
            test_risk2 = int(test_risk2)
        elif line.startswith('layer_name:'):
            layer_name = line.split(':')[1].strip()
        elif line.startswith('box_type:'):
            box_type = line.split(':')[1].strip()
        elif line.startswith('execution_mode:'):
            execution_mode = line.split(':')[1].strip()
        elif line.startswith('num_of_epoch:'):
            num_of_epoch = int(line.split(':')[1].strip())
        elif line.startswith('minibatch_size:'):
            minibatch_size = int(line.split(':')[1].strip())
        elif line.startswith('eval_interval:'):
            eval_interval = int(line.split(':')[1].strip())
        elif line.startswith('save_interval:'):
            save_interval = int(line.split(':')[1].strip())
        elif line.startswith('model_param_file:'):
            model_param_file_path = line.split(':')[1].strip()
            model_param_file_path = os.path.join(os.path.dirname(spec_file), model_param_file_path)
        elif line.startswith('tlog_path:'):
            tlog_path = line.split(':')[1].strip()
            tlog_path = os.path.join(os.path.dirname(spec_file), tlog_path)
        elif line.startswith('gpu_id:'):
            gpu_id = int(line.split(':')[1].strip())
        # End risk prediction part
            
    ## 10102019
    args = parser.parse_args()
    
    input_dir = args.input_dir
    layer_name_list = args.layer_name_list.split(',')
    output_dir = args.output_dir
    thresh = args.object_detection_threshold
    save_img = args.save_img
    video = args.video
    files = 0
    folders = 0
    
    for _, dirnames, filenames in os.walk(input_dir):
        files += len(filenames)
        folders += len(dirnames)
    
    print('' + str(files) + 'files, ' + str(folders)+ 'folders found')
    
    predictor = ObjectDetector(args.object_model_type, args.object_model_path, 
                               args.object_label_path, args.object_cfg_path, args.object_detection_threshold,
                               device=args.gpu)
    lable_names = np.array(predictor.get_bbox_labels())
    
    # When folder contains video
    if video:
        orig_input_dir = input_dir
        orig_output_dir = output_dir
        # フォルダ内のビデオ数だけループ / Loop only the number of videos in a folder 
        video_files = os.listdir(input_dir)
        for video_file in video_files:
            print('save %s feature...' % video_file)
            input_dir = orig_input_dir + '/' + video_file
            output_dir = orig_output_dir + '/' + video_file[:-4]
            # フォルダが無ければ新規作成 / Create a new folder if there is none previously
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            for layer in layer_name_list:
                if not os.path.isdir(os.path.join(output_dir, layer)):
                    os.mkdir(os.path.join(output_dir, layer))
            if save_img and not os.path.isdir(os.path.join(output_dir, 'img')):
                os.mkdir(os.path.join(output_dir, 'img'))
            if video and not os.path.isdir(os.path.join(output_dir, 'orig_img')):
                os.mkdir(os.path.join(output_dir, 'orig_img'))
            if not os.path.isdir(os.path.join(output_dir, 'ebox')):
                os.mkdir(os.path.join(output_dir, 'ebox'))

            # 動画から画像ファイルのリストを作成 / Create a list of image files from a video 
            print('load video...')
            i=1
            cap = cv2.VideoCapture(input_dir)

            while(cap.isOpened()):
                flag, frame = cap.read()
                if flag == False:
                    break
                cv2.imwrite(output_dir + '/orig_img/' + str(i).zfill(6) + '.jpg', frame)
                i+=1
            cap.release()

            file_list = os.listdir(output_dir + '/orig_img')
            img_files = [f for f in file_list if os.path.isfile(os.path.join(output_dir + '/orig_img', f))]
#
            # 最初の画像を読み込み / Loading the first image
            orig_img = cv2.imread(os.path.join(output_dir, img_files[0]))
            # 基準となる画像の高さと幅を取得 / Get the height and width of the base image
            #img_h, img_w = orig_img.shape[:2]
            img_h = 720
            img_w = 1280
#
            # ファイル数分繰り返す / Repeat files for a few minutes 
            for img_file in img_files:
                # 拡張子とファイル名を分ける / Separate file names from extensions
                file, ext = os.path.splitext(img_file)
                # 画像読み込み / Image loading
                orig_img = cv2.imread(os.path.join(output_dir + '/orig_img', img_file))
#
                # サイズが異なる場合は変形してから入力 / If the size is different, deform and then enter
                if (img_h, img_w) != orig_img.shape[:2] :
                    orig_img = cv2.resize(orig_img, (img_w, img_h))
#
                # 画像の高さと幅を取得 / Get the height and width of the image 
                img_h, img_w = orig_img.shape[:2]
                # 検出結果と特徴を取得 / Get detection results and features
                #results, img_features = predictor(orig_img, thresh, layer_list)
                bboxes, labels, scores, layer_ids, features = predictor(orig_img)
                
                # 検出結果を利用し、画像を切り取って保存 / Use the detection results to cut and save images
                #if save_img: save_images(orig_img, results, object_list, output_dir, file)
                if save_img: DatasetGenerator.save_images(orig_img, bboxes, output_dir, file)
                # 特徴ファイルを保存 / Save feature file
                #save_feature(img_features, output_dir, file+'.npz')
                DatasetGenerator.save_feature(features, layer_name_list, output_dir, file+'.npz')
                # 指定した物体の座標を保存 / Save the coordinates of the specified object
                #save_ebox(results, object_list, img_h, img_w, output_dir, 'e'+file+'.txt')
                DatasetGenerator.save_ebox(bboxes, labels, layer_ids, img_h, img_w, output_dir, 'e'+file+'.txt')
            # specfileを保存 / save specfile
            #save_specfile(output_dir, img_features)
    else:
        orig_input_dir = input_dir
        orig_output_dir = output_dir
        # フォルダ内のビデオ数だけループ / Loop only the number of videos in a folder
        video_files = os.listdir(input_dir)
        for video_file in video_files:
            if video_file[-5:-2]>='0':
                print(video_file)
                print('save %s feature...' % video_file)
                input_dir = orig_input_dir + '/' + video_file + '/orig_img'
                output_dir = orig_output_dir + '/' + video_file
                # フォルダが無ければ新規作成 / Create a new folder if there is none previously
                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir)
                for layer in layer_list:
                    if not os.path.isdir(os.path.join(output_dir, layer)):
                        os.mkdir(os.path.join(output_dir, layer))
                if save_img and not os.path.isdir(os.path.join(output_dir, 'img')):
                    os.mkdir(os.path.join(output_dir, 'img'))
                if not os.path.isdir(os.path.join(output_dir, 'ebox')):
                    os.mkdir(os.path.join(output_dir, 'ebox'))

                # 画像ファイルのリストを作成 / Create a list of image files
                print('load image...')
                file_list = os.listdir(input_dir)
                img_files = [f for f in file_list if os.path.isfile(os.path.join(input_dir, f))]
#
                # 最初の画像を読み込み / Loading the first image
                orig_img = cv2.imread(os.path.join(input_dir, img_files[0]))
                # 基準となる画像の高さと幅を取得
                img_h, img_w = orig_img.shape[:2]
#
                # ファイル数分繰り返す / Repeat files for a few minutes 
                for img_file in img_files:
                    # 拡張子とファイル名を分ける / Separate file names from extensions
                    file, ext = os.path.splitext(img_file)
                    # 画像読み込み / load image
                    orig_img = cv2.imread(os.path.join(input_dir, img_file))
#
                    # サイズが異なる場合は変形してから入力 / If the size is different, deform and then enter
                    if (img_h, img_w) != orig_img.shape[:2] :
                        orig_img = cv2.resize(orig_img, (img_w, img_h))
#
                    # 検出結果と特徴を取得 / retrieve detection results and features
                    results, img_features = predictor(orig_img, thresh, layer_list)

                    # 検出結果を利用し、画像を切り取って保存 / Use the detection results to cut and save images
                    if save_img: DatasetGenerator.save_images(orig_img, results, object_list, output_dir, file)
                    # 特徴ファイルを保存 / save feature file
                    DatasetGenerator.save_feature(img_features, output_dir, file+'.npz')
                    # 指定した物体の座標を保存 / save specified object's coordinates
                    DatasetGenerator.save_ebox(results, object_list, img_h, img_w, output_dir, 'e'+file+'.txt')
                # specfileを保存 save specfile
                #save_specfile(output_dir, img_features)        
    ## End estimation part -- 10102019
    """
    ## 10112019
    training_spec_file = input('Input training spec file (training_spec.txt): ')
    with open(training_spec_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line  in lines:
        if line[0] == '#':
            continue
        elif line.startswith('train_ds1'):
            train_ds_path1, train_spec_file_name1, train_risk1 = line.split(':')[1].strip().split()
            train_ds_path1 = os.path.join(os.path.dirname(training_spec_file), train_ds_path1)
            train_risk1 = int(train_risk1)
        elif line.startswith('train_ds2:'):
            train_ds_path2, train_spec_file_name2, train_risk2 = line.split(':')[1].strip().split()
            train_ds_path2 = os.path.join(os.path.dirname(training_spec_file), train_ds_path2)
            train_risk2 = int(train_risk2)
        elif line.startswith('test_ds1:'):
            test_ds_path1, test_spec_file_name1, test_risk1 = line.split(':')[1].strip().split()
            test_ds_path1 = os.path.join(os.path.dirname(training_spec_file), test_ds_path1)
            test_risk1 = int(test_risk1)
        elif line.startswith('test_ds2:'):
            test_ds_path2, test_spec_file_name2, test_risk2 = line.split(':')[1].strip().split()
            test_ds_path2 = os.path.join(os.path.dirname(training_spec_file), test_ds_path2)
            test_risk2 = int(test_risk2)
        elif line.startswith('layer_name:'):
            layer_name = line.split(':')[1].strip()
        elif line.startswith('box_type:'):
            box_type = line.split(':')[1].strip()
        elif line.startswith('execution_mode:'):
            execution_mode = line.split(':')[1].strip()
        elif line.startswith('num_of_epoch:'):
            num_of_epoch = int(line.split(':')[1].strip())
        elif line.startswith('minibatch_size:'):
            minibatch_size = int(line.split(':')[1].strip())
        elif line.startswith('eval_interval:'):
            eval_interval = int(line.split(':')[1].strip())
        elif line.startswith('save_interval:'):
            save_interval = int(line.split(':')[1].strip())
        elif line.startswith('model_param_file:'):
            model_param_file_path = line.split(':')[1].strip()
            model_param_file_path = os.path.join(os.path.dirname(training_spec_file), model_param_file_path)
        elif line.startswith('tlog_path:'):
            tlog_path = line.split(':')[1].strip()
            tlog_path = os.path.join(os.path.dirname(training_spec_file), tlog_path)
        elif line.startswith('gpu_id:'):
            gpu_id = int(line.split(':')[1].strip())
    tripTrainer = TripTrainer(train_ds_path1, train_spec_file_name1, train_risk1,
                              train_ds_path2, train_spec_file_name2, train_risk2,
                              test_ds_path1, test_spec_file_name1, test_risk1,
                              test_ds_path2, test_spec_file_name2, test_risk2,
                              layer_name, box_type, 
                              execution_mode, num_of_epoch, minibatch_size, eval_interval, save_interval,
                              model_param_file_path, tlog_path, gpu_id)
    if execution_mode == 'train' or execution_mode == 'retrain':
        tripTrainer.learn_model()
    else:
        tripTrainer.test_model()
    ##10112019
    """
        
        

    
    