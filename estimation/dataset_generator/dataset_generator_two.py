# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 13:05:58 2019

@author: atsumilab
@code: utf-8
"""

import os
import numpy as np
import cv2
from chainer import serializers, Variable
import chainer.functions as F
#from yolov2_darknet_predict import Predictor
from object_detector import ObjectDetector
import argparse

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dataset_maker')
    parser.add_argument('--object_model_type', choices=('yolo_v2', 'yolo_v3'), default='yolo_v3')
    parser.add_argument('--object_model_path', default='../model_v3/accident_KitDash_8000.npz')
    parser.add_argument('--object_label_path', default='../model_v3/obj.names') # must be specified other than 'coco' and 'voc'    
    parser.add_argument('--object_cfg_path', default='../model_v3/yolo-obj.cfg')
    parser.add_argument('--object_detection_threshold', type=float, default=0.1)
    parser.add_argument('--gpu', type=int, default=0)

    #parser.add_argument('--input_dir', default=r'E:\AtsumiLabMDS-2\TRIP\Dataset\DashcamAccidentDataset\videos\training\positive', help='input directory')
    #parser.add_argument('--output_dir', default=r'E:\AtsumiLabMDS-2\TRIP\Trip2018Q1\dataset_generator3.0\dataset_generator\test', help='directory where the dataset will be created')
    #change directory to E:\AtsumiLabMDS-2\TRIP\Trip2018Q1\Dashcam\ds3
    parser.add_argument('--input_dir', default=r'E:\AtsumiLabMDS-2\TRIP\Trip2018Q1\Dashcam\ds3', help='input directory')
    parser.add_argument('--output_dir', default=r'E:\AtsumiLabMDS-2\TRIP\Trip2018Q1\Dashcam\ds3', help='directory where the dataset will be created')

    parser.add_argument('--layer_name_list', default='conv33,conv39,conv45', help='list of hidden layers name to extract features')
    #parser.add_argument('--object_list', default='car,truck,person,tram,bicycle,motorbike,bus', help='list of object to get box coords')
    parser.add_argument('--save_img', type=bool, default=True, help='save_img option')
    parser.add_argument('--video', type=bool, default=True, help='video option')
    args = parser.parse_args()

    input_dir = args.input_dir
    layer_name_list = args.layer_name_list.split(',')
    #object_list = args.object_list.split(',')
    output_dir = args.output_dir
    thresh = args.object_detection_threshold
    save_img = args.save_img
    video = args.video
    files = 0 
    folders =  0

    for _, dirnames, filenames in os.walk(input_dir):
        files += len(filenames)
        folders += len(dirnames)
    print('' + str(files) + 'files, ' + str(folders)+ 'folders found')
    #Continue here, proceed to delete old folders and unwanted files (20190127)

    #predictor = Predictor(args.cfg_file, args.model_file, args.label_file)
    predictor = ObjectDetector(args.object_model_type, args.object_model_path, 
                                     args.object_label_path, args.object_cfg_path, args.object_detection_threshold,
                                     device=args.gpu)
    lable_names = np.array(predictor.get_bbox_labels())
    # フォルダ内が動画の時 / When folder contains a video
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
                if save_img: save_images(orig_img, bboxes, output_dir, file)
                # 特徴ファイルを保存 / Save feature file
                #save_feature(img_features, output_dir, file+'.npz')
                save_feature(features, layer_name_list, output_dir, file+'.npz')
                # 指定した物体の座標を保存 / Save the coordinates of the specified object
                #save_ebox(results, object_list, img_h, img_w, output_dir, 'e'+file+'.txt')
                save_ebox(bboxes, labels, layer_ids, img_h, img_w, output_dir, 'e'+file+'.txt')
            # specfileを保存 / save specfile
            #save_specfile(output_dir, img_features)



    # フォルダ内が画像のとき / When the folder contains an image
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
                    if save_img: save_images(orig_img, results, object_list, output_dir, file)
                    # 特徴ファイルを保存 / save feature file
                    save_feature(img_features, output_dir, file+'.npz')
                    # 指定した物体の座標を保存 / save specified object's coordinates
                    save_ebox(results, object_list, img_h, img_w, output_dir, 'e'+file+'.txt')
                # specfileを保存 save specfile
                #save_specfile(output_dir, img_features)

    #copy_file(output_dir)

