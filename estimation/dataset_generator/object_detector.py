# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 13:10:51 2019

@author: atsumilab
@code: utf-8
@update: removed yolo_v2 coding
"""

import chainer
#from chainercv.links import YOLOv2
from yolov3_feature_ext import YOLOv3
from chainer import cuda
import numpy as np
import cv2
import argparse

class ObjectDetector(object):
    """Object detector class"""
    coco_bbox_labels = ('person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
                        'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
                        'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
                        'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle',
                        'wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange',
                        'broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed',
                        'dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven',
                        'toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush') # 80
    voc_bbox_labels = ('aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
                       'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor') # 20

    def __init__(self, model_type, model_path, 
                 label_path, cfg_path, detection_threshold,
                 device=-1):
        """
           Args:
            model_type (str): model type 'yolo_v2' or 'yolo_v3'
            model_path (str): pretrained model path
            label_path (str): label path (v1.1)
            cfg_path (str): cfg file path (v1.1)
            detection_threshold (float): detection threshold (v1.1)
            device (int): GPU ID
        """
        # set bbox labels
        if label_path == '':
            if 'coco' in model_path:
                self.bbox_labels = self.coco_bbox_labels
            elif 'voc' in model_path:
                self.bbox_labels = self.voc_bbox_labels
            else:
                raise ValueError('An object label path must be specified.')
        else:
            self.bbox_labels = self.read_bbox_labels(label_path)
        # set bbox colors
        num_of_labels = len(self.bbox_labels)
        bbox_colors = np.zeros((num_of_labels, 3), dtype=int)
        ind = np.arange(num_of_labels, dtype=int)
        for shift in reversed(range(8)):
            for channel in range(3):
                bbox_colors[:, channel] |= ((ind >> channel) & 1) << shift
            ind >>= 3
        self.bbox_colors = tuple([(int(x[0]),int(x[1]),int(x[2])) for x in bbox_colors])
        # set model
        print('Loading ObjectNet...')
        """
        if model_type == 'yolo_v2':
            # set anchor
            if cfg_path != '':
                YOLOv2._anchors = self.read_anchors_v2(cfg_path)
            # construct model instance
            self.model = YOLOv2(n_fg_class=num_of_labels, pretrained_model=model_path)
        elif model_type == 'yolo_v3':
        """
        if model_type == 'yolo_v3':
            # set anchor
            if cfg_path != '':
                YOLOv3._anchors = self.read_anchors_v3(cfg_path)
            # construct model instance
            self.model = YOLOv3(n_fg_class=num_of_labels, pretrained_model=model_path)
        # set detection threshold
        self.model.score_thresh=detection_threshold
        # set device and gpu
        self.device = device
        if self.device >= 0:
            cuda.get_device_from_id(device).use()
            self.model.to_gpu()
    #
    def __call__(self, img):
        """
           Args:
            img (ndarray): image
           Returns:
            bbox (ndarray): box coordinate [left_top_y, left_top_x, right_bottom_y, right_bottom_x] list
            label (ndarray): label id list
            score (ndarray): score(float) list
        """
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb_img = np.asarray(rgb_img, dtype=np.float32)
        rgb_img = rgb_img.transpose((2, 0, 1))
        bboxes, labels, scores, layer_id, features = self.model.predict([rgb_img])
        bbox, label, score = bboxes[0], labels[0], scores[0]
        return bbox, label, score, layer_id, features
    #
    def get_bbox_labels(self):
        return self.bbox_labels
    #
    def get_bbox_colors(self):
        return self.bbox_colors
    #
    def read_bbox_labels(self, label_path):
        """Read bbox labels
           Args:
            label_path (str): label path
           Returns:
            tuple of object labels
        """
        with open(label_path, 'r') as f:
            lines = f.readlines()
        return tuple([line.strip() for line in lines])
    #
    def read_anchors_v2(self, cfg_path):
        """Read anchors of Yolo v2 from a cfg file
           Args:
            cfg_path (str): cfg file path
           Returns:
            tuple of anchors
        """
        with open(cfg_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.find('anchors') >= 0:
                anchors = list(map(float, line.replace(' ', '').strip('anchors=').split(',')))
            if line.find('num') >= 0:
                num = int(line.replace(' ', '').strip('num='))
        np_anchors = np.array(anchors).reshape(num, 2)
        return tuple(map(tuple, [np_anchors[i][::-1] for i in range(num)]))
    #
    def read_anchors_v3(self, cfg_path):
        """Read anchors of Yolo v3 from a cfg file
           Args:
            cfg_path (str): cfg file path
           Returns:
            tuple of anchors
        """
        with open(cfg_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.find('anchors') >= 0:
                anchors = list(map(float, line.replace(' ', '').strip('anchors=').split(',')))
            if line.find('mask') >= 0:
                len_mask = len(line.replace(' ', '').strip('mask=').split(','))
            if line.find('num') >= 0:
                num = int(line.replace(' ', '').strip('num='))
        anchors.reverse()
        np_anchors = np.array(anchors).reshape(int(num/len_mask), len_mask, 2)
        return tuple([tuple(map(tuple, np_anchors[i][::-1])) for i in range(int(num/len_mask))])
#
def draw_object_bbox(orig_img, bbox, label, score, bbox_labels, bbox_colors, no_person_bbox_drawing=False, bbox_with_score=True, bbox_thickness=2, bbox_label_color=(0, 0, 0), bbox_label_thickness=2):
    """
       Args:
        orig_img (ndarray): image
        bbox (ndarray): box coordinate [left_top_y, left_top_x, right_bottom_y, right_bottom_x] list
        label (ndarray): label id list
        score (ndarray): score(float) list
        bbox_labels (tuple): tuple of bbox names
        bbox_colors (tuple): tuple of bbox color (b, g, r) 
       Returns:
        image drawn
    """
    img = orig_img.copy()
    overlay = orig_img.copy()
    for i in range(len(bbox)):
        # no person bbox drawing
        if no_person_bbox_drawing and bbox_labels[label[i]] == 'person':
            continue
        # bbox coordinate
        left = bbox[i][1]
        top = bbox[i][0]
        right = bbox[i][3]
        bottom = bbox[i][2]
        # bbox drawing (as overlay)
        text = '%s' % (bbox_labels[label[i]])
        text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_label_thickness)[0][0]
        cv2.rectangle(overlay, (int(left), int(top)), (int(right), int(bottom)), bbox_colors[label[i]], bbox_thickness)
        if bbox_with_score:
            cv2.rectangle(overlay, (int(left), int(top)), (int(left)+max(text_width, 50), int(top)+30), (255,255,255), -1)
        else:
            cv2.rectangle(overlay, (int(left), int(top)), (int(left)+text_width, int(top)+15), (255,255,255), -1)
        cv2.putText(overlay, text, (int(left), int(top)+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_label_color, bbox_label_thickness)        
        if bbox_with_score:
            text = '(%.2f)' % (score[i])
            cv2.putText(overlay, text, (int(left), int(top)+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_label_color, bbox_label_thickness)
        img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
    return img
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', default='?') # 'image file path'|'?'
    parser.add_argument('--object_model_type', choices=('yolo_v2', 'yolo_v3'), default='yolo_v3')
    parser.add_argument('--object_model_path', default='models/yolo_v3_coco.npz') # '{ms coco|pascal voc|other} model path'|'?'
                                                                                  # If a path name contains a 'coco' or 'voc', it is not necessary to specify an object label path
    parser.add_argument('--object_label_path', default='') # must be specified other than 'coco' and 'voc'
    parser.add_argument('--object_cfg_path', default='?') # must be specified for 'yolo v3' other than 'voc' and 'coco' and 'yolo v2' other than 'voc'
    parser.add_argument('--object_detection_threshold', type=float, default=0.5)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--show_result', type=bool, default=True) # (v1.2)
    parser.add_argument('--save_result', type=bool, default=False) # (v1.2)
    args = parser.parse_args()
    # Select model path
    if args.object_model_path == '?':
        object_model_path = input('Input an object model path << ')
    else:
        object_model_path = args.object_model_path
    # Select object label and cfg path
    if args.object_label_path == '?':
        object_label_path = input('Input an object label path << ')
    else:
        object_label_path = args.object_label_path
    if args.object_cfg_path == '?':
        object_cfg_path = input('Input an object cfg path << ')
    else:
        object_cfg_path = args.object_cfg_path
    # setup model
    object_detector = ObjectDetector(args.object_model_type, object_model_path, 
                                     object_label_path, object_cfg_path, args.object_detection_threshold,
                                     device=args.gpu)
    # read image
    if args.img_path == '?':
        img_path = input('Input an image path << ')
    else:
        img_path = args.img_path
    img = cv2.imread(img_path)
    # detection
    bboxes, labels, scores = object_detector(img)
    # draw and save image
    img = draw_object_bbox(img, bboxes, labels, scores, object_detector.get_bbox_labels(), object_detector.get_bbox_colors())
    if args.show_result:
        cv2.imshow('result', img)
        cv2.waitKey(0)
    if args.save_result:
        print('Saving result into result.png...')
        cv2.imwrite('result.png', img)


