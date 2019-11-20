# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:39:26 2019

@author: atsumilab
@filename: gta2yolo.py
@code: utf-8
========================
Date          Comment
========================
11182019      First revision
11192019      Remove "other" category objects
11202019      Remove zero error to solve warning label
"""

import glob

IMG_WIDTH = 1920
IMG_HEIGHT = 1080

gta2kitdash = {21:7, # animal -> other
               22:4, # bicycle -> bicycle
               26:6, # bus -> bus
               24:0, # car -> car
               23:5, # motorcycle -> motorbike
               27:1, # truck -> truck
               20:2, # person -> person
               30:7, # plane -> other
               25:0, # van -> car
               28:1, # trailer -> truck
               16:7, # firehydrant -> other
               19:7, # trashcan -> other
               13:7, # trafficlight -> other
               17:7, # chair -> other
               7:7}  # tree -> other

unwanted_id = [21, 30, 16, 19, 13, 17, 7] #animal, plane, firehydrant, trashcan, trafficlight, chair, tree

def convert(csv_fname, out_fname):
    with open(csv_fname, 'r') as f:
        lines = f.readlines()
    with open(out_fname, 'w') as f:
        for line in lines:
            item = line.split(',')
            gtaclass_id, x1, y1, x2, y2 = list(map(int, item[1:6]))
            if gtaclass_id not in unwanted_id: #11192019   
                if((x2 != x1) and (y2 != y1)):  #11202019
                    kitdash_id = gta2kitdash[gtaclass_id]
                    cx = (x1 + (x2-x1)/2) / IMG_WIDTH
                    cy = (y1 + (y2-y1)/2) / IMG_HEIGHT
                    w = (x2-x1) / IMG_WIDTH
                    h = (y2-y1) / IMG_HEIGHT
                    w_list = list(map(str, [kitdash_id, cx, cy, w, h]))
                    f.write(' '.join(w_list) + '\n')

def main(path):
    # make csv file list
    csv_fnames = glob.glob(path + '/*.csv')
    for csv_fname in csv_fnames:
        output_fname = csv_fname.replace('.csv','.txt')
        convert(csv_fname, output_fname)

if __name__ == '__main__':
    path = input('input CSV Path ->')
    main(path)
     