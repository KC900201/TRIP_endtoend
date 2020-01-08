# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 12:56:37 2019

@author: atsumilab
@filename: make_result_table.py
@coding: utf-8
========================
Date          Comment
========================
12112019      First revision
"""

from collections import OrderedDict
import os

log_file = 'map_log.txt'
if not os.path.isfile(log_file):
    log_file = input('input map_log.txt >')

with open(log_file, 'r') as f:
    lines = f.readlines()


ap_dict = OrderedDict()
header = False
for l, line in enumerate(lines):
    line_sp = line.split()
    if 'Total BFLOPS' in line:
        iteration = lines[l-1].strip()
    elif 'class_id = ' in line:
        name = 'ap_' + line_sp[5].strip(',')
        ap = line_sp[8]
        ap_dict.update([(name, ap)])
    elif 'precision = ' in line:
        precision = line_sp[6].strip(',')
        recall = line_sp[9].strip(',')
        F1_score = line_sp[12]
    elif 'average IoU = ' in line:
        average_IoU = line_sp[16]
    elif ' mean average precision (mAP) = ' in line:
        mAP = line_sp[7]        
        if not header:
            ap_names = ','.join(list(ap_dict.keys()))
            output = 'iteration,mAP,' + ap_names + ',precision,recall,F1_score,average_IoU\n'
            header = True
        ap_values = ','.join(list(ap_dict.values()))
        output += ','.join([iteration, mAP, ap_values, precision, recall, F1_score, average_IoU]) + '\n'

with open('result_table.csv', 'w') as w:
    w.write(output)