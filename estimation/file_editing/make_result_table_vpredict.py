# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 17:28:26 2020

@author: setsu
@filename: make_result_table_vpredict.py
@coding: utf-8
========================
Date          Comment
========================
01152020      First revision
"""

from collections import OrderedDict
import os

if __name__ == '__main__':
    log_file = input('Input log file: ')

    with open(log_file, "r", encoding='utf-8') as f:
        lines = f.readlines()

    ap_dict = OrderedDict()
    header = False
    output = ''
    
    for l, line in enumerate(lines):
        line_sp = line.split()
        if 'risk of interval [0,9]' in line:
            ri_0 = line_sp[4].strip()
            name = line.split(':')[0].strip()
            ap_dict.update([(name, ri_0)])
        elif 'risk of interval [10,19]' in line:
            ri_1 = line_sp[4].strip()
            name = line.split(':')[0].strip()
            ap_dict.update([(name, ri_1)])
        elif 'risk of interval [20,29]' in line:
            ri_2 = line_sp[4].strip()
            name = line.split(':')[0].strip()
            ap_dict.update([(name, ri_2)])
        elif 'risk of interval [30,39]' in line:
            ri_3 = line_sp[4].strip()
            name = line.split(':')[0].strip()
            ap_dict.update([(name, ri_3)])
        elif 'risk of interval [40,49]' in line:
            ri_4 = line_sp[4].strip()
            name = line.split(':')[0].strip()
            ap_dict.update([(name, ri_4)])
        elif 'risk of interval [50,59]' in line:
            ri_5 = line_sp[4].strip()
            name = line.split(':')[0].strip()
            ap_dict.update([(name, ri_5)])
        elif 'risk of interval [60,69]' in line:
            ri_6 = line_sp[4].strip()
            name = line.split(':')[0].strip()
            ap_dict.update([(name, ri_6)])
        elif 'risk of interval [70,79]' in line:
            ri_7 = line_sp[4].strip()
            name = line.split(':')[0].strip()
            ap_dict.update([(name, ri_7)])
        elif 'risk of interval [80,89]' in line:
            ri_8 = line_sp[4].strip()
            name = line.split(':')[0].strip()
            ap_dict.update([(name, ri_8)])
        elif 'risk of interval [90,99]' in line:
            ri_9 = line_sp[4].strip()
            name = line.split(':')[0].strip()
            ap_dict.update([(name, ri_9)])
            ap_values = ','.join(list(ap_dict.values()))
            output += ','.join([ri_cat, ap_values]) + '\n'
            if not header:
                ap_names = ','.join(list(ap_dict.keys()))
                output = 'risk_class,' + ap_names + '\n'
                header = True
        else:
            ri_cat = line_sp[0]

        
    
    with open('risk_prediction_graph_town5.csv', 'w') as w:
        w.write(output)
