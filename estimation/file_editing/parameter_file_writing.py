"""
Created on Tue Sep 14 13:05:58 2020

@author: atsumilab
@filename: parameter_file_writing.py
@code: utf-8
========================
Date          Comment
========================
05122020      First revision
05132020      Test writing file in other directory
"""

import os
import shutil

if __name__ == '__main__':
    model_arch_comment = '# model parameter\n'
    model_arch = 'model_arch: MP-C-RRL-SPP4-LSTM\n'
    input_size = 'input_size: 1000\n'
    hidden_size = 'hidden_size: 100\n'
    roi_bg_comment = '# roi_bg: BG_ZERO | BG_GN std | BG_DP rate - GN:Gaussian Noise, DP:Depression\n'
    roi_bg = 'roi_bg: BG_ZERO\n'
    comparative_loss_margin = 'comparative_loss_margin: 0.3\n'
    risk_type = 'risk_type: seq_risk\n'
    threshold_of_similar_risk = 'threshold_of_similar_risk = 0.1\n'
    optimizer_info_comment_1 = '# adam | adadelta | adagrad [lr=0.001]  |rmsprop [lr=0.01] | momentum_sgd [lr=0.01 momentum=0.9] |\n'
    optimizer_info_comment_2 = '# nesterovag [lr=0.01 momentum=0.9] | rmspropgraves [lr=0.0001 momentum=0.9] | sgd [lr=0.01] | smorms3 [lr=0.001]\n'
    optimizer_info = 'optimizer_info: adam 0.001 0'

    for i in range(16, 21):
        # Opening a a specific file, with intention to write
        os.chdir('E:/AtsumiLabMDS-2/TRIP/Trip2018Q1/Dashcam') # 05132020
        paramFile = open('model_tbox_param' + str(i) + '.txt', 'w')
#        model_path = 'model_path: result/yolov3/model/yolov3/backup_ebox_accident_KitDashV_do_20191215/accident_KitDashV_6000_ebox_' + str(i) + '_model.npz\n'
        model_path = 'model_path: model/yolov3/backup_ebox_accident_A3DKitDashV_rrelu_20200505/accident_KitDashV_6000_ebox_' + str(i - 10) + '_model.npz\n'
    
        # Write text into file
        paramFile.write(model_arch_comment)    
        paramFile.write(model_path)
        paramFile.write(model_arch)
        paramFile.write(input_size)
        paramFile.write(hidden_size)
        paramFile.write(roi_bg_comment)
        paramFile.write(roi_bg)
        paramFile.write(comparative_loss_margin)
        paramFile.write(risk_type)
        paramFile.write(threshold_of_similar_risk)
        paramFile.write(optimizer_info_comment_1)
        paramFile.write(optimizer_info_comment_2)
        paramFile.write(optimizer_info)

        # Important to remember to close the file, otherwise it will
        # hang for a while and could cause problems in your script
        paramFile.close()


