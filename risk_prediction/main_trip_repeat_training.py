# coding: utf-8
# Copyright (C) 2017 Atsumi Laboratory. All rights reserved.

from risk_prediction.trip_trainer import TripTrainer
import os

#
if __name__ == '__main__':
    training_spec_file = input('Input training spec file (training_spec.txt): ')

    with open(training_spec_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    model_param_file_paths = [] # 20190510
    for line in lines:
        if line[0] == '#':
            continue
        elif line.startswith('train_ds1:'):
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
            model_param_file_paths.append(model_param_file_path) # 20190510
        elif line.startswith('tlog_path:'):
            tlog_path = line.split(':')[1].strip()
            tlog_path = os.path.join(os.path.dirname(training_spec_file), tlog_path)
        elif line.startswith('gpu_id:'):
            gpu_id = int(line.split(':')[1].strip())

    for count, model_param_file_path in enumerate(model_param_file_paths):
        print(count+1, '/', len(model_param_file_paths))
        repeat_tlog_path = os.path.splitext(tlog_path)[0] + '_({}).txt'.format(str(count+1))
        tripTrainer = TripTrainer(train_ds_path1, train_spec_file_name1, train_risk1,
                                  train_ds_path2, train_spec_file_name2, train_risk2,
                                  test_ds_path1, test_spec_file_name1, test_risk1,
                                  test_ds_path2, test_spec_file_name2, test_risk2,
                                  layer_name, box_type, 
                                  execution_mode, num_of_epoch, minibatch_size, eval_interval, save_interval,
                                  model_param_file_path, repeat_tlog_path, gpu_id)
        if execution_mode == 'train' or execution_mode == 'retrain':
            tripTrainer.learn_model()
        else:
            tripTrainer.test_model()