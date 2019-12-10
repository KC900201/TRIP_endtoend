# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 17:24:26 2019

@author: setsu
@filename: trip_trainer.py
@coding: utf-8
========================
Date          Comment
========================
09142019      First revision
10222019      Amend directory path for model_path
11302019      Enhance risk prediction training to have one more parameter for virtual data input
12042019      New function to train with virtual data (real, virtual, real + virtual)
12102019      Include one input for choosing testing data
"""

import chainer
from chainer import serializers, iterators, cuda, optimizers, datasets
import chainer.functions as F
from risk_prediction.trip_dataset import TripDataset
from risk_prediction.trip_lstm import TripLSTM
import numpy as np
import time
import os
# <ADD>
from risk_prediction.trip_c_lstm import TripCLSTM
# </ADD>

# 12102019
train_data_group = ['R', 'V', 'M']

class TripTrainer(object):
    """A class of TRIP(Traffic Risk Prediction) trainer
    """
    def __init__(self, train_ds_path1, train_spec_file_name1, train_risk1,
                       train_ds_path2, train_spec_file_name2, train_risk2,
                       test_ds_path1, test_spec_file_name1, test_risk1,
                       test_ds_path2, test_spec_file_name2, test_risk2,
                       vtrain_ds_path1, vtrain_spec_file_name1, vtrain_risk1, #11302019
                       vtrain_ds_path2, vtrain_spec_file_name2, vtrain_risk2, #11302019
#                       vtest_ds_path1, vtest_spec_file_name1, vtest_risk1,    #11302019
#                       vtest_ds_path2, vtest_spec_file_name2, vtest_risk2,    #11302019
                       layer_name, box_type, execution_mode, num_of_epoch, minibatch_size, 
                       eval_interval, save_interval, model_param_file_path, tlog_path, gpu_id):
        """ Constructor
            Args:
             train_ds_path1 (str): a train dataset path 1
             train_spec_file_name1 (str): a train dataset spec file name 1
             train_risk1 (int): risk level of train dataset 1 which is 0 for no-accident and 1 for accident
             train_ds_path2 (str): a train dataset path 2
             train_spec_file_name2 (str): a train dataset spec file name 2
             train_risk2 (int): risk level of train dataset 2 which is 0 for no-accident and 1 for accident
             test_ds_path1 (str): a test dataset path 1
             test_spec_file_name1 (str): a test dataset spec file name 
             test_risk1 (int): risk level of test dataset 1 which is 0 for no-accident and 1 for accident
             test_ds_path1 (str): a test dataset path 2
             test_spec_file_name1 (str): a test dataset spec file name 2
             test_risk1 (int): risk level of test dataset 2 which is 0 for no-accident and 1 for accident
             vtrain_ds_path1 (str): a train dataset path 1 for virtual data
             vtrain_spec_file_name1 (str): a train dataset spec file name 1 for virtual data
             vtrain_risk1 (int): risk level of train dataset 1 which is 0 for no-accident and 1 for accident for virtual data
             vtrain_ds_path2 (str): a train dataset path 2 for virtual data
             vtrain_spec_file_name2 (str): a train dataset spec file name 2 for virtual data
             vtrain_risk2 (int): risk level of train dataset 2 which is 0 for no-accident and 1 for accident for virtual data
             vtest_ds_path1 (str): a test dataset path 1 for virtual data
             vtest_spec_file_name1 (str): a test dataset spec file name for virtual data
             vtest_risk1 (int): risk level of test dataset 1 which is 0 for no-accident and 1 for accident for virtual data
             vtest_ds_path1 (str): a test dataset path 2 for virtual data
             vtest_spec_file_name1 (str): a test dataset spec file name 2 for virtual data
             vtest_risk1 (int): risk level of test dataset 2 which is 0 for no-accident and 1 for accident for virtual data
             layer_name (str): a layer name
             execution_mode (str): execution mode (train | retrain | test)
             num_of_epoch (int): the number of epochs
             minibatch_size (int): the size of minibatch
             eval_interval (int): evaluation interval
             save_interval (int): save interval
             box_type (str): a type of boxes - 'tbox' or 'ebox'
             model_param_file_path (str): a model parameter file path
             tlog_path (str): a training log file path
             gpu_id (int): GPU ID (-1 for CPU) 
        """
        # set dataset
        self.train_ds1 = TripDataset(train_ds_path1, train_spec_file_name1, layer_name, box_type)
        self.train_ds2 = TripDataset(train_ds_path2, train_spec_file_name2, layer_name, box_type)
        self.train_risk1 = train_risk1
        self.train_risk2 = train_risk2
        self.test_ds1 = TripDataset(test_ds_path1, test_spec_file_name1, layer_name, box_type)
        self.test_ds2 = TripDataset(test_ds_path2, test_spec_file_name2, layer_name, box_type)
        self.test_risk1 = test_risk1
        self.test_risk2 = test_risk2
        # 11302019
        self.vtrain_ds1 = TripDataset(vtrain_ds_path1, vtrain_spec_file_name1, layer_name, box_type)
        self.vtrain_ds2 = TripDataset(vtrain_ds_path2, vtrain_spec_file_name2, layer_name, box_type)
        self.vtrain_risk1 = vtrain_risk1
        self.vtrain_risk2 = vtrain_risk2
#        self.vtest_ds1 = TripDataset(vtest_ds_path1, vtest_spec_file_name1, layer_name, box_type)
#        self.vtest_ds2 = TripDataset(vtest_ds_path2, vtest_spec_file_name2, layer_name, box_type)
#        self.vtest_risk1 = vtest_risk1
#        self.vtest_risk2 = vtest_risk2
        # check dataset
        train_ds_length1 = self.train_ds1.get_length()
        train_ds_length2 = self.train_ds2.get_length()
        test_ds_length1 = self.test_ds1.get_length()
        test_ds_length2 = self.test_ds2.get_length()
        # 11302019
        vtrain_ds_length1 = self.vtrain_ds1.get_length()
        vtrain_ds_length2 = self.vtrain_ds2.get_length()
#        vtest_ds_length1 = self.vtest_ds1.get_length()
#        vtest_ds_length2 = self.vtest_ds2.get_length()
        if train_ds_length1 == train_ds_length2:
            self.train_ds_length = train_ds_length1
        else:
            raise ValueError('Mismatch of training dataset length')
        if test_ds_length1 == test_ds_length2:
            self.test_ds_length = test_ds_length1
        else:
            raise ValueError('Mismatch of training dataset length')
        # 11302019
        if vtrain_ds_length1 == vtrain_ds_length2:
            self.vtrain_ds_length = vtrain_ds_length1
        else:
            raise ValueError('Mismatch of virtual training dataset length')
#        if vtest_ds_length1 == vtest_ds_length2:
#            self.vtest_ds_length = vtest_ds_length1
#        else:
#            raise ValueError('Mismatch of virtual training dataset length')
        train_layer_info1 = self.train_ds1.get_layer_info()
        train_layer_info2 = self.train_ds2.get_layer_info()
        test_layer_info1 = self.test_ds1.get_layer_info()
        test_layer_info2 = self.test_ds2.get_layer_info()
        if (train_layer_info1 != train_layer_info2) or (test_layer_info1 != test_layer_info2) or (train_layer_info1 != test_layer_info1):
            raise ValueError('Mismatch of layer infos')
        train_feature_type1 = self.train_ds1.get_feature_type()
        train_feature_type2 = self.train_ds2.get_feature_type()
        if train_feature_type1 != train_feature_type2:
            raise ValueError('Mismatch of training feature types')
        train_box_type1 = self.train_ds1.get_box_type()
        train_box_type2 = self.train_ds2.get_box_type()
        if train_box_type1 != train_box_type2:
            raise ValueError('Mismatch of training box types')
        test_feature_type1 = self.test_ds1.get_feature_type()
        test_feature_type2 = self.test_ds2.get_feature_type()
        if test_feature_type1 != test_feature_type2:
            raise ValueError('Mismatch of test feature types')
        test_box_type1 = self.test_ds1.get_box_type()
        test_box_type2 = self.test_ds2.get_box_type()
        if test_box_type1 != test_box_type2:
            raise ValueError('Mismatch of test box types')
        # 11302019
        vtrain_layer_info1 = self.vtrain_ds1.get_layer_info()
        vtrain_layer_info2 = self.vtrain_ds2.get_layer_info()
#        vtest_layer_info1 = self.vtest_ds1.get_layer_info()
#        vtest_layer_info2 = self.vtest_ds2.get_layer_info()
        if (vtrain_layer_info1 != vtrain_layer_info2):
            raise ValueError('Mismatch of layer infos')
        vtrain_feature_type1 = self.vtrain_ds1.get_feature_type()
        vtrain_feature_type2 = self.vtrain_ds2.get_feature_type()
        if vtrain_feature_type1 != vtrain_feature_type2:
            raise ValueError('Mismatch of virtual training feature types')
        vtrain_box_type1 = self.vtrain_ds1.get_box_type()
        vtrain_box_type2 = self.vtrain_ds2.get_box_type()
        if vtrain_box_type1 != vtrain_box_type2:
            raise ValueError('Mismatch of virtual training box types')
#        vtest_feature_type1 = self.vtest_ds1.get_feature_type()
#        vtest_feature_type2 = self.vtest_ds2.get_feature_type()
#        if vtest_feature_type1 != vtest_feature_type2:
#            raise ValueError('Mismatch of virtual test feature types')
#        vtest_box_type1 = self.vtest_ds1.get_box_type()
#        vtest_box_type2 = self.vtest_ds2.get_box_type()
#        if vtest_box_type1 != vtest_box_type2:
#            raise ValueError('Mismatch of virtual test box types')
        # set gpu
        self.gpu_id = gpu_id
        if self.gpu_id >= 0:
            cuda.get_device_from_id(self.gpu_id).use()
        # set model
        self.set_model(execution_mode, model_param_file_path)
        # epoch, minibatch, eval interval, save interval, training log file path
        self.num_of_epoch = num_of_epoch
        self.minibatch_size = minibatch_size
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.tlog_path = tlog_path
        self.tlogf = None
    #
    def set_model(self, execution_mode, model_param_file_path):
        """ Set a model and its parameters
            Args:
             execution_mode (str): execution mode (train | retrain | test)
             model_path (str): a model parameter file path
        """
        self.execution_mode = execution_mode
        # default parameters
        self.model_path = './model/trip_model.npz'
        # <ADD>
        self.model_arch = 'FC-LSTM'
        # </ADD>
        self.input_size = 1024
        self.hidden_size = 50
        # <ADD>
        self.roi_bg = ('BG_ZERO')
        # </ADD>
        self.comparative_loss_margin = 0.1
        self.risk_type = 'seq_risk'
        self.threshold_of_similar_risk = 0.1
        self.optimizer_info = 'adam'
        # read parameters
        with open(model_param_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            if line[0] == '#':
                continue
            param = line.split(':')
            if param[0].strip() == 'model_path':                
#                self.model_path = os.path.join(os.path.dirname(model_param_file_path), param[1].strip()) 
                self.model_path = param[1].strip() #10222019
            # <ADD>
            elif param[0].strip() == 'model_arch':
                self.model_arch = param[1].strip()
            # </ADD>
            elif param[0].strip() == 'input_size':
                self.input_size = int(param[1].strip())
            elif param[0].strip() == 'hidden_size':
                self.hidden_size = int(param[1].strip())
            # <ADD>
            elif param[0].strip() == 'roi_bg':
                bg_param = param[1].strip().split()
                if len(bg_param) == 2:
                    bg_param[1] = float(bg_param[1])
                self.roi_bg = tuple(bg_param)
            # </ADD>
            elif param[0].strip() == 'comparative_loss_margin':
                self.comparative_loss_margin = float(param[1].strip())
            elif param[0].strip() == 'risk_type': # { 'seq_risk' | 'seq_mean_risk' }
                self.risk_type = param[1].strip()
            elif param[0].strip() == 'threshold_of_similar_risk':
                self.threshold_of_similar_risk = float(param[1].strip())
            elif param[0].strip() == 'optimizer_info': # { 'adam' | 'adagrad [lr=0.001]' | 'momentum_sgd [lr=0.01] [momentum=0.9]' | 'rmsprop [lr=0.01]' |
                self.optimizer_info = param[1].strip() #   'adadelta' | 'rmspropgraves [lr=0.0001 momentum=0.9]' | 'sgd [lr=0.01]' |
            else:                                      #   'nesterovag [lr=0.01 momentum=0.9]' | 'smorms3 [lr=0.001]'}
                continue
        # set model
        # <MOD>
        if self.model_arch == 'FC-LSTM':
            self.model = TripLSTM(self.input_size, self.hidden_size)
        else:
            self.model = TripCLSTM(self.input_size, self.hidden_size, self.model_arch)
        # </MOD>
        if self.execution_mode == 'train' or self.execution_mode == 'retrain':
            # set optimizer
            optimizer_info = self.optimizer_info.split()
            if optimizer_info[0] == 'adam':
                #add parameters - 20190506
                if len(optimizer_info) == 3:
                    learning_rate = float(optimizer_info[1].strip())
                    weight_decay = float(optimizer_info[2].strip())
                    self.optimizer = optimizers.Adam(alpha=learning_rate, weight_decay_rate=weight_decay)
                elif len(optimizer_info) == 2:
                    learning_rate = float(optimizer_info[1].strip())
                    self.optimizer = optimizers.Adam(alpha=learning_rate)
                else:
                    self.optimizer = optimizers.Adam()
                #self.optimizer = optimizers.Adam()
            elif optimizer_info[0] == 'adadelta':
                self.optimizer = optimizers.AdaDelta()
            elif optimizer_info[0] == 'adagrad':
                if len(optimizer_info) == 2:
                    learning_rate = float(optimizer_info[1].strip())
                    self.optimizer = optimizers.AdaGrad(lr=learning_rate)
                else:
                    self.optimizer = optimizers.AdaGrad()
            elif optimizer_info[0] == 'momentum_sgd':
                if len(optimizer_info) == 3:
                    learning_rate = float(optimizer_info[1].strip())
                    momentum_val = float(optimizer_info[2].strip())
                    self.optimizer = optimizers.MomentumSGD(lr=learning_rate, momentum=momentum_val)
                elif len(optimizer_info) == 2:
                    learning_rate = float(optimizer_info[1].strip())
                    self.optimizer = optimizers.MomentumSGD(lr=learning_rate)
                else:
                    self.optimizer = optimizers.MomentumSGD()
            elif optimizer_info[0] == 'rmsprop':
                if len(optimizer_info) == 2:
                    learning_rate = float(optimizer_info[1].strip())
                    self.optimizer = optimizers.RMSprop(lr=learning_rate)
                else:
                    self.optimizer = optimizers.RMSprop()
            elif optimizer_info[0] == 'rmspropgraves':
                if len(optimizer_info) == 3:
                    learning_rate = float(optimizer_info[1].strip())
                    momentum_val = float(optimizer_info[2].strip())
                    self.optimizer = optimizers.RMSpropGraves(lr=learning_rate, momentum=momentum_val)
                elif len(optimizer_info) == 2:
                    learning_rate = float(optimizer_info[1].strip())
                    self.optimizer = optimizers.RMSpropGraves(lr=learning_rate)
                else:
                    self.optimizer = optimizers.RMSpropGraves()
            elif optimizer_info[0] == 'sgd':
                if len(optimizer_info) == 2:
                    learning_rate = float(optimizer_info[1].strip())
                    self.optimizer = optimizers.SGD(lr=learning_rate)
                else:
                    self.optimizer = optimizers.SGD()
            elif optimizer_info[0] == 'nesterovag':
                if len(optimizer_info) == 3:
                    learning_rate = float(optimizer_info[1].strip())
                    momentum_val = float(optimizer_info[2].strip())
                    self.optimizer = optimizers.NesterovAG(lr=learning_rate, momentum=momentum_val)
                elif len(optimizer_info) == 2:
                    learning_rate = float(optimizer_info[1].strip())
                    self.optimizer = optimizers.NesterovAG(lr=learning_rate)
                else:
                    self.optimizer = optimizers.NesterovAG()
            elif optimizer_info[0] == 'smorms3':
                if len(optimizer_info) == 2:
                    learning_rate = float(optimizer_info[1].strip())
                    self.optimizer = optimizers.SMORMS3(lr=learning_rate)
                else:
                    self.optimizer = optimizers.SMORMS3()
            else:
                raise ValueError('Illegal optimizer info')
            # set model to optimizer
            if self.execution_mode == 'train':
                # set model to optimizer
                self.optimizer.setup(self.model)
            else: # 'retrain'
                # load a model, setup and load an optimizer for resuming training (the order is important; if an optimizer is loaded before a model setup, "optimizer has no attribute 't'" error.)
                print('Loading a model: {}'.format(self.model_path))
                serializers.load_npz(self.model_path, self.model) # load a model
                self.optimizer.setup(self.model) # set the model to an optimizer
                serializers.load_npz(self.model_path.replace('model.npz','optimizer.npz'), self.optimizer) # load an optimizer
                print(' done')
        else: # 'test'
            # load a model for test
            print('Loading a model: {}'.format(self.model_path))
            serializers.load_npz(self.model_path, self.model)
            print(' done')
        if self.gpu_id >= 0:
            self.model.to_gpu() # self.xp of Link is set to return cuda,cupy (which originally returns numpy)
    #
    def open_log_file(self):
        """ Open a log file
        """
        if self.execution_mode == 'train':
            self.tlogf = open(self.tlog_path, 'x', encoding='utf-8') # 20190510
        elif self.execution_mode == 'retrain':
            self.tlogf = open(self.tlog_path, 'a', encoding='utf-8')
    #
    def close_log_file(self):
        """ Close a log file
        """
        if self.tlogf is not None:
            self.tlogf.close()
    #
    def write_log_header(self):
        self.tlogf.write('[Head]\n')
        self.tlogf.write('Train DS 1: {0}, {1}\n'.format(os.path.basename(self.train_ds1.ds_path), self.train_risk1))
        self.tlogf.write('Train DS 2: {0}, {1}\n'.format(os.path.basename(self.train_ds2.ds_path), self.train_risk2))
        self.tlogf.write('Test DS 1: {0}, {1}\n'.format(os.path.basename(self.test_ds1.ds_path), self.test_risk1))
        self.tlogf.write('Test DS 2: {0}, {1}\n'.format(os.path.basename(self.test_ds2.ds_path), self.test_risk2))
        self.tlogf.write('Train DS length: {}\n'.format(self.train_ds_length))
        self.tlogf.write('Test DS length: {}\n'.format(self.test_ds_length))
        # 11302019
        self.tlogf.write('Virtual Train DS 1: {0}, {1}\n'.format(os.path.basename(self.vtrain_ds1.ds_path), self.vtrain_risk1))
        self.tlogf.write('Virtual Train DS 2: {0}, {1}\n'.format(os.path.basename(self.vtrain_ds2.ds_path), self.vtrain_risk2))
#        self.tlogf.write('Virtual Test DS 1: {0}, {1}\n'.format(os.path.basename(self.vtest_ds1.ds_path), self.vtest_risk1))
#        self.tlogf.write('Virtual Test DS 2: {0}, {1}\n'.format(os.path.basename(self.vtest_ds2.ds_path), self.vtest_risk2))
        self.tlogf.write('Virtual Train DS length: {}\n'.format(self.vtrain_ds_length))
#        self.tlogf.write('Virtual Test DS length: {}\n'.format(self.vtest_ds_length))
        layer_info = self.train_ds1.get_layer_info()
        self.tlogf.write('Layer: {0} ({1},{2},{3})\n'.format(layer_info[0], layer_info[1], layer_info[2], layer_info[3]))
        self.tlogf.write('Box type: {}\n'.format(self.train_ds1.get_box_type()))
        # 11302019
        vlayer_info = self.vtrain_ds1.get_layer_info()
        self.tlogf.write('Layer (Virtual data): {0} ({1},{2},{3})\n'.format(vlayer_info[0], vlayer_info[1], vlayer_info[2], vlayer_info[3]))
        self.tlogf.write('Box type (Virtual data): {}\n'.format(self.vtrain_ds1.get_box_type()))
        self.tlogf.write('Model: {}\n'.format(os.path.basename(self.model_path)))
 
        # <ADD>
        self.tlogf.write('Model arch: {}\n'.format(self.model_arch))
        # </ADD>
        self.tlogf.write('Input size: {}\n'.format(self.input_size))
        self.tlogf.write('Hidden size: {}\n'.format(self.hidden_size))
        # <ADD>
        if len(self.roi_bg) == 2:
            bg = '('+self.roi_bg[0]+','+str(self.roi_bg[1])+')'
        else:
            bg = '('+self.roi_bg[0]+')'
        self.tlogf.write('ROI BG: {}\n'.format(bg))
        # </ADD>
        self.tlogf.write('Risk type: {}\n'.format(self.risk_type))
        self.tlogf.write('Comparative loss margin: {}\n'.format(self.comparative_loss_margin))
        self.tlogf.write('Optimizer info: {}\n'.format(self.optimizer_info))
        self.tlogf.write('Minibatch size: {}\n'.format(self.minibatch_size))
        self.tlogf.write('Threshold of similar risk: {}\n'.format(self.threshold_of_similar_risk))
        self.tlogf.write('[Body]\n')
        self.tlogf.flush()
    #
    def learn_model(self):
        """ Learning (without trainer)
        """
        # open a log file
        self.open_log_file()
        self.write_log_header()
        # redefine the number of epochs (for retrain)
        start_epoch = self.optimizer.epoch  # optimizer.epoch: the number of epochs (epoch value is 0 when optimizer is generated)
                                            # cf. optimizer.t: the number of iterations
        num_of_epoch = self.num_of_epoch - start_epoch
        # set iterators
        train_iterator1 = iterators.MultithreadIterator(self.train_ds1, self.minibatch_size)
        train_iterator2 = iterators.MultithreadIterator(self.train_ds2, self.minibatch_size)
        # training loop
        epoch = 0
        while train_iterator1.epoch < num_of_epoch:
            # preprocessing
            if epoch == 0 or train_iterator1.is_new_epoch:
                self.optimizer.new_epoch() # increment the optimizer.epoch by calling new_epock() for retrain
                epoch = train_iterator1.epoch + 1
                cur_epoch = start_epoch + epoch
                print('Epoch: {0:d}'.format(cur_epoch))
                if self.tlogf is not None:
                    self.tlogf.write('Epoch: {0:d}\n'.format(cur_epoch))
                epoch_loss = 0.0
                start_time = time.time()
            # get a minibatch of each dataset (a list of examples)
            train_batch1 = train_iterator1.next() # a list of minibatch elements of train dataset 1
            train_batch2 = train_iterator2.next() # a list of minibatch elements of train dataset 2
            # prepare input sequences
            input_feature_seq1 = self.train_ds1.prepare_input_sequence(train_batch1, self.roi_bg) # <ADD self.roi_bg/>
            input_feature_seq2 = self.train_ds2.prepare_input_sequence(train_batch2, self.roi_bg) # <ADD self.roi_bg/>
            # forward recurrent propagation and risk prediction
            if self.risk_type == 'seq_risk':
                r1 = self.model.predict_risk(input_feature_seq1)
                r2 = self.model.predict_risk(input_feature_seq2)
            elif self.risk_type == 'seq_mean_risk':
                r1 = self.model.predict_mean_risk(input_feature_seq1)
                r2 = self.model.predict_mean_risk(input_feature_seq2)
            # compute comparative loss
            rel = self.compare_risk_level(train_batch1, train_batch2, self.train_risk1, self.train_risk2)
            batch_loss = self.model.comparative_loss(r1, r2, rel, self.comparative_loss_margin)
            epoch_loss += float(cuda.to_cpu(batch_loss.data))
            # backward propagation and update parameters
            self.model.cleargrads()
            batch_loss.backward()
            batch_loss.unchain_backward() # unchain backward for each video clip
            self.optimizer.update()
            # post processing
            if train_iterator1.is_new_epoch:
                # result
                end_time = time.time()
                print(' loss: {0:.6f} ({1:.2f} sec)'.format(epoch_loss, end_time-start_time))
                if self.tlogf is not None:
                    self.tlogf.write(' loss: {0:.6f} ({1:.2f} sec)\n'.format(epoch_loss, end_time-start_time))
                    self.tlogf.flush()
                # evaluation
                if (cur_epoch == 1) or (epoch % self.eval_interval == 0) or (epoch == num_of_epoch):
                    print(' train data evaluation:')
                    self.evaluate('train')
                    print(' test data evaluation:')
                    self.evaluate('test')
                    print()
                # save interim model and optimizer
                if self.save_interval is not None and ((epoch % self.save_interval == 0) or (epoch == num_of_epoch)):
                    interim_model_path = self.model_path[:-4] + '.' + str(cur_epoch) + self.model_path[-4:]
                    interim_optimizer_path = self.model_path.replace('model.npz','optimizer.npz')
                    interim_optimizer_path = interim_optimizer_path[:-4] + '.' + str(cur_epoch) + interim_optimizer_path[-4:]
                    print('Saving an interim model: {}'.format(interim_model_path))
                    serializers.save_npz(interim_model_path, self.model)
                    serializers.save_npz(interim_optimizer_path, self.optimizer)
                    print(' done')
        # save final model and optimizer
        print('Saving a final model: {}'.format(self.model_path))
        serializers.save_npz(self.model_path, self.model)
        serializers.save_npz(self.model_path.replace('model.npz','optimizer.npz'), self.optimizer)
        print(' done')
        # close a log file
        self.close_log_file()
    # 11302019
    def learn_model_mix(self): # 12042019
        """ Learning (without trainer)
        """
        # open a log file
        self.open_log_file()
        self.write_log_header()
        # redefine the number of epochs (for retrain)
        start_epoch = self.optimizer.epoch  # optimizer.epoch: the number of epochs (epoch value is 0 when optimizer is generated)
                                            # cf. optimizer.t: the number of iterations
        num_of_epoch = self.num_of_epoch - start_epoch
        # set iterators (real + virtual)
        train_iterator1 = iterators.MultithreadIterator(datasets.ConcatenatedDataset(self.train_ds1, self.vtrain_ds1), self.minibatch_size)
        train_iterator2 = iterators.MultithreadIterator(datasets.ConcatenatedDataset(self.train_ds2, self.vtrain_ds2), self.minibatch_size)
        # training loop
        epoch = 0
        while train_iterator1.epoch < num_of_epoch:
            # preprocessing
            if epoch == 0 or train_iterator1.is_new_epoch:
                self.optimizer.new_epoch() # increment the optimizer.epoch by calling new_epock() for retrain
                epoch = train_iterator1.epoch + 1
                cur_epoch = start_epoch + epoch
                print('Epoch: {0:d}'.format(cur_epoch))
                if self.tlogf is not None:
                    self.tlogf.write('Epoch: {0:d}\n'.format(cur_epoch))
                epoch_loss = 0.0
                start_time = time.time()
            # get a minibatch of each dataset (a list of examples)
            train_batch1 = train_iterator1.next() # a list of minibatch elements of train dataset 1
            train_batch2 = train_iterator2.next() # a list of minibatch elements of train dataset 2
            # prepare input sequences
            input_feature_seq1 = self.train_ds1.prepare_input_sequence(train_batch1, self.roi_bg) # <ADD self.roi_bg/>
            input_feature_seq2 = self.train_ds2.prepare_input_sequence(train_batch2, self.roi_bg) # <ADD self.roi_bg/>
            input_feature_vseq1 = self.vtrain_ds1.prepare_input_sequence(train_batch1, self.roi_bg) # <ADD self.roi_bg/>
            input_feature_vseq2 = self.vtrain_ds2.prepare_input_sequence(train_batch2, self.roi_bg) # <ADD self.roi_bg/>
            # forward recurrent propagation and risk prediction
            if self.risk_type == 'seq_risk':
                r1 = self.model.predict_risk(input_feature_seq1)
                r2 = self.model.predict_risk(input_feature_seq2)
                vr1 = self.model.predict_risk(input_feature_vseq1)
                vr2 = self.model.predict_risk(input_feature_vseq2)
            elif self.risk_type == 'seq_mean_risk':
                r1 = self.model.predict_mean_risk(input_feature_seq1)
                r2 = self.model.predict_mean_risk(input_feature_seq2)
                vr1 = self.model.predict_mean_risk(input_feature_vseq1)
                vr2 = self.model.predict_mean_risk(input_feature_vseq2)
            # compute comparative loss
            rel = self.compare_risk_level(train_batch1, train_batch2, self.train_risk1 + self.vtrain_risk1, self.train_risk2 + self.vtrain_risk2)
            batch_loss = self.model.comparative_loss(r1 + vr1, r2 + vr2, rel, self.comparative_loss_margin)
            epoch_loss += float(cuda.to_cpu(batch_loss.data))
            # backward propagation and update parameters
            self.model.cleargrads()
            batch_loss.backward()
            batch_loss.unchain_backward() # unchain backward for each video clip
            self.optimizer.update()
            # post processing
            if train_iterator1.is_new_epoch:
                # result
                end_time = time.time()
                print(' loss: {0:.6f} ({1:.2f} sec)'.format(epoch_loss, end_time-start_time))
                if self.tlogf is not None:
                    self.tlogf.write(' loss: {0:.6f} ({1:.2f} sec)\n'.format(epoch_loss, end_time-start_time))
                    self.tlogf.flush()
                # evaluation
                if (cur_epoch == 1) or (epoch % self.eval_interval == 0) or (epoch == num_of_epoch):
                    print(' train data evaluation:')
                    self.evaluate('train')
                    print(' test data evaluation:')
                    self.evaluate('test')
                    print()
                # save interim model and optimizer
                if self.save_interval is not None and ((epoch % self.save_interval == 0) or (epoch == num_of_epoch)):
                    interim_model_path = self.model_path[:-4] + '.' + str(cur_epoch) + self.model_path[-4:]
                    interim_optimizer_path = self.model_path.replace('model.npz','optimizer.npz')
                    interim_optimizer_path = interim_optimizer_path[:-4] + '.' + str(cur_epoch) + interim_optimizer_path[-4:]
                    print('Saving an interim model: {}'.format(interim_model_path))
                    serializers.save_npz(interim_model_path, self.model)
                    serializers.save_npz(interim_optimizer_path, self.optimizer)
                    print(' done')
        # save final model and optimizer
        print('Saving a final model: {}'.format(self.model_path))
        serializers.save_npz(self.model_path, self.model)
        serializers.save_npz(self.model_path.replace('model.npz','optimizer.npz'), self.optimizer)
        print(' done')
        # close a log file
        self.close_log_file() 
    
    def learn_model_virtual(self): 
        """ Learning (without trainer)
        """
        # open a log file
        self.open_log_file()
        self.write_log_header()
        # redefine the number of epochs (for retrain)
        start_epoch = self.optimizer.epoch  # optimizer.epoch: the number of epochs (epoch value is 0 when optimizer is generated)
                                            # cf. optimizer.t: the number of iterations
        num_of_epoch = self.num_of_epoch - start_epoch
        # set iterators (real + virtual)
        train_iterator1 = iterators.MultithreadIterator(self.vtrain_ds1, self.minibatch_size)
        train_iterator2 = iterators.MultithreadIterator(self.vtrain_ds2, self.minibatch_size)
        # training loop
        epoch = 0
        while train_iterator1.epoch < num_of_epoch:
            # preprocessing
            if epoch == 0 or train_iterator1.is_new_epoch:
                self.optimizer.new_epoch() # increment the optimizer.epoch by calling new_epock() for retrain
                epoch = train_iterator1.epoch + 1
                cur_epoch = start_epoch + epoch
                print('Epoch: {0:d}'.format(cur_epoch))
                if self.tlogf is not None:
                    self.tlogf.write('Epoch: {0:d}\n'.format(cur_epoch))
                epoch_loss = 0.0
                start_time = time.time()
            # get a minibatch of each dataset (a list of examples)
            train_batch1 = train_iterator1.next() # a list of minibatch elements of train dataset 1
            train_batch2 = train_iterator2.next() # a list of minibatch elements of train dataset 2
            # prepare input sequences
            input_feature_seq1 = self.vtrain_ds1.prepare_input_sequence(train_batch1, self.roi_bg) # <ADD self.roi_bg/>
            input_feature_seq2 = self.vtrain_ds2.prepare_input_sequence(train_batch2, self.roi_bg) # <ADD self.roi_bg/>
            # forward recurrent propagation and risk prediction
            if self.risk_type == 'seq_risk':
                r1 = self.model.predict_risk(input_feature_seq1)
                r2 = self.model.predict_risk(input_feature_seq2)
            elif self.risk_type == 'seq_mean_risk':
                r1 = self.model.predict_mean_risk(input_feature_seq1)
                r2 = self.model.predict_mean_risk(input_feature_seq2)
            # compute comparative loss
            rel = self.compare_risk_level(train_batch1, train_batch2, self.vtrain_risk1, self.vtrain_risk2)
            batch_loss = self.model.comparative_loss(r1, r2, rel, self.comparative_loss_margin)
            epoch_loss += float(cuda.to_cpu(batch_loss.data))
            # backward propagation and update parameters
            self.model.cleargrads()
            batch_loss.backward()
            batch_loss.unchain_backward() # unchain backward for each video clip
            self.optimizer.update()
            # post processing
            if train_iterator1.is_new_epoch:
                # result
                end_time = time.time()
                print(' loss: {0:.6f} ({1:.2f} sec)'.format(epoch_loss, end_time-start_time))
                if self.tlogf is not None:
                    self.tlogf.write(' loss: {0:.6f} ({1:.2f} sec)\n'.format(epoch_loss, end_time-start_time))
                    self.tlogf.flush()
                # evaluation
                if (cur_epoch == 1) or (epoch % self.eval_interval == 0) or (epoch == num_of_epoch):
                    print(' train data evaluation:')
                    self.evaluate('train')
                    print(' test data evaluation:')
                    self.evaluate('test')
                    print()
                # save interim model and optimizer
                if self.save_interval is not None and ((epoch % self.save_interval == 0) or (epoch == num_of_epoch)):
                    interim_model_path = self.model_path[:-4] + '.' + str(cur_epoch) + self.model_path[-4:]
                    interim_optimizer_path = self.model_path.replace('model.npz','optimizer.npz')
                    interim_optimizer_path = interim_optimizer_path[:-4] + '.' + str(cur_epoch) + interim_optimizer_path[-4:]
                    print('Saving an interim model: {}'.format(interim_model_path))
                    serializers.save_npz(interim_model_path, self.model)
                    serializers.save_npz(interim_optimizer_path, self.optimizer)
                    print(' done')
        # save final model and optimizer
        print('Saving a final model: {}'.format(self.model_path))
        serializers.save_npz(self.model_path, self.model)
        serializers.save_npz(self.model_path.replace('model.npz','optimizer.npz'), self.optimizer)
        print(' done')
        # close a log file
        self.close_log_file()

    #
    def compare_risk_level(self, batch1, batch2, risk1, risk2):
        """ Compare risk level
            Args:
             batch1 (list of dataset samples): a list of samples of dataset 1 
             batch2 (list of dataset samples): a list of samples of dataset 2
             risk1 (int): risk level of dataset 1 ({1, 0})
             risk2 (int): risk level of dataset 2 ({1, 0})
            Returns:
             comparison result (numpy array): each element is one of {[1], [-1], [0]}
        """
        if risk1 > risk2:
            return np.array([[1]]*len(batch1))
        elif risk1 < risk2:
            return np.array([[-1]]*len(batch1))
        else:
            box_batch1 = []
            box_batch2 = []
            for one_batch in batch1:
                bb = [element[1] for element in one_batch[1]]
                box_batch1.append(bb)
            for one_batch in batch2:
                bb = [element[1] for element in one_batch[1]]
                box_batch2.append(bb)
            # calculate the numbers of boxes of batchs and compare them
            rel = []
            for i in range(len(box_batch1)):
                num_of_box1 = 0
                num_of_box2 = 0
                for element in box_batch1[i]:
                    num_of_box1 += len(element)
                for element in box_batch2[i]:
                    num_of_box2 += len(element)
                if num_of_box1 > num_of_box2:
                    rel.append([1])
                elif num_of_box1 < num_of_box2:     
                    rel.append([-1])
                else:
                    rel.append([0])
            return np.array(rel)
    #
    def test_model(self):
        """ Test
        """
        # evaluation
        self.evaluate('test')
    # 12102019
    def test_model_select(self, sel_data):
        """ Test
        """
        # evaluation
        if str(sel_data).upper() in train_data_group:
            if str(sel_data).upper() == train_data_group[0]:
                self.evaluate('test')
            elif str(sel_data).upper() == train_data_group[1]:
                # 11302019 - evaluation with virtual
                self.evaluate_virtual('test')
            else:
                self.evaluate_mix('test')
        else:
            print("Wrong data input")
    #
    def evaluate(self, stage):
        """ Evaluation
            Args:
             stage (str): a stage 'train' or 'test'
        """
        #print("stage: " + str(stage) + ", ds_length: " + str(self.train_ds_length) + ", ds_length (test): " + str(self.test_ds_length)) #testing - 20190213
        if stage == 'train':
            ds1 = self.train_ds1
            ds2 = self.train_ds2
            risk1 = self.train_risk1
            risk2 = self.train_risk2
            ds_length = self.train_ds_length
        elif stage == 'test':
            ds1 = self.test_ds1
            ds2 = self.test_ds2
            risk1 = self.test_risk1
            risk2 = self.test_risk2              
            ds_length = self.test_ds_length
        if ds1.ds_path != ds2.ds_path:
            different_ds = True
        else:
            different_ds = False
        # evaluate model(forward recurrent propagation and risk prediction) and count accurate prediction
        start_time = time.time()
        accurate_prediction = 0
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            for i in range(ds_length):
                if different_ds:
                    sid1 = i
                    sid2 = i
                else:
                    sid1 = i
                    sid2 = i+1
                    if sid2 == ds_length:
                        sid2 = 0
                sample1 = [ds1.get_example(sid1)]
                sample2 = [ds2.get_example(sid2)]
                input_feature_seq1 = ds1.prepare_input_sequence(sample1, self.roi_bg) # <ADD self.roi_bg/>
                input_feature_seq2 = ds2.prepare_input_sequence(sample2, self.roi_bg) # <ADD self.roi_bg/> 
                if self.risk_type == 'seq_risk':
                    r1 = self.model.predict_risk(input_feature_seq1)
                    r2 = self.model.predict_risk(input_feature_seq2)
                elif self.risk_type == 'seq_mean_risk':
                    r1 = self.model.predict_mean_risk(input_feature_seq1)
                    r2 = self.model.predict_mean_risk(input_feature_seq2)
                rel = self.compare_risk_level(sample1, sample2, risk1, risk2) # a numpy array each element of which is one of {[1], [-1], [0]}
                rd1 = cuda.to_cpu(r1.data)
                rd2 = cuda.to_cpu(r2.data)
                # count up the number of rel == 1 and r1 > r2
                accurate_prediction += len(np.where(np.logical_and(rel.flatten() == 1, rd1.flatten() > rd2.flatten()))[0])
                # count up the number of rel == -1 and r1 < r2
                accurate_prediction += len(np.where(np.logical_and(rel.flatten() == -1, rd1.flatten() < rd2.flatten()))[0])
                # count up the number of rel == 0 and abs(r1-r2) < self.threshold_of_similar_risk
                accurate_prediction += len(np.where(np.logical_and(rel.flatten() == 0, np.abs(rd1-rd2).flatten() < self.threshold_of_similar_risk))[0])
        end_time = time.time()
        # calculation of accuracy
        accuracy = 100.0 * accurate_prediction / ds_length
        print('  accuracy: {0:.3f}% [{1:,d}/{2:,d}] ({3:.2f} sec)'.format(accuracy, accurate_prediction, ds_length, end_time-start_time))
        if self.tlogf is not None:
            self.tlogf.write(' {0} data evaluation:\n'.format(stage))
            self.tlogf.write('  accuracy: {0:.3f}% [{1:,d}/{2:,d}] ({3:.2f} sec)\n'.format(accuracy, accurate_prediction, ds_length, end_time-start_time))
            self.tlogf.flush()

    # 12042019
    def evaluate_mix(self, stage):
        """ Evaluation
            Args:
             stage (str): a stage 'train' or 'test'
        """
        #print("stage: " + str(stage) + ", ds_length: " + str(self.train_ds_length) + ", ds_length (test): " + str(self.test_ds_length)) #testing - 20190213
        if stage == 'train':
            ds1 = datasets.ConcatenatedDataset(self.train_ds1, self.vtrain_ds1)
            ds2 = datasets.ConcatenatedDataset(self.train_ds2, self.vtrain_ds2)
            risk1 = self.vtrain_risk1 + self.vtrain_risk1
            risk2 = self.vtrain_risk2 + self.vtrain_risk2
            ds_length = self.train_ds_length + self.vtrain_ds_length
        elif stage == 'test':
            ds1 = self.test_ds1 
            ds2 = self.test_ds2 
            risk1 = self.test_risk1 
            risk2 = self.test_risk2 
            ds_length = self.test_ds_length 
        if ds1.ds_path != ds2.ds_path:
            different_ds = True
        else:
            different_ds = False
        # evaluate model(forward recurrent propagation and risk prediction) and count accurate prediction
        start_time = time.time()
        accurate_prediction = 0
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            for i in range(ds_length):
                if different_ds:
                    sid1 = i
                    sid2 = i
                else:
                    sid1 = i
                    sid2 = i+1
                    if sid2 == ds_length:
                        sid2 = 0
                sample1 = [ds1.get_example(sid1)]
                sample2 = [ds2.get_example(sid2)]
                input_feature_seq1 = ds1.prepare_input_sequence(sample1, self.roi_bg) # <ADD self.roi_bg/>
                input_feature_seq2 = ds2.prepare_input_sequence(sample2, self.roi_bg) # <ADD self.roi_bg/> 
                if self.risk_type == 'seq_risk':
                    r1 = self.model.predict_risk(input_feature_seq1)
                    r2 = self.model.predict_risk(input_feature_seq2)
                elif self.risk_type == 'seq_mean_risk':
                    r1 = self.model.predict_mean_risk(input_feature_seq1)
                    r2 = self.model.predict_mean_risk(input_feature_seq2)
                rel = self.compare_risk_level(sample1, sample2, risk1, risk2) # a numpy array each element of which is one of {[1], [-1], [0]}
                rd1 = cuda.to_cpu(r1.data)
                rd2 = cuda.to_cpu(r2.data)
                # count up the number of rel == 1 and r1 > r2
                accurate_prediction += len(np.where(np.logical_and(rel.flatten() == 1, rd1.flatten() > rd2.flatten()))[0])
                # count up the number of rel == -1 and r1 < r2
                accurate_prediction += len(np.where(np.logical_and(rel.flatten() == -1, rd1.flatten() < rd2.flatten()))[0])
                # count up the number of rel == 0 and abs(r1-r2) < self.threshold_of_similar_risk
                accurate_prediction += len(np.where(np.logical_and(rel.flatten() == 0, np.abs(rd1-rd2).flatten() < self.threshold_of_similar_risk))[0])
        end_time = time.time()
        # calculation of accuracy
        accuracy = 100.0 * accurate_prediction / ds_length
        print('  accuracy: {0:.3f}% [{1:,d}/{2:,d}] ({3:.2f} sec)'.format(accuracy, accurate_prediction, ds_length, end_time-start_time))
        if self.tlogf is not None:
            self.tlogf.write(' {0} data evaluation:\n'.format(stage))
            self.tlogf.write('  accuracy: {0:.3f}% [{1:,d}/{2:,d}] ({3:.2f} sec)\n'.format(accuracy, accurate_prediction, ds_length, end_time-start_time))
            self.tlogf.flush()    

    # 11302019
    def evaluate_virtual(self, stage):
        """ Evaluation
            Args:
             stage (str): a stage 'train' or 'test'
        """
        #print("stage: " + str(stage) + ", ds_length: " + str(self.train_ds_length) + ", ds_length (test): " + str(self.test_ds_length)) #testing - 20190213
        if stage == 'train':
            ds1 = self.vtrain_ds1
            ds2 = self.vtrain_ds2
            risk1 = self.vtrain_risk1
            risk2 = self.vtrain_risk2
            ds_length = self.vtrain_ds_length
        elif stage == 'test':
            ds1 = self.test_ds1 
            ds2 = self.test_ds2 
            risk1 = self.test_risk1 
            risk2 = self.test_risk2 
            ds_length = self.test_ds_length 
        if ds1.ds_path != ds2.ds_path:
            different_ds = True
        else:
            different_ds = False
        # evaluate model(forward recurrent propagation and risk prediction) and count accurate prediction
        start_time = time.time()
        accurate_prediction = 0
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            for i in range(ds_length):
                if different_ds:
                    sid1 = i
                    sid2 = i
                else:
                    sid1 = i
                    sid2 = i+1
                    if sid2 == ds_length:
                        sid2 = 0
                sample1 = [ds1.get_example(sid1)]
                sample2 = [ds2.get_example(sid2)]
                input_feature_seq1 = ds1.prepare_input_sequence(sample1, self.roi_bg) # <ADD self.roi_bg/>
                input_feature_seq2 = ds2.prepare_input_sequence(sample2, self.roi_bg) # <ADD self.roi_bg/> 
                if self.risk_type == 'seq_risk':
                    r1 = self.model.predict_risk(input_feature_seq1)
                    r2 = self.model.predict_risk(input_feature_seq2)
                elif self.risk_type == 'seq_mean_risk':
                    r1 = self.model.predict_mean_risk(input_feature_seq1)
                    r2 = self.model.predict_mean_risk(input_feature_seq2)
                rel = self.compare_risk_level(sample1, sample2, risk1, risk2) # a numpy array each element of which is one of {[1], [-1], [0]}
                rd1 = cuda.to_cpu(r1.data)
                rd2 = cuda.to_cpu(r2.data)
                # count up the number of rel == 1 and r1 > r2
                accurate_prediction += len(np.where(np.logical_and(rel.flatten() == 1, rd1.flatten() > rd2.flatten()))[0])
                # count up the number of rel == -1 and r1 < r2
                accurate_prediction += len(np.where(np.logical_and(rel.flatten() == -1, rd1.flatten() < rd2.flatten()))[0])
                # count up the number of rel == 0 and abs(r1-r2) < self.threshold_of_similar_risk
                accurate_prediction += len(np.where(np.logical_and(rel.flatten() == 0, np.abs(rd1-rd2).flatten() < self.threshold_of_similar_risk))[0])
        end_time = time.time()
        # calculation of accuracy
        accuracy = 100.0 * accurate_prediction / ds_length
        print('  accuracy: {0:.3f}% [{1:,d}/{2:,d}] ({3:.2f} sec)'.format(accuracy, accurate_prediction, ds_length, end_time-start_time))
        if self.tlogf is not None:
            self.tlogf.write(' {0} data evaluation:\n'.format(stage))
            self.tlogf.write('  accuracy: {0:.3f}% [{1:,d}/{2:,d}] ({3:.2f} sec)\n'.format(accuracy, accurate_prediction, ds_length, end_time-start_time))
            self.tlogf.flush()
    
#