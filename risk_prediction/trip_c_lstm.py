# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 17:21:36 2019

@author: setsu
@filename: trip_c_lstm.py
@coding: utf-8
========================
Date          Comment
========================
09142019      First revision
10252019      Enhancement code to improve result (>70% accuracy in risk prediction)
10262019      Reduce ksize, stride, pad for input_middle_conv 
10272019      Increase convolutional layer (input_sec_middle_conv)
10282019      Increase lstm (lstm3)
10292019      Increase convolutional layer (input_third_middle_conv)
11012019    　Remove max_pooling at extra input middle conv layer to prevent data loss, insert dropout in between middle conv layer
12182019      New model architecture function to test increasing accuracy
12242019      New model architecture function that reverts ReLu and Tanh
03252020      New model architecture (RReLU)
03292020      New model architecture (CReLU, ELU)
04032020      New variant model architecture (CReLU, ELU)
04132020      Implement zoneout on ReLU model architecture
04142020      Modify model architecture to reinsert TanH function
"""

import chainer.functions as F
import chainer.links as L
from risk_prediction.trip_lstm import TripLSTM
#from trip_lstm import TripLSTM

class TripCLSTM(TripLSTM):
    """A class of TRIP(Traffic Risk Prediction) model, which has a pooling layer on the input side
    """
    def __init__(self, input_size, hidden_size, model_arch='MP-C-SPP-FC-LSTM'):
        """ Constructor
            Args:
             input_size (int): an input size of LSTM
             hidden_size (int): a hidden size of LSTM
             model_arch (str): a model architecture
        """
        super(TripCLSTM, self).__init__(input_size, hidden_size)
        with self.init_scope():
            self.input_conv = L.Convolution2D(None, 512, ksize=3, stride=1, pad=1) 
#            self.input_middle_conv = L.Convolution2D(None, 1024, ksize=6, stride=2, pad=2) #10252019
            self.input_middle_conv = L.Convolution2D(None, 512, ksize=3, stride=1, pad=1) #10262019
            self.input_sec_middle_conv = L.Convolution2D(None, 512, ksize=3, stride=1, pad=1) #10272019
            self.input_third_middle_conv = L.Convolution2D(None, 512, ksize=3, stride=1, pad=1) #10292019
        self.model_arch = model_arch
    #
    def __call__(self, x):
        """ Forward propagation
            Args:
             x (a Variable of feature array): a feature array
            Returns:
             h (a Variable of hidden state array): a hidden state array
        """
        dropout_ratio = 0.2
        if self.model_arch == 'MP-C-SPP-FC-LSTM': # 69.565
            z = F.max_pooling_2d(x, 2) # ksize=2, stride=2
            z = F.tanh(self.input_conv(z)) # 512, ksize=3, stride=1. pad=1
            #z = F.spatial_pyramid_pooling_2d(z, 3, pooling_class=F.MaxPooling2D)
            z = F.spatial_pyramid_pooling_2d(z, 3, pooling="max")
            z = F.tanh(self.input(z))
        elif self.model_arch == 'MP-C-SPP-FC-DO-LSTM': # 68.944
            z = F.max_pooling_2d(x, 2)
            z = F.tanh(self.input_conv(z))
            #z = F.spatial_pyramid_pooling_2d(z, 3, pooling_class=F.MaxPooling2D)
            z = F.spatial_pyramid_pooling_2d(z, 3, pooling="max")
            z = F.tanh(self.input(z))
            z = F.dropout(z, ratio=dropout_ratio)
        # 12182019
        elif self.model_arch == 'MP-C-SPP4-FC-LSTM': # revert stride=2 as previously doesn't work - 12192019
            z = F.max_pooling_2d(x, 2) # ksize=2, stride=2
            z = F.tanh(self.input_conv(z)) # 512, ksize=3, stride=1. pad=1
            z = F.spatial_pyramid_pooling_2d(z, 4, pooling="max")
            z = F.tanh(self.input(z))
        elif self.model_arch == 'MP-C-SPP4-FC-DO-LSTM': # 68.944
            z = F.max_pooling_2d(x, 2) # ksize=2, stride=2
            z = F.tanh(self.input_conv(z))
            z = F.spatial_pyramid_pooling_2d(z, 4, pooling="max") #pyramid_height=4
            z = F.tanh(self.input(z))
            z = F.dropout(z, ratio=dropout_ratio)
        elif self.model_arch == 'MP-C-SPP4-RL-LSTM':
            z = F.max_pooling_2d(x, 2) # ksize=2, stride=2
            z = F.tanh(self.input_conv(z))
            z = F.spatial_pyramid_pooling_2d(z, 4, pooling="max") #pyramid_height=4
            z = F.relu(self.input(z))
        elif self.model_arch == 'MP-C-SPP4-RL-DO-LSTM':
            z = F.max_pooling_2d(x, 2) # ksize=2, stride=2
            z = F.tanh(self.input_conv(z))
            z = F.spatial_pyramid_pooling_2d(z, 4, pooling="max") #pyramid_height=4
            z = F.relu(self.input(z))
            z = F.dropout(z, ratio=dropout_ratio)
        elif self.model_arch == 'MP-C-SPP4-RL2-LSTM':
            z = F.max_pooling_2d(x, 2) # ksize=2, stride=2
            z = F.relu(self.input_conv(z))
            z = F.spatial_pyramid_pooling_2d(z, 4, pooling="max") #pyramid_height=4
            z = F.relu(self.input(z))
        elif self.model_arch == 'MP-C-SPP4-RL2-DO-LSTM':
            z = F.max_pooling_2d(x, 2) # ksize=2, stride=2
            z = F.relu(self.input_conv(z))
            z = F.spatial_pyramid_pooling_2d(z, 4, pooling="max") #pyramid_height=4
            z = F.relu(self.input(z))
            z = F.dropout(z, ratio=dropout_ratio)
        # end 12182019
        # 12242019
        elif self.model_arch == 'MP-C-RL-SPP4-LSTM': # Avg = 67.251 (highest), Highest acc = 68.580
            z = F.max_pooling_2d(x, 2) # ksize=2, stride=2
            z = F.relu(self.input_conv(z))
            z = F.spatial_pyramid_pooling_2d(z, 4, pooling="max") #pyramid_height=4
            z = F.tanh(self.input(z))
        elif self.model_arch == 'MP-C-RL-SPP4-DO-LSTM': # Avg = 67.613 (highest), Highest acc = 70.393
            z = F.max_pooling_2d(x, 2) # ksize=2, stride=2
            z = F.relu(self.input_conv(z))
            z = F.spatial_pyramid_pooling_2d(z, 4, pooling="max") #pyramid_height=4
            z = F.tanh(self.input(z))
            z = F.dropout(z, ratio=dropout_ratio)
        elif self.model_arch == 'MP-C-RL-SPP4-ZO-LSTM': # 04132020
            z = F.max_pooling_2d(x, 2) # ksize=2, stride=2
            z = F.relu(self.input_conv(z))
            z = F.spatial_pyramid_pooling_2d(z, 4, pooling="max") #pyramid_height=4
            previous_z = z
            z = F.tanh(self.input(z))
            z = F.zoneout(previous_z, z, ratio=dropout_ratio)
        # 04152020
        elif self.model_arch == 'MP-C-RL2-SPP4-LSTM': 
            z = F.max_pooling_2d(x, 2) # ksize=2, stride=2
            z = F.relu(self.input_conv(z))
            z = F.relu(self.input_conv(z))
            z = F.spatial_pyramid_pooling_2d(z, 4, pooling="max") #pyramid_height=4
            z = F.tanh(self.input(z))
        elif self.model_arch == 'MP-C-RL2-SPP4-DO-LSTM': 
            z = F.max_pooling_2d(x, 2) # ksize=2, stride=2
            z = F.relu(self.input_conv(z))
            z = F.relu(self.input_conv(z))
            z = F.spatial_pyramid_pooling_2d(z, 4, pooling="max") #pyramid_height=4
            z = F.tanh(self.input(z))        
            z = F.dropout(z, ratio=dropout_ratio)
        # end 04152020
        # end 12242019
        # 03252020
        elif self.model_arch == 'MP-C-RRL-SPP4-LSTM': # Avg = 66.515 (highest), Highest acc = 68.946
            z = F.max_pooling_2d(x, 2) # ksize=2, stride=2
            z = F.rrelu(self.input_conv(z))
            z = F.spatial_pyramid_pooling_2d(z, 4, pooling="max") #pyramid_height=4
            z = F.tanh(self.input(z))
        elif self.model_arch == 'MP-C-RRL-SPP4-DO-LSTM': # Avg = 65.641 (highest), Highest acc = 66.952
            z = F.max_pooling_2d(x, 2) # ksize=2, stride=2
            z = F.rrelu(self.input_conv(z))
            z = F.spatial_pyramid_pooling_2d(z, 4, pooling="max") #pyramid_height=4
            z = F.tanh(self.input(z))
            z = F.dropout(z, ratio=dropout_ratio)
        elif self.model_arch == 'MP-C-RRL-SPP4-ZO-LSTM':
            z = F.max_pooling_2d(x, 2) # ksize=2, stride=2
            z = F.rrelu(self.input_conv(z))
            z = F.spatial_pyramid_pooling_2d(z, 4, pooling="max") #pyramid_height=4
            previous_z = z
            z = F.tanh(self.input(z))
            z = F.zoneout(previous_z, z, ratio=dropout_ratio)
        # 04152020
        elif self.model_arch == 'MP-C-RRL2-SPP4-LSTM': 
            z = F.max_pooling_2d(x, 2) # ksize=2, stride=2
            z = F.rrelu(self.input_conv(z))
            z = F.rrelu(self.input_conv(z))
            z = F.spatial_pyramid_pooling_2d(z, 4, pooling="max") #pyramid_height=4
            z = F.tanh(self.input(z))
        elif self.model_arch == 'MP-C-RRL2-SPP4-DO-LSTM': 
            z = F.max_pooling_2d(x, 2) # ksize=2, stride=2
            z = F.rrelu(self.input_conv(z))
            z = F.rrelu(self.input_conv(z))
            z = F.spatial_pyramid_pooling_2d(z, 4, pooling="max") #pyramid_height=4
            z = F.tanh(self.input(z))        
            z = F.dropout(z, ratio=dropout_ratio)
        # end 04152020    
        # end 03252020
        # 04032020
        elif self.model_arch == 'MP-C-RRL-SPP4-RRL-LSTM':
            z = F.max_pooling_2d(x, 2) # ksize=2, stride=2
            z = F.rrelu(self.input_conv(z))
            z = F.spatial_pyramid_pooling_2d(z, 4, pooling="max") #pyramid_height=4
            z = F.rrelu(self.input(z))
        elif self.model_arch == 'MP-C-RRL-SPP4-RRL-DO-LSTM':
            z = F.max_pooling_2d(x, 2) # ksize=2, stride=2
            z = F.rrelu(self.input_conv(z))
            z = F.spatial_pyramid_pooling_2d(z, 4, pooling="max") #pyramid_height=4
            z = F.rrelu(self.input(z))
            z = F.dropout(z, ratio=dropout_ratio)
        # 04032020
        # 03292020
        elif self.model_arch == 'MP-C-CRL-SPP4-LSTM':
            z = F.max_pooling_2d(x, 2) # ksize=2, stride=2
            z = F.crelu(self.input_conv(z), axis=1)
            z = F.spatial_pyramid_pooling_2d(z, 4, pooling="max") #pyramid_height=4
            z = F.tanh(self.input(z))
        elif self.model_arch == 'MP-C-CRL-SPP4-DO-LSTM':
            z = F.max_pooling_2d(x, 2) # ksize=2, stride=2
            z = F.crelu(self.input_conv(z), axis=1)
            z = F.spatial_pyramid_pooling_2d(z, 4, pooling="max") #pyramid_height=4
            z = F.tanh(self.input(z))
            z = F.dropout(z, ratio=dropout_ratio)
        elif self.model_arch == 'MP-C-EL-SPP4-LSTM': # Avg = 66.382 (highest), Highest acc = 68.376
            z = F.max_pooling_2d(x, 2) # ksize=2, stride=2
            z = F.elu(self.input_conv(z), alpha=1.0)
            z = F.spatial_pyramid_pooling_2d(z, 4, pooling="max") #pyramid_height=4
            z = F.tanh(self.input(z))
        elif self.model_arch == 'MP-C-EL-SPP4-DO-LSTM': # Avg = 66.325 (highest), Highest acc = 68.091
            z = F.max_pooling_2d(x, 2) # ksize=2, stride=2
            z = F.elu(self.input_conv(z), alpha=1.0)
            z = F.spatial_pyramid_pooling_2d(z, 4, pooling="max") #pyramid_height=4
            z = F.tanh(self.input(z))
            z = F.dropout(z, ratio=dropout_ratio)
        # 03292020
        # 04032020
        elif self.model_arch == 'MP-C-EL-SPP4-EL-LSTM':
            z = F.max_pooling_2d(x, 2) # ksize=2, stride=2
            z = F.elu(self.input_conv(z), alpha=1.0)
            z = F.spatial_pyramid_pooling_2d(z, 4, pooling="max") #pyramid_height=4
            z = F.elu(self.input(z), alpha=1.0)
        elif self.model_arch == 'MP-C-EL-SPP4-EL-DO-LSTM':
            z = F.max_pooling_2d(x, 2) # ksize=2, stride=2
            z = F.elu(self.input_conv(z), alpha=1.0)
            z = F.spatial_pyramid_pooling_2d(z, 4, pooling="max") #pyramid_height=4
            z = F.elu(self.input(z), alpha=1.0)
            z = F.dropout(z, ratio=dropout_ratio)
        # 04032020
        elif self.model_arch == 'DO-MP-C-SPP-FC-LSTM':
            z = F.dropout(x, ratio=dropout_ratio)
            z = F.max_pooling_2d(z, 2)
            z = F.tanh(self.input_conv(z))
            #z = F.spatial_pyramid_pooling_2d(z, 3, pooling_class=F.MaxPooling2D)
            z = F.spatial_pyramid_pooling_2d(z, 3, pooling="max")
            z = F.tanh(self.input(z))
        elif self.model_arch == 'MP-C-SPP-FC-DO-LSTM2' or self.model_arch == 'MP-C-SPP-FC-DO-LSTM3': # 10252019, 10282019
            #z = F.max_pooling_2d(x, 2) #11012019
            z = F.tanh(self.input_conv(x))
            z = F.dropout(z, ratio=dropout_ratio) #11012019
            #z = F.max_pooling_2d(z, 2) #11012019
            z = F.tanh(self.input_middle_conv(z))
            #z = F.max_pooling_2d(z, 2)  # 10272019, 11012019
            z = F.dropout(z, ratio=dropout_ratio) #11012019
            z = F.tanh(self.input_sec_middle_conv(z))   # 10272019  
            #z = F.max_pooling_2d(z, 2) # 10292019, 11012019
            z = F.dropout(z, ratio=dropout_ratio) #11012019
            z = F.tanh(self.input_third_middle_conv(z)) # 10292019
            #z = F.spatial_pyramid_pooling_2d(z, 3, pooling_class=F.MaxPooling2D)
            z = F.spatial_pyramid_pooling_2d(z, 3, pooling="max")
            z = F.tanh(self.input(z))
            z = F.dropout(z, ratio=dropout_ratio)
        elif self.model_arch == 'MP-C-SPP-FC-LSTM2' or self.model_arch == 'MP-C-SPP-FC-LSTM3': # 10252019, 10282019
            #z = F.max_pooling_2d(x, 2) # ksize=2, stride=2
            z = F.tanh(self.input_conv(x)) # 512, ksize=3, stride=1. pad=1
            z = F.dropout(z, ratio=dropout_ratio) #11012019
            #z = F.max_pooling_2d(x, 2) #11012019
            z = F.tanh(self.input_middle_conv(z))
            #z = F.max_pooling_2d(z, 2)  # 10272019
            z = F.dropout(z, ratio=dropout_ratio) #11012019
            z = F.tanh(self.input_sec_middle_conv(z))   # 10272019                        
            #z = F..max_pooling_2d(z, 2) # 10292019
            z = F.dropout(z, ratio=dropout_ratio) #11012019
            z = F.tanh(self.input_third_middle_conv(z)) # 10292019
            #z = F.spatial_pyramid_pooling_2d(z, 3, pooling_class=F.MaxPooling2D)
            z = F.spatial_pyramid_pooling_2d(z, 3, pooling="max")
            z = F.tanh(self.input(z))
        
        if self.model_arch == 'MP-C-SPP-FC-DO-LSTM2' or self.model_arch == 'MP-C-SPP-FC-LSTM2': #10252019
            return self.lstm2(self.lstm(z))
        elif self.model_arch == 'MP-C-SPP-FC-DO-LSTM3' or self.model_arch == 'MP-C-SPP-FC-LSTM3': #10282019
            return self.lstm3(self.lstm2(self.lstm(z)))
        else:
            return self.lstm(z)
#        return self.lstm(z)
