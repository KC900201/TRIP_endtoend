# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 16:22:40 2019

@author: setsu
@filename: trip_lstm.py
@coding: utf-8
========================
Date          Comment
========================
09142019      First revision
10252019      Enhancement code to improve result (>70% accuracy in risk prediction)
10282019      Increase one more lstm (lstm3)
03022020      Predict maximum risk function in between n frames 
03092020      Predict maximum risk method for risk prediction experiment
"""
from chainer import Chain, cuda, Variable
import chainer.links as L
import chainer.functions as F
import numpy as np
import cupy as cp
#import numpy

class TripLSTM(Chain):
    """A class of TRIP(Traffic Risk Prediction) model
    """
    def __init__(self, input_size, hidden_size):
        """ Constructor
            Args:
             input_size (int): an input size of LSTM
             hidden_size (int): a hidden size of LSTM
        """
        super(TripLSTM, self).__init__()
        with self.init_scope():
            self.input = L.Linear(None, input_size)
            self.lstm = L.LSTM(input_size, hidden_size)
            self.lstm2 = L.LSTM(hidden_size, hidden_size) # 10252019
            self.lstm3 = L.LSTM(hidden_size, hidden_size) # 10282019
            self.ho = L.Linear(hidden_size, 1)
    #
    def __call__(self, x):
        """ Forward propagation
            Args:
             x (a Variable of feature array): a feature array
            Returns:
             h (a Variable of hidden state array): a hidden state array
        """
        z = F.tanh(self.input(x))
        return self.lstm(z)
    #
    def predict_risk(self, x):
        """ Risk prediction
            Args:
             x (a list of feature array): a feature array list
            Returns:
             r (a Variable of float): a risk value
        """
        # reset lstm state
        self.lstm.reset_state()
        # recurrent reasoning and risk prediction
        for t in range(len(x)):
            v = Variable(self.xp.array(x[t], dtype=self.xp.float32))
            h = self(v)
        return F.sigmoid(self.ho(h))
    #
    def predict_mean_risk(self, x):
        """ Risk prediction (mean)
            Args:
             x (a list of feature array): a feature array list
            Returns:
             r (a Variavle of float): a risk value
        """
        # reset lstm state
        self.lstm.reset_state()
        # recurrent reasoning and risk prediction
        mr = 0
        for t in range(len(x)):
            v = Variable(self.xp.array(x[t], dtype=self.xp.float32))
            h = self(v)
            r = F.sigmoid(self.ho(h))
            mr += r
        return mr/len(x)
    # Max risk value - 03022020
    def predict_max_risk(self, x):
        """ Risk prediction (mean)
            Args:
             x (a list of feature array): a feature array list
            Returns:
             r (a Variavle of float): a risk value
        """
        # reset lstm state
        self.lstm.reset_state()
        # recurrent reasoning and risk prediction
        max_r = 0
        for t in range(len(x)):
            v = Variable(self.xp.array(x[t], dtype=self.xp.float32))
            h = self(v)
            r = F.sigmoid(self.ho(h))
            if t == 0:
                max_r = r
            else:
                if r.data > max_r.data:
                    max_r = r
                else:
                    max_r = max_r
        return max_r
    # 03092020
    def predict_max_risk_2(self, x):
        """ Risk prediction (mean)
            Args:
             x (a list of feature array): a feature array list
            Returns:
             r (a Variavle of float): a risk value
        """
        # reset lstm state
        self.lstm.reset_state()
        # recurrent reasoning and risk prediction
        max_r = 0
        for t in range(len(x)):
            v = Variable(self.xp.array(x[t], dtype=self.xp.float32))
            h = self(v)
            r = F.sigmoid(self.ho(h))
            if t == 0:
                max_r = r
            else:
                max_r = Variable(cp.maximum(r.data, max_r.data))
        return max_r
    #
    def comparative_loss(self, ra, rc, rel, margin=0.05):
        """ Comparative loss function
            Args:
             ra (a Variable of float array): anchor risk (minibatch size)
             rc (a Variable of float array): comparison risk (minibatch size)
             rel (a numpy array of int {[1], [-1], [0]} array: if 'ra' element must be greater than 'rc' element, 'rel' element is [1]. 
                                                               if 'ra' element must be less than 'rc' element, 'rel' element is [-1].
                                                               if 'ra' element and 'rc' element must be equal, 'rel' element is [0].
             margin (float): margin of risk (a small value(<1.0))
            Returns:
             loss (Variable of float array): comparative loss
        """
        rel = self.xp.array(rel)
        zero = Variable(self.xp.array([[0.]]*len(ra.data), dtype=self.xp.float32))
        cl = F.where(rel > 0, F.maximum(rc-ra+margin, zero), F.where(rel < 0, F.maximum(ra-rc+margin, zero), F.absolute(rc - ra)))
        loss = F.sum(F.square(cl/2))
        return loss

