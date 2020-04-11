# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:55:11 2019

@author: Admin
"""


import numpy as np
from scipy.special import expit

def softmax(x):
    y = np.exp(x)
    if x.ndim == 1:
        s = y.sum()
    else:
        m = x.shape[1]
        s = y.sum(axis=1)
        s = np.tile(s, (m,1))
    return y/s.T

def read_alpha_digit(characters, data):
    """
    function read_alpha_digit:
        parameters:
            -data: dataset of images (20*16 pixels) from which the unsupervised models will learn
            -characters: list of characters to extract from the dataset
        returns:
            -numpy array containing the images of all the characters passed in argument
    """
    list_images = []
    list_labels = []
    for c in characters:
        if c.isdigit():
            for i in range(len(data["dat"][int(c)])):
                list_images.append(np.expand_dims(data["dat"][int(c)][i].flatten(), axis=0))
                
                new_label = np.zeros(36)
                new_label[int(c)] = 1
                list_labels.append(new_label)
            
        if c.isalpha():
            for i in range(len(data["dat"][ord(c)-55])):
                list_images.append(np.expand_dims(data["dat"][ord(c)-55][i].flatten(), axis=0))
                
                new_label = np.zeros(36)
                new_label[ord(c)-55] = 1
                list_labels.append(new_label)

    return np.concatenate(list_images), np.array(list_labels)
    
class RBM(object):
    def __init__(self, visible_size, hidden_size):
        self.visible_size = visible_size
        self.hidden_size = hidden_size  
        self.visible_bias = np.zeros(visible_size)
        self.hidden_bias = np.zeros(hidden_size)
        self.weights = np.random.normal(0, np.sqrt(0.1), (visible_size, hidden_size))
        

    def getWeights(self):
        return self.weights
    
    def getVisibleBias(self):
        return self.visible_bias    
    
    def setVisibleBias(self, visible_bias):
        self.visible_bias = visible_bias
        return self
    
    def setHiddenBias(self, hidden_bias):
        self.hidden_bias = hidden_bias
        return self
    
    def getHiddenBias(self):
        return self.hidden_bias
    
    def getVisibleSize(self):
        return self.visible_size
    
    def getHiddenSize(self):
        return self.hidden_size
    
        
    def input_output_RBM(self, data):
        """
        function input_output_RBM:
            parameters:
                -data: an array of data (the input data can be in the form of a batch)
            returns:
                -probabilities of each hidden node to be equal to 1 (array of arrays stacked n times)
        """
        n=data.shape[0]
        out_data = expit(data.dot(self.weights) + np.tile(self.hidden_bias, (n, 1)))
        return out_data
    
    def output_input_RBM(self, data):
        """
        function output_input_RBM:
            parameters:
                -data: an array of data (the input data can be in the form of a batch)
            returns:
                -probabilities of each visible node to be equal to 1 (array of arrays stacked n times)
        """
        n=data.shape[0]
        in_data = expit(data.dot(self.weights.T) + np.tile(self.visible_bias, (n, 1)))
        return in_data
        
    
    def train_RBM(self, data, nb_iter=100, batch_size=1, learning_rate=0.1):
        '''function train_RBM:
            parameters:
                -data: array of data (numpy array preferentially)
                -nb_iter: number of iterations for the gradient descent algorithm (default: 100)
                -batch_size: size of the batch of data to use in the mini-batch gradient descent (default: 1 (regular gradient descent))
                -learning_rate: learning rate for the gradient descent (default: 0.1)
            returns:
                -trains the RBM object on which is called the function (trained with vanilla gradient descent with mini-batch)
                -returns the mean square error of the model at the end of the training
        '''
        n = data.shape[0]
        data_copy = np.copy(data)
        for i in range(nb_iter):
            new_data = np.random.permutation(data_copy)
            for batch in range(0, n, batch_size):
                x = new_data[batch:min(n, batch_size+batch)]
                v_0 = x
                m = len(x)
                
                p_h_v0 = self.input_output_RBM(v_0)
                h_0 = np.random.binomial(n=1, p=p_h_v0, size=(m, self.hidden_size))
                
                p_v_h0 = self.output_input_RBM(h_0)
                v1 = np.random.binomial(n=1, p=p_v_h0, size=(m, self.visible_size))
                
                p_h_v1 = self.input_output_RBM(v1)
                
                da = np.sum(x - v1, axis=0)
                db = np.sum(p_h_v0 - p_h_v1, axis=0)
                pos = x.T.dot(p_h_v0)
                neg = v1.T.dot(p_h_v1)
                dW = pos - neg
                
                self.weights += (learning_rate/batch_size)*dW
                self.visible_bias += (learning_rate/batch_size)*da
                self.hidden_bias += (learning_rate/batch_size)*db
            RBM_data = self.input_output_RBM(data)
            new_input = self.output_input_RBM(RBM_data)
            MSE = np.mean((new_input - data)**2)
            
        return MSE


    def generate_image(self, nb_gibbs_iter):
        """
        function generate_image:
            parameters:
                -nb_gibbs_iter: number of iterations for the Gibbs sampling step
            returns:
                -visible states (v) following the distribution induced by the RBM's parameters
                -hidden states (h) following the distribution induced by the RBM's parameters
        """            
        v = np.random.binomial(n=1, p=0.5, size=self.visible_size)

        for t in range(nb_gibbs_iter):
            
            p_h_v = self.input_output_RBM(v)[0]
            h = np.random.binomial(n=1, p=p_h_v)
            
            p_v_h = self.output_input_RBM(h)[0]
            v = np.random.binomial(n=1, p=p_v_h)
        return v, h
    

    def compute_softmax(self, data):
        '''
        function compute_softmax:
            parameters:
                -array of data
            returns:
                -computes the probabilities of the hidden nodes to be activated from the data in input
        '''
        n=len(data)
        output = data.dot(self.weights) + np.tile(self.hidden_bias, (n, 1))
        probs = softmax(output)
        return probs