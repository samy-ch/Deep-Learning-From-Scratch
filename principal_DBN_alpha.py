# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 15:43:19 2019

@author: Admin
"""

import numpy as np
import principal_RBM_alpha

def softmax(x):
    y = np.exp(x)
    if x.ndim == 1:
        s = y.sum()
    else:
        m = x.shape[1]
        s = y.sum(axis=1)
        s = np.tile(s, (m,1))
    return y/s.T

class DBN(object):
    def __init__(self, neurons_per_layer):
        self.neurons_per_layer = neurons_per_layer
        self.number_layers = len(neurons_per_layer)
        self.structure = []
        self.number_RBM = self.number_layers-1
        for i in range(self.number_layers-1):
            self.structure.append(principal_RBM_alpha.RBM(visible_size=neurons_per_layer[i], hidden_size=neurons_per_layer[i+1]))
        
    def getWeights(self, layer):
        return self.structure[layer].getWeights()
    
    def getBias(self, layer):
        return self.structure[layer].getVisibleBias()
    
    def getStructure(self):
        return self.structure
    
    def train_DBN(self, data, nb_iter_grad=100, learning_rate=0.1, batch_size=1):
        '''
        function train_DBN:
            parameters:
                -data: array of data  which the DBN will learn the distribution of
                -nb_iter_grad: number of iterations of the gradient descent algorithm (default: 100)
                -learning_rate: learning rate for the gradient descent (default: 0.1)
                -batch_size: number of data in a batch for the gradient descent mini-batch (default: 1)
            returns:
                -trained DBN instance with the greedy layer-wise procedure
        '''
        data_copy = np.copy(data)
        for rbm in self.structure:
            rbm.train_RBM(self, data=data_copy, nb_iter=nb_iter_grad, batch_size=batch_size, learning_rate=learning_rate)
            data_copy = rbm.input_output_RBM(data_copy)
        return self
    
    def generate_image(self, nb_gibbs_iter):
        v, h = self.structure[-1].generate_image(nb_gibbs_iter=nb_gibbs_iter)
        for rbm in reversed(self.structure[:-1]):
            v = rbm.output_input_RBM(v)[0]
#            v = np.random.binomial(n=1, p=w)
        return (v > 0.5).astype(float)
    
    def output_input_network(self, data):
        '''
        function output_input_network:
            parameters:
                -data: array in input
            returns:
                -a list with the output of each layer (as well as the data in input).
                Th output of the last layer are probabilities
        '''
        new_data = np.copy(data)
        table = [new_data]
        for i in range(self.number_RBM  - 1):
            rbm = self.structure[i]
            if data.ndim == 1:
                ouput_data = rbm.input_output_RBM(new_data)[0]
            else:
                ouput_data = rbm.input_output_RBM(new_data)
            table.append(ouput_data)
            new_data = ouput_data
            
        probs = self.structure[-1].compute_softmax(new_data)
        table.append(probs)
        
        return table
    
    
    def backpropagation(self, data, labels, nb_iter_grad=100, learning_rate=0.1, batch_size=1):
        '''
        function backpropagation:
            parameters:
                -data: data for the training (array)
                -labels: labeels fr the training (in the same order as the data)
                -nb_iter_grad: number of iterations of the gradient descent algorithm (default 100)
                -learning_rate: learning rate for the grdient descent algorithm (default 0.1)
                -batch_size: size of the batch to use during training (default 1)
            returns:
                -trained feed forward neural network with regular vanilla gradient descent (with mini-batch if batch_size more than 1)
                -entropies: list of the cross entropy computed at the end of one iteration over a batch (for each iteration of the gradient descent) to check if the model learns properly
        '''
        entropies = []
        training_labels = np.copy(labels)
        training_data = np.copy(data)
        for iteration in range(nb_iter_grad):
            
           n=len(data)
           permutation = np.arange(0, n)
           permutation = np.random.permutation(permutation)
           training_data = training_data[permutation]
           training_labels = training_labels[permutation]
           old_dnn_W = [None]*self.number_RBM 
    
           for batch in range(0, n, batch_size):
               
               for l in range(self.number_RBM ):
                   old_dnn_W[l] = np.copy(self.structure[l].weights)
                   
               new_data = training_data[batch:min(n, batch_size+batch)]
    
               table = self.output_input_network(new_data)
    
               c = table[-1] - training_labels[batch:min(n, batch_size+batch)]
               grad_E_W = table[-2].T @ c
               grad_E_b = np.sum(c, axis=0)
               
               
               self.structure[-1].weights +=  -(learning_rate/batch_size)*grad_E_W
               self.structure[-1].hidden_bias +=  -(learning_rate/batch_size)*grad_E_b
                        
               for l in reversed(range(1, self.number_layers-1)):
                   
                   x = np.multiply(table[l], (1-table[l]))
                   tmp = c.dot(old_dnn_W[l].T)
                   c = np.multiply(tmp, x)
    
                   grad_E_W = table[l-1].T @ c
                   grad_E_b = np.sum(c, axis=0)
                    
                   self.structure[l-1].weights += -(learning_rate/batch_size)*grad_E_W
                   self.structure[l-1].hidden_bias += -(learning_rate/batch_size)*grad_E_b
                
               estimations = self.output_input_network(training_data)[-1]
               cross_entropy = - np.sum(np.sum(np.multiply(training_labels, np.log(estimations)),axis=1))/batch_size
               entropies.append(cross_entropy)
       
        return entropies
    
    
    def test_DNN(self, data, labels):
        '''
        function test_DNN:
            parameters:
                -data: array of data
                -labels: labels associated with the labels
            returns:
                -rate of good classificaion over the data in input of the model on which the method is called
        '''
        probs = self.output_input_network(data)[-1]
        estimated_index = np.argmax(probs, axis=1)
        
        true_index = np.argmax(labels, axis=1)
        
        mistakes = np.equal(estimated_index, true_index).astype(float)
        tau = mistakes.sum()/len(data)
        
        return tau
