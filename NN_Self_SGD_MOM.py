# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 10:14:44 2021

@author: dssrk
"""

import numpy as np

def softmax(x):
    """
    Calculate softmax
    """
    smax=np.float32(np.zeros_like(x))
    l = x.shape[0]
    w=x.shape[1]
    for i in range(l):
        for j in range(w):
            smax[i][j] = np.exp(x[i][j])/np.sum(np.exp(x[i]))
    return smax


import hashlib
import os
import pickle
from urllib.request import urlretrieve
import tensorflow as tf
from numba import vectorize
from numba import jit,njit

import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from timeit import default_timer as timer


import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import resample
from tqdm import tqdm
from zipfile import ZipFile


print('All modules imported.')
def uncompress_features_labels(file):
    """
    Uncompress features and labels from a zip file
    :param file: The zip file to extract the data from
    """
    features = []
    labels = []

    with ZipFile(file) as zipf:
        # Progress Bar
        filenames_pbar = tqdm(zipf.namelist(), unit='files')
        
        # Get features and labels from all files
        for filename in filenames_pbar:
            # Check if the file is a directory
            if not filename.endswith('/'):
                with zipf.open(filename) as image_file:
                    image = Image.open(image_file)
                    image.load()
                    # Load image data as 1 dimensional array
                    # We're using float32 to save on memory space
                    feature = np.array(image, dtype=np.float32).flatten()
                # Get the the letter from the filename.  This is the letter of the image.
                label = os.path.split(filename)[1][0]
                features.append(feature)
                labels.append(label)
    return np.array(features), np.array(labels)

train_features, train_labels = uncompress_features_labels('notMNIST_train.zip')
test_features, test_labels = uncompress_features_labels('notMNIST_test.zip')
# Limit the amount of data to work with a docker container
docker_size_limit = 150000
train_features, train_labels = resample(train_features, train_labels, n_samples=docker_size_limit)

def normalize_grayscale(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    mn = 0.1
    mx = 0.9
    image_norm = ((image_data-np.min(image_data))/(np.max(image_data)-np.min(image_data)))
    
    image_n = mn + (mx-mn)*image_norm
    return image_n

train_features = normalize_grayscale(train_features)
test_features = normalize_grayscale(test_features)

encoder = LabelBinarizer()
encoder.fit(train_labels)




train_labels = encoder.transform(train_labels)
test_labels = encoder.transform(test_labels)
train_features, valid_features, train_labels, valid_labels = train_test_split(
    train_features,
    train_labels,
    test_size=0.05,
    random_state=832289)

print('work started')
# Network size
N_input = 28*28
N_labels = 10
learning_rate = 0.005

def batches(features_f,labels_f,batch_size):
    output_batches = []
    
    sample_size = len(features_f)
    for start_i in range(0, sample_size, batch_size):
        end_i = start_i + batch_size
        batch = [features_f[start_i:end_i], labels_f[start_i:end_i]]
        output_batches.append(batch)
        
    return output_batches


def input_output(weights,labels_f,features_f,biases_f):

    
    hidden_layer_in = np.dot(features_f,weights)+biases_f
    
    hidden_layer_out = softmax(hidden_layer_in)
    
    cross_entropy = - (1/features_f.shape[0])*np.sum(np.dot(labels_f.transpose(),np.log(hidden_layer_out)))

    return hidden_layer_out,cross_entropy

def accuracy_f(hidden_layer_out_f,labels_f):
    # Determine if the predictions are correct
    is_correct_prediction = np.equal(np.argmax(hidden_layer_out_f, 1), np.argmax(labels_f, 1))
    # Calculate the accuracy of the predictions
    accuracy_fun = np.mean(is_correct_prediction)

    return accuracy_fun

def updated_weights(momentum,delta_prev,hidden_layer_out_f,features_f,labels_f,weights_f,biases_f,learning_rate):

    delta_curr = (hidden_layer_out_f-labels_f)
    delta = momentum*delta_prev - learning_rate*delta_curr
    del_weights = 1/features_f.shape[0]*np.dot(features_f.transpose(),delta)
    weights_input_to_output = weights_f + (del_weights)
    biases = biases_f + np.mean(delta,axis=0)
    
    return weights_input_to_output, biases, delta




#optimizer
def optimizer(momentum,hidden_layer_out,features_f,labels_f,weights_input_to_output,biases,learning_rate):
    accuracy = 0
    cross_entropy_1 = []

    while accuracy < 1:
        # if np.remainder(count,10)==0:
        #     print(accuracy)
        if len(cross_entropy_1)==0:
            delta_prev = 0
        if len(cross_entropy_1)>1:
            # print(abs(cross_entropy_1[-2] - cross_entropy_1[-1]))
            if abs(cross_entropy_1[-2] - cross_entropy_1[-1])<0.001:
                print('chakde',accuracy)
                break
        weights_input_to_output,biases,delta_prev = updated_weights(momentum,delta_prev,hidden_layer_out, features_f, labels_f, weights_input_to_output, biases, learning_rate)
        hidden_layer_out,cross_entropy= input_output(weights_input_to_output, labels_f, features_f,biases)
        accuracy = accuracy_f(hidden_layer_out,labels_f)
        cross_entropy_1.append(cross_entropy)
    return weights_input_to_output,biases,accuracy,cross_entropy

# optimizer_jit = jit(nopython=False)(optimizer)

#%%


#SGD starts

#random weights and biases




weights_input_to_output = np.random.normal(0, scale=0.1, size=(N_input, N_labels))
biases = np.zeros(N_labels)
momentum = 0.9
learning_rate = 0.0001
log_batch_step = 50
loss_batch = []
train_acc_batch = []
valid_acc_batch = []

batches_r = batches(train_features,train_labels,log_batch_step)
count = 0
for batch in batches_r:
    # print(count)
    batch_features = batch[0]
    batch_labels = batch[1]
    batch_output,batch_cs = input_output(weights_input_to_output,batch_labels,batch_features,biases)
    weights_input_to_output,biases,acc,batch_cs = optimizer(momentum,batch_output,batch_features,batch_labels,weights_input_to_output,biases,learning_rate)
    loss_batch.append(batch_cs)
    train_acc_batch.append(acc)
    count = count + 1
    print(count)

# test features

hidden_layer_out,cross_entropy = input_output(weights_input_to_output, test_labels, test_features,biases)    
accuracy = accuracy_f(hidden_layer_out,test_labels)

print(accuracy)

# print('Hidden-layer Output:')
# print(hidden_layer_out)

# output_layer_in = np.dot(hidden_layer_out,weights_hidden_to_output)
# y_hat = sigmoid(output_layer_in)

# print('Output-layer Output:')
# print(y_hat)

