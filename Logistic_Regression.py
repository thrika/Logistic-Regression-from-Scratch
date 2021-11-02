#!/usr/bin/env python
# coding: utf-8

# In[1]:


from csv import reader
import os
import csv
import numpy as np
import math
import operator
from math import exp


# In[2]:


def load_file(filename):
    file = open(filename, "r")
    lines = reader(file)
    data = list(lines)
    vector = ''
    vector_array = []
    for row in range(len(data)):
        vector += data[row][0]
    for i in range(len(vector)):
        vector_array.append(int(vector[i]))
    return vector_array


# In[3]:


def load_class_data(filename, label, path):
    files = os.listdir(path)
    class_data = []
    for file in files:
        if file.startswith(filename):
            class_data.append([load_file(path + '\\' + file), label])
    return class_data


# In[4]:


class_0 = load_class_data('class_0', 0, r'dataset1/training_validation')
class_6 = load_class_data('class_6', 1, r'dataset1/training_validation')


# In[5]:


dataset = class_0 + class_6


# In[6]:


def predict(instance, coefficients):
    yhat = coefficients[0]
    for i in range(len(instance[0])):
        yhat += coefficients[i + 1] * instance[0][i]
    return 1.0 / (1.0 + exp(-yhat))


# In[9]:


def coefficients_sgd(train, learning_rate, epochs):
    coef = [0.0 for i in range(len(train[0][0])+1)]
    for epoch in range(epochs):
        sum_error = 0
        for instance in train:
            yhat = predict(instance, coef)
            error = instance[1] - yhat
            sum_error += error**2
            coef[0] = coef[0] + learning_rate * error * yhat * (1.0 - yhat)
            for i in range(len(instance[0])):
                coef[i + 1] = coef[i + 1] + learning_rate * error * yhat * (1.0 - yhat) * instance[0][i]
        print('>epoch=%d, learning_rate=%.3f, error=%.3f' % (epoch, learning_rate, sum_error))
    return coef


# In[10]:


coefs = coefficients_sgd(dataset, 0.3, 10)


# In[11]:


def get_all_predictions(testing_dataset, coefs):
    prediction_array = []
    for instance in testing_dataset:
        predictions = predict(instance, coefs)
        if (predictions<0.5):
            predictions = 0
        elif (predictions>=0.5):
            predictions = 1
        prediction_array.append([predictions, instance[1]])
    return prediction_array


# In[12]:


classt_0 = load_class_data('class_0', 0, r'dataset1/test')
classt_6 = load_class_data('class_6', 1, r'dataset1/test')
testing_dataset = classt_0 + classt_6


# In[13]:


prediction_array = get_all_predictions(testing_dataset, coefs)


# In[14]:


def get_accuracy(array):
    correct = 0
    for i in range(len(array)):
        if array[i][0] == array[i][1]:
            correct += 1
    return correct / float(len(array)) * 100.0


# In[15]:


get_accuracy(prediction_array)


# In[ ]:




