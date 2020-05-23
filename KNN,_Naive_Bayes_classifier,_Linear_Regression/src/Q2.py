#!/usr/bin/env python
# coding: utf-8

# ## Initial Imports

# In[1]:


import numpy as np
import pandas as pd
import ipdb
import pprint
import sys
import matplotlib.pyplot as plt
import math
import operator
from tabulate import tabulate

eps = np.finfo(float).eps
from numpy import log2 as log


# ## Reading Data

# In[13]:


data = pd.read_csv("../input_data/LoanDataset/data.csv", header=None)
data.columns = ['ID', 'Age', 'Num_years_exp', 'Income', 'Zipcode', 'Family_size', 'Avg_spending_pm', 'Education_level', 'Mortgage_value', 'Loan', 'Securities_account', 'CD_account', 'Internet_banking', 'Credit_card']
data = data.drop(data.index[0])
labels = data["Loan"].unique()

# Random Selection of 80% Training Data and 20% Validation Data
# msk = np.random.rand(len(data)) < 0.8
# train = data[msk].reset_index(drop=True)
# validate = data[~msk].reset_index(drop=True)

# Selecting first 80% as Training Data and remaining as Validation Data
train, validate = np.split(data, [int(.8*len(data))])
validate = validate.reset_index(drop=True)

numerical_attributes = ['Age', 'Num_years_exp', 'Income', 'Zipcode', 'Family_size', 'Avg_spending_pm', 'Mortgage_value']
categorical_attributes = ['Education_level', 'Securities_account', 'CD_account', 'Internet_banking', 'Credit_card']


# In[4]:


label_count = []
for label in labels:
    label_count.append(train["Loan"].value_counts()[label])

total_label_count = label_count[0] + label_count[1]
label_probability = []
label_probability.append(label_count[0] / total_label_count)
label_probability.append(1 - label_probability[0])

count_list = [{}, {}]
categorical_probability = [{}, {}]

def calc_categorical_probabilities():
    for attr in categorical_attributes:
        for label in labels:
            label = int(label)
            count_list[label][attr] = {}
            categorical_probability[label][attr] = {}
            for val in train[attr].unique():
                count_list[label][attr][val] = 0

    for index, training_sample in train.iterrows():
        label = int(training_sample["Loan"])
        for attr in categorical_attributes:
            count_list[label][attr][training_sample[attr]] += 1

    for attr in categorical_attributes:
        for label in labels:
            label = int(label)
            for val in train[attr].unique():
                categorical_probability[label][attr][val] = (count_list[label][attr][val] / label_count[label])


# In[14]:


calc_categorical_probabilities()
categorical_probability


# In[17]:


mean_list = [{}, {}]
std_dev_list = [{}, {}]

def calc_numerical_mean_and_std_dev():
    for attr in numerical_attributes:
        for label in labels:
            label = int(label)
            mean_list[label][attr] = 0
            std_dev_list[label][attr] = 0

    for index, training_sample in train.iterrows():
        label = int(training_sample["Loan"])
        for attr in numerical_attributes:
            mean_list[label][attr] += training_sample[attr]

    for attr in numerical_attributes:
        for label in labels:
            label = int(label)
            mean_list[label][attr] /= label_count[label]
    
    for index, training_sample in train.iterrows():
        label = int(training_sample["Loan"])
        for attr in numerical_attributes:
            std_dev_list[label][attr] += pow((training_sample[attr] - mean_list[label][attr]), 2)

    for attr in numerical_attributes:
        for label in labels:
            label = int(label)
            std_dev_list[label][attr] /= (label_count[label] - 1)
            std_dev_list[label][attr] = math.sqrt(std_dev_list[label][attr])


# In[19]:


calc_numerical_mean_and_std_dev()


# In[20]:


def gaussian_prob(attr, val, label):
    mean = mean_list[label][attr]
    stdev = std_dev_list[label][attr]
    exponent = math.exp(-(math.pow(val - mean,2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent


# In[21]:


def naive_bayes_validation():
    TP = 0; TN = 0; FP = 0; FN = 0;

    for index, validation_sample in validate.iterrows():
        log_probability = [0] * 2
        for label in labels:
            label = int(label)
            for attr in numerical_attributes:
                log_probability[label] += log(gaussian_prob(attr, validation_sample[attr], label))
        
            for attr in categorical_attributes:
                log_probability[label] += log(categorical_probability[label][attr][validation_sample[attr]])

        log_probability[int(labels[0])] += log(label_probability[0])
        log_probability[int(labels[1])] += log(label_probability[1])        

        if log_probability[int(labels[0])] > log_probability[int(labels[1])]:
            prediction = labels[0]
        else:
            prediction = labels[1]

        if prediction == labels[1]:
            if prediction == validation_sample["Loan"]:
                TP += 1
            else:
                FP += 1
        else:
            if prediction == validation_sample["Loan"]:
                TN += 1
            else:
                FN += 1

    accuracy = float(TP + TN) / (TP + FP + TN + FN)
    if TP == 0:
        recall = 0.0
        precision = 0.0
        f1_score = 2 / ((1/(eps)) + (1/(eps)))
    else:
        recall = float(TP) / (TP + FN)
        precision = float(TP) / (TP + FP)
        f1_score = 2 / ((1/(recall)) + (1/(precision)))
    
    print ("TP = {}, TN = {}, FP = {}, FN = {}".format(TP, TN, FP, FN))
    print ("accuracy = {}, recall = {}, precision = {}, f1_score = {}".format(accuracy, recall, precision, f1_score))
    return (accuracy, precision, recall, f1_score)


# In[22]:


naive_bayes_validation()


# I plotted the distinct values of all attributes corresponding to their frequencies. <br>
# That plot seemed similar to Gaussian Distribution for some attributes. <br>
# Hence, I have used Gaussian probability distribution to calculate naive bayes probabilities

# In[ ]:




