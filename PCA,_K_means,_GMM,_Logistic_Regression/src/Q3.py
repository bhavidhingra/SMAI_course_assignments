#!/usr/bin/env python
# coding: utf-8

# In[52]:


import numpy as np
import pandas as pd
import ipdb
import pprint
import sys
import matplotlib.pyplot as plt
import math
import operator
from copy import deepcopy
from tabulate import tabulate

eps = np.finfo(float).eps
from numpy import log2 as log


# # Q3

# ## One vs All

# ## Reading Data

# In[53]:


def standardize_data(X):
    return (X - X.mean())/X.std()


# In[54]:


data = pd.read_csv("../input_data/wine-quality/data.csv", delimiter=';')

# removing the output column
data_std = standardize_data(data.iloc[:, :-1])
data_std[["quality"]] = data[["quality"]]
data_std.describe()

# msk = np.random.rand(len(data)) < 0.8
# train = X_std[msk].reset_index (drop=True)
# validate = X_std[~msk].reset_index (drop=True)

# Selecting first 80% as Training Data and remaining as Validation Data
train, validate = np.split(data_std, [int(.8*len(data_std))])
validate = validate.reset_index(drop=True)


# In[56]:


class LogisticRegression:
    def __init__(self, learning_rate = 0.01, max_iterations = 1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

    def intercept_add(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        X = self.intercept_add(X)

        # weights initialization
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.max_iterations):
            z = np.dot(X, self.theta)   # dot product
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= (self.learning_rate * gradient)
    
    def predict(self, X, threshold):
        X = self.intercept_add(X)
        predicted_prob = self.sigmoid(np.dot(X, self.theta))
        return predicted_prob >= threshold


# In[57]:


model = LogisticRegression(0.01, 10000)


# In[62]:


X_train = train.iloc[:, :-1]
y_train = train.iloc[:, -1]

num_samples = X_train.shape[0]
num_features = X_train.shape[1]
num_labels = 11


# In[63]:


X_validate = validate.iloc[:, :-1]
X_validate = model.intercept_add(X_validate)
y_validate = validate.iloc[:, -1]


# In[59]:


classifiers = np.zeros(shape=(num_labels, num_features+1))

for c in range(0, num_labels):
    label = (y_train == c).astype(int)
    model.fit(X_train, label)
    classifiers[c, :] = deepcopy(model.theta)


# In[60]:


z = np.dot(X_validate, classifiers.T)   # dot product
classProbabilities = model.sigmoid(z)
predictions = classProbabilities.argmax(axis=1)


# In[61]:


predictions
print("Training accuracy:", str(100 * np.mean(predictions == y_validate)) + "%")


# In[ ]:





# In[ ]:





# ## One vs One

# ## Make nc2 classifiers with subtables

# In[ ]:


num_classifiers = (num_labels) * (num_labels + 1) / 2
classifiers = np.zeros(shape=(num_classifiers, num_features+1))


# In[ ]:


c = 0
for i in range(num_labels):
    for j in range(i+1, num_labels):
        X_train = train[train["quality"] ]
        classifiers[c, :] = 


# ## Append the theta values of each classifier

# ## For each sample 
# ### pass through all the classifiers, assign a label having maximum frequency 
