#!/usr/bin/env python
# coding: utf-8

# ## Initial Imports

# In[73]:


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


# # Q1

# ## Part-1: PCA

# ## Reading Data

# In[74]:


data = pd.read_csv("../input_data/data.csv")
labels = list(data.iloc[:,-1])
# data


# In[75]:


data_std = data.iloc[:, :-1]
data_std = (data_std - data_std.mean())/data_std.std()
data_std[["xAttack"]] = data[["xAttack"]]


# In[76]:


# split data table into data X_std and class labels y
X_std = (data_std.iloc[:, :-1]).values

# Mean
mean_vec = np.array([(data_std.describe()).iloc[1]])


# In[77]:


cov_mat = (((X_std - mean_vec).T) @ (X_std - mean_vec)) / (X_std.shape[0] - 1)
# cov_mat.shape


# In[78]:


cov_mat = np.cov(X_std.T)
# cov_mat.shape


# In[79]:


eig_vals, eig_vecs = np.linalg.eig(cov_mat)
# eig_vecs.shape
# print('Eigenvectors \n%s' %eig_vecs)
# print('\nEigenvalues \n%s' %eig_vals)


# In[80]:


for ev in eig_vecs:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
print('Norm of all eigen vectors is 1')


# In[81]:


# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
# print('Eigenvalues in descending order:')
# for i in eig_pairs:
#     print(i[0])


# In[82]:


tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

trace1 = dict(
    type='bar',
    x=['PC %s' %i for i in range(1,5)],
    y=var_exp,
    name='Individual'
)

trace2 = dict(
    type='scatter',
    x=['PC %s' %i for i in range(1,5)], 
    y=cum_var_exp,
    name='Cumulative'
)

t = [trace1, trace2]

layout=dict(
    title='Explained variance by different principal components',
    yaxis=dict(
        title='Explained variance in percent'
    ),
    annotations=list([
        dict(
            x=1.16,
            y=1.05,
            xref='paper',
            yref='paper',
            text='Explained Variance',
            showarrow=False,
        )
    ])
)


# In[124]:


errors = []
num_attributes_used = list(range(1, 30))
for num_features in num_attributes_used:
    # initializing matrix with a dummy column of 0's
    matrix_w = np.zeros(shape=(29,1))
    for i in range(num_features):
        matrix_w = np.hstack((matrix_w, eig_pairs[i][1].reshape(29,1)))

    # removing the first dummy columns
    matrix_w = matrix_w[:, 1:]
    reduced_X_std = X_std @ matrix_w
    reconstructed_X_std = ((reduced_X_std @ matrix_w.T))
    E = ((X_std - reconstructed_X_std) ** 2).mean() * 100
    errors.append(E)


# In[127]:


fig, ax = plt.subplots(figsize=(12,6))
ax.plot(num_attributes_used, errors, color="purple", lw=1, ls='-', marker='s', markersize=4, 
        markerfacecolor="yellow", markeredgewidth=2, markeredgecolor="blue", label='Reconstruction Error (%)');

plt.title("Error (%) vs Number of Features used")
plt.xticks(num_attributes_used)
plt.ylabel("Error (%)")
plt.xlabel("Number of Features")
plt.legend()
plt.show()


# In[86]:





# In[87]:



    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot( attribute_list,error_list, color="Red", lw=1, ls='-');
    plt.xlabel("No. of Attributes ")
    plt.ylabel("Error Percentage")
    plt.show()


# ## Todo :
#    Plot error vs number of features.

# In[ ]:





# In[ ]:





# ## Part-2: K-means

# In[88]:


def KMeans(data, k, max_iterations):
    n = data.shape[0]
    c = data.shape[1]
    
    std = np.std(data, axis = 0)
    mean = np.mean(data, axis = 0)
    centers = np.random.randn(k,c)*std + mean

    centers_old = np.zeros(centers.shape)   # to store old centers
    centers_new = deepcopy(centers)         # Store new centers

    clusters = np.zeros(n)
    distances = np.zeros((n,k))
    error = np.linalg.norm(centers_new - centers_old)
    
    fig, ax = plt.subplots(figsize=(16,8))
    ax.scatter(data[:,0], data[:,1], s=7)
    ax.scatter(centers_new[:,0], centers_new[:,1], marker='*', c='r', s=150)
    plt.show()

    # When, after an update, the estimate of that center stays the same, exit loop
    while max_iterations != 0 and error != 0:        
        # Measure the distance to every center
        for i in range(k):
            distances[:,i] = np.linalg.norm(data - centers_new[i], axis=1)

        # Assign all training data to closest center
        clusters = np.argmin(distances, axis = 1)

        centers_old = deepcopy(centers_new)
        # Calculate mean for every cluster and update the center
        for i in range(k):
            centers_new[i] = np.mean(data[clusters == i], axis=0)

        error = np.linalg.norm(centers_new - centers_old)
        max_iterations -= 1
        
#         fig, ax = plt.subplots(figsize=(16,8))
#         ax.scatter(data[:,0], data[:,1], s=7)
#         ax.scatter(centers_new[:,0], centers_new[:,1], marker='*', c='r', s=150)
#         plt.show()

    print (max_iterations)
    fig, ax = plt.subplots(figsize=(14,7))
    colors = ['r', 'g', 'b', 'c', 'y']
    markers = ['+', '^', 'o', 'v', 's']
    for i in range(k):
        ax.scatter(data[clusters == i][:, 0], data[clusters == i][:, 1], s=7, marker=markers[i], c=colors[i])
        ax.scatter(centers_new[i,0], centers_new[i,1], marker='*', c='black', s=150)
    plt.show()

    return clusters, centers_new


# In[89]:


clusters, centers = KMeans(reduced_X_std, 5, 100)


# In[90]:


cluster_series = pd.Series(clusters, name='Clusters')
label_series = pd.Series(labels, name='Classes')
df_confusion = pd.crosstab(label_series, cluster_series, margins=True)
df_confusion = df_confusion.iloc[:-1, :-1]
print(tabulate(df_confusion, headers='keys', tablefmt='psql'))


# In[91]:


# Pie chart using Pandas

for i in range(0,5):
    plot = df_confusion.plot.pie(y= i, figsize=(5, 5))

# # Pie chart using Matplotlib
# for i in range(0,5):
#     labels = ['dos', 'normal', 'prob', 'r2l' , 'u2r']
#     values = df_confusion[i].tolist()
#     # only "explode" the 2nd slice (i.e. 'Hogs')
#     explode = (0, 0, 0, 0,0)
#     #add colors
#     colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#ff9901']
#     fig1, ax1 = plt.subplots()
#     ax1.pie(values, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
#             shadow=True, startangle=90)
#     # Equal aspect ratio ensures that pie is drawn as a circle
#     ax1.axis('equal')
#     plt.tight_layout()
#     plt.show()


# In[110]:


def calculate_purity(confusion_df):
    max_vals = np.array([confusion_df.max()])
    purity = max_vals.sum()/data.shape[0]
    print (purity)


# In[93]:


calculate_purity(df_confusion)


# In[94]:


reduced_X_std.shape


# In[ ]:





# In[ ]:





# ## Part-3: GMM

# ### Imports

# In[95]:


from sklearn.mixture import GaussianMixture
# data["xAttack"].unique()


# In[96]:


gmm = GaussianMixture(n_components=5, random_state=3)
clf = gmm.fit(reduced_X_std)
clf.weights_


# In[97]:


fig, ax = plt.subplots(figsize=(14,7))
ax.scatter(reduced_X_std[:,0], reduced_X_std[:,1])
for i in range(5):
    ax.scatter(gmm.means_[i,0], gmm.means_[i,1], marker='*', c='black', s=150)
# ax.axis('equal')
plt.show()


# In[98]:


pred_clusters = gmm.predict(reduced_X_std)
pred_clusters


# In[99]:


print(gmm.covariances_.shape)


# In[104]:


cluster_series = pd.Series(pred_clusters, name='Clusters')
label_series = pd.Series(labels, name='Classes')
gmm_confusion_df = pd.crosstab(label_series, cluster_series, margins=True)
gmm_confusion_df = gmm_confusion_df.iloc[:-1, :-1]
print(tabulate(gmm_confusion_df, headers='keys', tablefmt='psql'))


# In[105]:


for i in range(0,5):
    plot = gmm_confusion_df.plot.pie(y= i, figsize=(5, 5))
    
# # Pie chart using Matplotlib
# for i in range(0,5):
#     labels = ['dos', 'normal', 'prob', 'r2l' , 'u2r']
#     values = gmm_confusion_df[i].tolist()
#     # only "explode" the 2nd slice (i.e. 'Hogs')
#     explode = (0, 0, 0, 0,0)
#     #add colors
#     colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#ff9901']
#     fig1, ax1 = plt.subplots()
#     ax1.pie(values, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
#             shadow=True, startangle=90)
#     # Equal aspect ratio ensures that pie is drawn as a circle
#     ax1.axis('equal')
#     plt.tight_layout()
#     plt.show()


# In[111]:


calculate_purity(gmm_confusion_df)


# ## Part-4

# In[113]:


from sklearn.cluster import AgglomerativeClustering


# In[114]:


cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')  
cluster.fit_predict(reduced_X_std)


# In[115]:


fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(reduced_X_std[:,0], reduced_X_std[:,1], c=cluster.labels_, cmap='rainbow')


# In[116]:


cluster_series = pd.Series(cluster.labels_, name='Clusters')
label_series = pd.Series(labels, name='Classes')
hc_confusion_df = pd.crosstab(label_series, cluster_series, margins=True)
hc_confusion_df = hc_confusion_df.iloc[:-1, :-1]
print(tabulate(hc_confusion_df, headers='keys', tablefmt='psql'))


# In[118]:


for i in range(0,5):
    plot = hc_confusion_df.plot.pie(y= i, figsize=(5, 5))
    
# # Pie chart using Matplotlib
# for i in range(0,5):
#     labels = ['dos', 'normal', 'prob', 'r2l' , 'u2r']
#     values = gmm_confusion_df[i].tolist()
#     # only "explode" the 2nd slice (i.e. 'Hogs')
#     explode = (0, 0, 0, 0,0)
#     #add colors
#     colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#ff9901']
#     fig1, ax1 = plt.subplots()
#     ax1.pie(values, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
#             shadow=True, startangle=90)
#     # Equal aspect ratio ensures that pie is drawn as a circle
#     ax1.axis('equal')
#     plt.tight_layout()
#     plt.show()


# In[119]:


calculate_purity(hc_confusion_df)


# ## Part-5

# ### If you were to do dimensionality reduction on original data, could you use PCA?
# * If our goal is solely dimension reduction, then PCA is the way to go, we generally regard PCA on mixtures of variables with less than enthusiasm. Doing regular PCA on the raw variables is not recommended. We should first create a distance matrix and then operate on that.
# * In my opinion, PCA can be applied on mixed data-type by following way :-
# 
# * Make distance matrix using gower’s distance . In R it can be done by : - library(cluster); dist <- daisy(college.data,metric = "gower") Then we can use this distance matrix to reduce dimension.
# 
# * But I will recommend this method only if we have small data-set of less than 7000 rows. Let say if we have data-set of 100,000 rows , then it’s better to do one good encoding and then PCA or MULTIPLE CORRESPONDENCE ANALYSIS (MCA) rather than distance matrix because size of distance matrix will be (100,000 X 100,000), which will take forever for PCA to operate on.
