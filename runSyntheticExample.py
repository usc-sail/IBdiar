#!/bin/python

# Date created: Dec 20 2017

# Used mostly during algorithm development
# Defines a synthetic variable with 4 clusters. AWGN is added to the conditional distribution mainly to help with visualising the dendrogram

import numpy as np
from functions import *
from scipy.cluster.hierarchy import fcluster

np.random.seed(1000)
np.set_printoptions(precision=5)   
N,P = 60,4
p_y_x = np.empty((N,P))
for i in range(N/4):    
    p_y_x[i] = [0.91,0.03,0.03,0.03]
for i in range(N/4,N/2):    
    p_y_x[i] = [0.03,0.91,0.03,0.03]
for i in range(N/2,3*N/4):    
    p_y_x[i] = [0.03,0.03,0.91,0.03]
for i in range(3*N/4,N):    
    p_y_x[i] = [0.03,0.03,0.03,0.91]
p_y_x = p_y_x + 10e-2*np.random.randn(p_y_x.shape[0],p_y_x.shape[1])
p_y_x[p_y_x<0] = 0
Z,C = cluster(p_y_x, 10, 1)
predLabels = fcluster(Z,4,criterion='maxclust')
