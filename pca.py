# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 14:35:45 2023
@author: Mohak Sharda
"""

import numpy as np

class PCA:

    def __init__(self, desired_principal_components):
        self.desired_principal_components = desired_principal_components
        self.extracted_eigenvectors = None
        self.feature_mean = None
        self.covariance_matrix = None
        
    def fit(self,feature_table):
	    #mean
        self.feature_mean = np.mean(feature_table,axis=0)
        feature_table = feature_table - self.feature_mean
        #calculate covariance matrix
        #row = 1 sample, column = feature
        self.covariance_matrix = np.cov(feature_table.T) #this function needs column to be a sample
        #self.covariance_matrix = np.corrcoef(feature_table.T) #this function needs column to be a sample
        #eigenvectors and eigenvalues
        eigenvalues,eigenvectors = np.linalg.eig(self.covariance_matrix)
        #v[:, i]
        #sort eigenvectors
        eigenvectors = eigenvectors.T
        indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[indices]
        eigenvectors = eigenvectors[indices]
        #store first n ei genvectors
        self.extracted_eigenvectors = eigenvectors[0:self.desired_principal_components]
        
    def transform(self,feature_table):
        #project our data
        feature_table = feature_table - self.feature_mean
        return np.dot(feature_table,self.extracted_eigenvectors.T)
    

    def fit_transform(self,feature_table):
        self.fit(feature_table=feature_table)
        return self.transform(feature_table=feature_table)
    
    def get_weights(self):
        return self.extracted_eigenvectors
    
    def get_covariance(self):
        print(np.linalg.eig(self.covariance_matrix))
        print()
        return self.covariance_matrix