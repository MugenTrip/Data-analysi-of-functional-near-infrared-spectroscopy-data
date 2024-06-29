import os
import sys
import path
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)
import mne
import numpy as np
from DataPath import DataPath
from sklearn.decomposition import PCA


path = "L:\\LovbeskyttetMapper\\CONNECT-ME\\DTU\\Alex_Data\\DoC\\test\\"
datapath = DataPath(path, recursive = False, fileType = "npy")
print(len(datapath.getDataPaths()))
print(datapath.getDataPaths())

for id,file in enumerate(datapath.getDataPaths()):
    print(file)
    control = np.load(file) # channels X samples
    for area in range(4):
        pca = PCA(n_components=1)
        data_tmp = control[2*area:2*area+2,:]
        #print(data_tmp.shape)
        # PCA
        pca.fit(data_tmp.T)
        # Sign check 
        max_id = np.argmax(np.abs(pca.components_))
        if pca.components_[0,max_id] < 0:
            pca.components_ = -pca.components_
        print(pca.components_)
