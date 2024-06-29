import os
import sys
import path
import scipy.stats as stats 
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)
from DataLoader import DataLoader
import numpy as np

doc_i = "L:\\LovbeskyttetMapper\\CONNECT-ME\\DTU\\Alex_Data\\DOC\\data_initial"
dataloader = DataLoader(data_path=doc_i+"\\data.npy", event_path=doc_i+"\\events.npy", isDoc=False, isPCA=False)
patient_map = np.load(doc_i + "\\map.npy")
stimuli_features, imagery_features = dataloader.getFeaturesImageryNoPCATtest()
print(stimuli_features.shape)
for idx, patient in enumerate(stimuli_features):
    print(f"Patient: {patient_map[idx]}")
    for channel in range(8):
        # Performing the paired sample t-test 
        for i in range(2):
            feature = "Mean" if i==0 else "Slope"
            results = stats.ttest_rel(stimuli_features[idx,:,channel,i], imagery_features[idx,:,channel,i])
            print(f"Mean Feature - Channel: {channel + 1}: {results.pvalue}, corrected: {results.pvalue*16}")
    print("\n")