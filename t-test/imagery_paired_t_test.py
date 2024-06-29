import os
import sys
import path
import scipy.stats as stats 
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)
from DataLoader import DataLoader

hc_path = "L:\\LovbeskyttetMapper\\CONNECT-ME\\DTU\\Alex_Data\\ICUPatients\\data_followup"
dataloader = DataLoader(data_path=hc_path+"\\data.npy", event_path=hc_path+"\\events.npy", isDoc=False, isPCA=False)
stimuli_features, imagery_features = dataloader.getFeaturesImageryNoPCATtest()
print(stimuli_features.shape)
stimuli_features = stimuli_features.reshape(stimuli_features.shape[0] * stimuli_features.shape[1], stimuli_features.shape[2], stimuli_features.shape[3])
imagery_features = imagery_features.reshape(imagery_features.shape[0] * imagery_features.shape[1], imagery_features.shape[2], imagery_features.shape[3])
print(stimuli_features.shape)
for channel in range(8):
    # Performing the paired sample t-test 
    for i in range(2):
        feature = "Mean" if i==0 else "Slope"
        results = stats.ttest_rel(stimuli_features[:,channel,i], imagery_features[:,channel,i])
        print(f"{feature} Channel: {channel+1}: {results.pvalue}, corrected: {results.pvalue*16}")