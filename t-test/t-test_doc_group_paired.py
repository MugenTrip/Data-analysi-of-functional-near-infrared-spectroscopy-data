import os
import sys
import path
import scipy.stats as stats 
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)
from DataLoader import DataLoader


doc_i = "L:\\LovbeskyttetMapper\\CONNECT-ME\\DTU\\Alex_Data\\DOC\\data_initial"
dataloader = DataLoader(data_path=doc_i+"\\data.npy", event_path=doc_i+"\\events.npy", isDoc=False, isPCA=False)
stimuli_features, imagery_features = dataloader.getFeaturesImageryNoPCATtest()
print(stimuli_features.shape)
stimuli_features = stimuli_features.reshape(stimuli_features.shape[0] * stimuli_features.shape[1], stimuli_features.shape[2], stimuli_features.shape[3])
imagery_features = imagery_features.reshape(imagery_features.shape[0] * imagery_features.shape[1], imagery_features.shape[2], imagery_features.shape[3])
print(stimuli_features.shape)
for channel in range(8):
    # Performing the paired sample t-test 
    results = stats.ttest_rel(stimuli_features[:,channel,1], imagery_features[:,channel,1])
    print(f"Channel: {channel}: {results.pvalue}, corrected: {results.pvalue*8}")