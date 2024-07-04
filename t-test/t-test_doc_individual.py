import os
import sys
import path
import scipy.stats as stats 
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)
from DataLoader import DataLoader
import numpy as np
import argparse

doc_path = os.path.join(os.path.curdir, '..\data\doc')
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--session", type=str, choices=["initial", "followup"], help="Session, could be initial or followup.", required=True)
args = parser.parse_args()
if args.session == "initial":
    doc_path = os.path.join(doc_path, 'initial')
elif args.session == "followup":
    doc_path = os.path.join(doc_path, 'followup')

dataloader = DataLoader(data_path=os.path.join(doc_path, 'data.npy'), event_path=os.path.join(doc_path, 'events.npy'), isDoc=True, isPCA=False)
patient_map = np.load(os.path.join(doc_path, 'map.npy'))

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