import os
import sys
import path
import scipy.stats as stats 
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)
from DataLoader import DataLoader
import argparse

BASE_PATH = os.path.join(os.path.curdir, 'data')
parser = argparse.ArgumentParser()
parser.add_argument("-d" , "--datatype", type=str, choices=["healthy", "icu", "doc"], help="Determine the data type, could be [healthy, icu, doc].", required=True)
parser.add_argument("-s", "--session", type=str, choices=["initial", "followup"], help="Session, could be initial or followup.", required=True)
args = parser.parse_args()

path_ = None
if args.datatype == "healthy":
        path_ = os.path.join(BASE_PATH, 'healthy')
elif args.datatype == "icu":
        path_ = os.path.join(BASE_PATH, 'icu')
elif args.datatype == "doc":
        path_ = os.path.join(BASE_PATH, 'doc')

if args.session == "initial":
        path_ = os.path.join(path_, 'initial')
elif args.session == "followup":
        path_ = os.path.join(path_, 'followup')

dataloader = DataLoader(data_path=os.path.join(path_, 'data.npy'), event_path=os.path.join(path_, 'events.npy'), isDoc=False, isPCA=False)
stimuli_features, imagery_features = dataloader.getFeaturesNoPCA()
print(stimuli_features.shape)
# stimuli_features = stimuli_features.reshape(stimuli_features.shape[0] * stimuli_features.shape[1], stimuli_features.shape[2], stimuli_features.shape[3])
# imagery_features = imagery_features.reshape(imagery_features.shape[0] * imagery_features.shape[1], imagery_features.shape[2], imagery_features.shape[3])
print(stimuli_features.shape)
for channel in range(8):
    for i in range(2):
        feature = "Mean" if i==0 else "Slope"
        # Performing the paired sample t-test 
        results = stats.ttest_rel(stimuli_features[:,channel,i], imagery_features[:,channel,i])
        print(f"{feature} Channel: {channel}: {results.pvalue}, corrected: {results.pvalue*16}")