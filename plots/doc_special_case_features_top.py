import sys
import path
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)
import os
from DataPath import DataPath
import mne
from Hemo import HemoData
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import math
from sklearn import svm
from DataLoader import DataLoader
from matplotlib import colormaps
import matplotlib.pyplot as plt
from mpl_toolkits.axisartist.axislines import Subplot 

def getName(file):
       '''
        Helping to get the name of an .snirf file.
        '''
       return file.split("\\")[-1].replace(".snirf","") 

def slopePerChannel(data):
        '''
        Calculate the slope for each channel of the data array.
        data: n_channels X samples
        '''
        # Calculate slope for each event for each channel: events X channels X 1 (slope)
        return np.polyfit(np.arange(data.T.shape[0]), data.T,1)[0]

def extractFeatures(data, start, stop, feature_window: int= 53):
        '''
        Method used to extract the mean and the slope for DOC participants. You can add more featutes in the tuple if you want.
        '''
        # For the slope we need to do this for every patient seperately
        slope = np.expand_dims(slopePerChannel(data[:,start:stop]), axis=0)
        return np.concatenate(
                        (
                         slope.reshape(slope.shape[1],1), 
                         np.expand_dims(np.mean(data[:,stop-feature_window:stop], axis=1), axis=1)
                        ),
                        axis = 1
                    )

def getSpecialCaseFeature(data, events):
    duration = 153
    end = 295
    for event_id, event in enumerate(events):
        features_physical = extractFeatures(data,event[0], event[0]+duration)
        features_rest = extractFeatures(data,event[0]+duration, event[0]+end)

        if event_id == 0:
            event_features = np.expand_dims(features_physical - features_rest, axis=0)
        else:
            event_features = np.concatenate((event_features, np.expand_dims(features_physical - features_rest, axis=0)), axis=0)

    print(event_features.shape)
    return event_features

BASE_PATH = os.path.join(os.path.curdir, 'data/doc/special_cases')
datapath = DataPath(BASE_PATH, recursive=False)

for id,file in enumerate(datapath.getDataPaths()):
    raw = HemoData(file, preprocessing=True, isDoC=False, isPloting=False)
    raw_haemo = raw.getMneIoRaw()
    ch_idx_by_type = mne.channel_indices_by_type(raw_haemo.info)
    data = raw_haemo.get_data(picks=["hbo"]) # channels x samples
    events, event_dict = mne.events_from_annotations(raw_haemo)

    # Normalize each channel separetely
    scaler = StandardScaler()
    data = scaler.fit_transform(data.T).T
    paired_features =  getSpecialCaseFeature(data, events)

    figure, axis = plt.subplots(1, 2)
    figure.suptitle(f"Participant {getName(file)}")
    mn = np.mean(paired_features, axis=0)
    print(mn)
    #Mean
    im, cn = mne.viz.plot_topomap( mn[:,0], mne.pick_info(raw_haemo.info, sel=ch_idx_by_type["hbo"]), axes=axis[0], show=False,  vlim=(-0.03,0.03), cmap=colormaps["coolwarm"])
    axis[0].set_title("Mean")
    cax = figure.colorbar(im, ax=axis[0], fraction=0.046, pad=0.04)
    #Slope
    im, cn = mne.viz.plot_topomap( mn[:,1], mne.pick_info(raw_haemo.info, sel=ch_idx_by_type["hbo"]), axes=axis[1], show=False,  vlim=(-2,2), cmap=colormaps["coolwarm"])
    axis[1].set_title("Slope")
    cax = figure.colorbar(im, ax=axis[1], fraction=0.046, pad=0.04)
    print(paired_features.shape)
    plt.show()
