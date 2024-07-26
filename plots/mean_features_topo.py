import sys
import path
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)
import argparse
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from DataLoader import DataLoader
from model_svm import model_SVM
from DataLoader import DataLoader
from Hemo import HemoData
import mne
import math
from matplotlib import colormaps
from DataPath import DataPath
import matplotlib.pyplot as plt



BASE_PATH = os.path.join(os.path.curdir, 'data/')
MAX_LENGTH_HEALTHY = 1224
MAX_LENGTH_DOC = 7756

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d" , "--datatype", type=str, choices=["healthy", "icu", "doc"], help="Determine the data type, could be [healthy, icu, doc].", required=True)
    parser.add_argument("-s", "--session", type=str, choices=["initial", "followup"], help="Session, could be initial or followup.", required=True)
    parser.add_argument("-t" , "--task", type=str, choices=["physical", "imagery"], help="Determine the task type, it could be [physical, imagery].", required=True)
    args = parser.parse_args()

    data_path = None
    if args.datatype == "healthy":
            data_path = os.path.join(BASE_PATH, 'healthy')
            max_length = MAX_LENGTH_HEALTHY
    elif args.datatype == "icu":
            data_path = os.path.join(BASE_PATH, 'icu')
            max_length = MAX_LENGTH_HEALTHY
    elif args.datatype == "doc":
            data_path = os.path.join(BASE_PATH, 'doc')
            max_length = MAX_LENGTH_DOC

    if args.session == "initial":
            data_path = os.path.join(data_path, 'initial')
    elif args.session == "followup":
            data_path = os.path.join(data_path, 'followup')

    # Load a random file to obtain the montage
    datapath = DataPath(data_path, recursive=False)
    hemo = HemoData( datapath.getDataPaths()[0], preprocessing=True).getMneIoRaw()
    ch_idx_by_type = mne.channel_indices_by_type(hemo.info)

    dataloader = DataLoader(data_path=os.path.join(data_path,"data.npy"), event_path=os.path.join(data_path,"events.npy"), isDoc=False, isPCA=False)
    figure, axis = plt.subplots(1, 2)

    if args.datatype != "doc":
        if args.task == "physical":      
            stimuli_features, _ = dataloader.getFeaturesNormal(channels=[0,1,2,3,4,5,6,7])
            separetor = math.floor(stimuli_features.shape[0]/2)
            stimuli_features[separetor:,:,:] = -stimuli_features[separetor:,:,:]
        else:
            stimuli_features = dataloader.getFeaturesImageryNoPCA(channels=[0,1,2,3,4,5,6,7])
            stimuli_features = np.mean(stimuli_features, axis=1)

        mean_stimuli_features = np.mean(stimuli_features, axis=0)
        im, cn = mne.viz.plot_topomap( mean_stimuli_features[:,0].T, mne.pick_info(hemo.info, sel=ch_idx_by_type["hbo"]), show=False, axes=axis[0], vlim=(-0.015,0.015), cmap=colormaps["coolwarm"]) # vlim=(-4e-9,+4e-9)
        axis[0].set_title("(a) Mean(last 5 sec) of physical tongue movement trial (paired).", y=-0.1)
        cax = figure.colorbar(im, ax=axis[0], fraction=0.046, pad=0.04)

        mean_stimuli_features = np.mean(stimuli_features, axis=0)
        im, cn = mne.viz.plot_topomap( mean_stimuli_features[:,1].T, mne.pick_info(hemo.info, sel=ch_idx_by_type["hbo"]), show=False, axes=axis[1], vlim=(-1,1), cmap=colormaps["coolwarm"]) # 2e-7,2e-7
        axis[1].set_title("(b) Slope of physical tongue movement trial (paired).", y=-0.1)
        cax = figure.colorbar(im, ax=axis[1], fraction=0.046, pad=0.04)
    else:
        stimuli_features = dataloader.getFeaturesImageryNoPCA(channels=[0,1,2,3,4,5,6,7])
        stimuli_features = np.mean(stimuli_features, axis=1)

        mean_stimuli_features = np.mean(stimuli_features, axis=0)
        im, cn = mne.viz.plot_topomap( mean_stimuli_features[:,0].T, mne.pick_info(hemo.info, sel=ch_idx_by_type["hbo"]), show=False, axes=axis[0], vlim=(-0.015,0.015), cmap=colormaps["coolwarm"]) # vlim=(-4e-9,+4e-9)
        axis[0].set_title("(a) Mean(last 5 sec) of physical tongue movement trial (paired).", y=-0.1)
        cax = figure.colorbar(im, ax=axis[0], fraction=0.046, pad=0.04)

        mean_stimuli_features = np.mean(stimuli_features, axis=0)
        im, cn = mne.viz.plot_topomap( mean_stimuli_features[:,1].T, mne.pick_info(hemo.info, sel=ch_idx_by_type["hbo"]), show=False, axes=axis[1], vlim=(-1,1), cmap=colormaps["coolwarm"]) # 2e-7,2e-7
        axis[1].set_title("(b) Slope of physical tongue movement trial (paired).", y=-0.1)
        cax = figure.colorbar(im, ax=axis[1], fraction=0.046, pad=0.04)

    plt.show()