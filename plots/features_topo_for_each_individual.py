import sys
import path
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)
import argparse
import os
import numpy as np
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
    initial_map = np.load(os.path.join(data_path,"map.npy"))

    if args.datatype == "doc":
        features = dataloader.getFeaturesImageryNoPCA(channels=[0,1,2,3,4,5,6,7])
    else:
        if args.task == "physical":      
            features, _ = dataloader.getFeaturesNormal(channels=[0,1,2,3,4,5,6,7])
            separetor = math.floor(features.shape[0]/2)
            features[separetor:,:,:] = -features[separetor:,:,:]
            features = np.expand_dims(features, axis=1)
            print(features.shape)
        else:
            features = dataloader.getFeaturesImageryNoPCA(channels=[0,1,2,3,4,5,6,7])
            print(features.shape)     

    rows = math.ceil(features.shape[0] / 5)
    print(rows)
    figure, axis = plt.subplots(rows, 5)
    figure.suptitle("Average 'mean' feature for each individual, as a topomap.")
    for idx,patient in enumerate(features): 
        mn = np.mean(features[idx], axis=0)
        im, cn = mne.viz.plot_topomap( mn[:,0], mne.pick_info(hemo.info, sel=ch_idx_by_type["hbo"]), show=False, axes=axis[idx%rows][idx//rows],  vlim=(-0.015,0.015), cmap=colormaps["coolwarm"])
        axis[idx%rows][idx//rows].set_title(f"{initial_map[idx][0]}", y=-0.3)

        cax = figure.colorbar(im, ax=axis[idx%rows][idx//rows], fraction=0.046, pad=0.04)
        #figure.delaxes(axis[7,4])
    plt.show()

    figure, axis = plt.subplots(rows, 5)
    figure.suptitle("Average 'slope' feature for each individual, as a topomap.")
    for idx,patient in enumerate(features): 
        mn = np.mean(features[idx], axis=0)
        im, cn = mne.viz.plot_topomap( mn[:,1], mne.pick_info(hemo.info, sel=ch_idx_by_type["hbo"]), show=False, axes=axis[idx%rows][idx//rows],  vlim=(-1,1), cmap=colormaps["coolwarm"])
        axis[idx%rows][idx//rows].set_title(f"{initial_map[idx][0]}", y=-0.3)

        cax = figure.colorbar(im, ax=axis[idx%rows][idx//rows], fraction=0.046, pad=0.04)
        #figure.delaxes(axis[7,4])
    plt.show()