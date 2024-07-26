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
from matplotlib.legend_handler import HandlerTuple

def getName(file):
       '''
        Helping to get the name of an .snirf file.
        '''
       return file.split("\\")[-1].replace(".snirf","") 

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

    print(events.shape)
    figure, axis = plt.subplots(4, 1, figsize=(18, 9))
    figure.suptitle(f"Participant {getName(file)}")
    axis[0].set_title("Suplementery Motor Area")
    axis[1].set_title("Frontal Area")
    axis[2].set_title("Motor Area")
    axis[3].set_title("Parietal Area")
    # # Iterrate through channels
    for id, _ in enumerate(data):
        if id % 2 == 0:
            p2, p3 = [], []
            p6, = axis[id//2].plot(np.arange(data.shape[1]), data[id],color='blue')
            p7, = axis[id//2].plot(np.arange(data.shape[1]), data[id+1],color='red')
            axis[id//2].set_ylim([-4, 5])
            for idx in range(events.shape[0]):
                aggrated_time_of_event = idx*(153+142) + (idx // 8) * 336 # Every 15 sec 
                tmp = axis[id//2].axvline(x = events[idx,0], color = 'y')
                p2.append(tmp)
                tmp = axis[id//2].axvline(x =events[idx,0] + 153, color = 'c')
                p3.append(tmp)

            
            l = axis[id//2].legend([tuple(p2), tuple(p3), p6, p7], 
                                ['Physical Trigger', 'Rest Trigger', f"Channel {id+1}", f"Channel {id+2}"], handler_map={tuple: HandlerTuple(ndivide=None)}, loc='right')
    figure.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

