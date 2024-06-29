import os
import sys
import path
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)
import matplotlib.pyplot as plt
import numpy as np
from DataLoader import DataLoader
from Hemo import HemoData
import mne
from matplotlib import colormaps

hemo = HemoData("L:\\LovbeskyttetMapper\\CONNECT-ME\\DTU\\Alex_Data\\HealthyPatients\\data_initial\\C19_raw.snirf", preprocessing=True).getMneIoRaw()
ch_idx_by_type = mne.channel_indices_by_type(hemo.info)

doc_i = "L:\\LovbeskyttetMapper\\CONNECT-ME\\DTU\\Alex_Data\\DOC\\data_initial\\"
int_dataloader = DataLoader(data_path=doc_i+"data.npy", event_path=doc_i+"events.npy", isDoc=False, isPCA=False)
init_features = int_dataloader.getFeaturesImageryNoPCA()
initial_map = np.load(doc_i + "map.npy")

doc_f = "L:\\LovbeskyttetMapper\\CONNECT-ME\\DTU\\Alex_Data\\DOC\\data_followup\\"
f_dataloader = DataLoader(data_path=doc_f+"data.npy", event_path=doc_f+"events.npy", isDoc=False, isPCA=False)
f_features = f_dataloader.getFeaturesImageryNoPCA()
f_map = np.load(doc_f + "map.npy")

responders = ["P29_1", "P28_1", "P36_1", "P3_2", "P13_1", "P36_3", "P40_1"]
names = []
doc_features = None
for idx,patient in enumerate(initial_map):
    if patient in responders:
        print(patient)
        if doc_features is None:
            doc_features = np.expand_dims(init_features[idx], axis=0)
        else:
            doc_features = np.concatenate((doc_features, np.expand_dims(init_features[idx], axis=0)), axis=0)
        names.append(patient)

for idx,patient in enumerate(f_map):
    if patient in responders:
        print(patient)
        doc_features = np.concatenate((doc_features, np.expand_dims(f_features[idx], axis=0)), axis=0)
        names.append(patient)


print(len(names))
print(doc_features.shape)

figure, axis = plt.subplots(2, 4)

for idx,patient in enumerate(names): 
    mn = np.mean(doc_features[idx], axis=0)
    print(mn.shape)
    im, cn = mne.viz.plot_topomap( mn[:,1], mne.pick_info(hemo.info, sel=ch_idx_by_type["hbo"]), show=False, axes=axis[idx%2][idx//2],  vlim=(-1,1), cmap=colormaps["coolwarm"])
    axis[idx%2][idx//2].set_title(f"Average paired-slope feature of {patient[0]}.", y=-0.2)

    cax = figure.colorbar(im, ax=axis[idx%2][idx//2], fraction=0.046, pad=0.04)
figure.delaxes(axis[1,3])

# mne.viz.plot_topomap( mean_stimuli_features[:,1], mne.pick_info(hemo.info, sel=ch_idx_by_type["hbo"]), show=False, axes=axis[1], vlim=(-0.5,0.5))
# print(mean_stimuli_features[:,1])
# axis[1].set_title("(b) Slope of imagery tongue motor period (un-paired).", y=-0.1)
plt.show()