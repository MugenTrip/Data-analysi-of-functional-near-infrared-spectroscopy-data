import os
import sys
import path
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent.parent)
import matplotlib.pyplot as plt
import numpy as np
from DataLoader import DataLoader
from Hemo import HemoData
from matplotlib import colormaps
import mne


hemo = HemoData("L:\\LovbeskyttetMapper\\CONNECT-ME\\DTU\\Alex_Data\\HealthyPatients\\data_initial\\C19_raw.snirf", preprocessing=True).getMneIoRaw()
ch_idx_by_type = mne.channel_indices_by_type(hemo.info)

hc_path = "L:\\LovbeskyttetMapper\\CONNECT-ME\\DTU\\Alex_Data\\ICUPatients\\data_initial\\"
dataloader_initial = DataLoader(data_path=hc_path+"data.npy", event_path=hc_path+"events.npy", isDoc=False, isPCA=False)
initial_map = np.load(hc_path + "map.npy")
hc_path = "L:\\LovbeskyttetMapper\\CONNECT-ME\\DTU\\Alex_Data\\ICUPatients\\data_followup\\"
dataloader_followup = DataLoader(data_path=hc_path+"data.npy", event_path=hc_path+"events.npy", isDoc=False, isPCA=False)
followup_map = np.load(hc_path + "map.npy")

ids = []
for idx, patient_name in enumerate(followup_map):
    for idy, query_name in enumerate(initial_map):
        if patient_name[0].replace("_F","") == query_name[0]:
            print(f"HIT {patient_name[0]} == {query_name[0]}")
            ids.append(idy)

print(ids)

figure, axis = plt.subplots(2, 2)

stimuli_features = dataloader_followup.getFeaturesImageryNoPCA(channels=[0,1,2,3,4,5,6,7])
print(stimuli_features.shape)

mean_stimuli_features = np.mean(np.mean(stimuli_features, axis=1),axis=0)
print(mean_stimuli_features.shape)
im, cn = mne.viz.plot_topomap( mean_stimuli_features[:,0].T, mne.pick_info(hemo.info, sel=ch_idx_by_type["hbo"]), show=False, axes=axis[0,0], vlim=(-0.015,0.015), cmap=colormaps["coolwarm"])
print(mean_stimuli_features[:,0])
axis[0,0].set_title("(a) Average paired-mean feature. (16 participants of the followup session of the ICU)", y=-0.1)
cax = figure.colorbar(im, ax=axis[0,0], fraction=0.046, pad=0.04)

im, cn = mne.viz.plot_topomap( mean_stimuli_features[:,1].T, mne.pick_info(hemo.info, sel=ch_idx_by_type["hbo"]), show=False, axes=axis[0,1], vlim=(-1,1), cmap=colormaps["coolwarm"])
print(mean_stimuli_features[:,1])
axis[0,1].set_title("(b) Average paired-slope feature. (16 participants of the followup session of the ICU)", y=-0.1)
cax = figure.colorbar(im, ax=axis[0,1], fraction=0.046, pad=0.04)
# ####################################################################################################################################

stimuli_features = dataloader_initial.getFeaturesImageryNoPCA(channels=[0,1,2,3,4,5,6,7])
print(stimuli_features.shape)
print("******************************************************")
print(stimuli_features.shape)
stimuli_features = stimuli_features[ids,:,:]
print(stimuli_features.shape)

mean_stimuli_features = np.mean(np.mean(stimuli_features, axis=1),axis=0)
print(mean_stimuli_features.shape)
im, cn = mne.viz.plot_topomap( mean_stimuli_features[:,0], mne.pick_info(hemo.info, sel=ch_idx_by_type["hbo"]), show=False, axes=axis[1,0], vlim=(-0.015,0.015), cmap=colormaps["coolwarm"])
print(mean_stimuli_features[:,0])
axis[1,0].set_title("(c) Average paired-mean feature for the same 16 patients of the ICU initial session.", y=-0.1)
cax = figure.colorbar(im, ax=axis[1,0], fraction=0.046, pad=0.04)

im, cn = mne.viz.plot_topomap( mean_stimuli_features[:,1], mne.pick_info(hemo.info, sel=ch_idx_by_type["hbo"]), show=False, axes=axis[1,1], vlim=(-1,1), cmap=colormaps["coolwarm"])
print(mean_stimuli_features[:,1])
axis[1,1].set_title("(d) Average paired-slope feature for the same 16 patients of the ICU initial session.", y=-0.1)
cax = figure.colorbar(im, ax=axis[1,1], fraction=0.046, pad=0.04)
plt.show()
