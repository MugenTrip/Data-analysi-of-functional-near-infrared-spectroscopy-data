import matplotlib.pyplot as plt
import numpy as np
from DataLoader import DataLoader
from Hemo import HemoData
import mne
import math
from matplotlib import colormaps

hemo = HemoData("L:\\LovbeskyttetMapper\\CONNECT-ME\\DTU\\Alex_Data\\HealthyPatients\\data_initial\\C19_raw.snirf", preprocessing=True).getMneIoRaw()
ch_idx_by_type = mne.channel_indices_by_type(hemo.info)

hc_path = "L:\\LovbeskyttetMapper\\CONNECT-ME\\DTU\\Alex_Data\\HealthyPatients\\data_initial\\"
dataloader = DataLoader(data_path=hc_path+"data.npy", event_path=hc_path+"events.npy", isDoc=False, isPCA=False)

figure, axis = plt.subplots(1, 2)

stimuli_features, _ = dataloader.getFeaturesNormal(channels=[0,1,2,3,4,5,6,7])
separetor = math.floor(stimuli_features.shape[0]/2)
stimuli_features[separetor:,:,:] = -stimuli_features[separetor:,:,:]
print(stimuli_features.shape)

mean_stimuli_features = np.mean(stimuli_features, axis=0)
print(mean_stimuli_features.shape)
im, cn = mne.viz.plot_topomap( mean_stimuli_features[:,0].T, mne.pick_info(hemo.info, sel=ch_idx_by_type["hbo"]), show=False, axes=axis[0], vlim=(-0.015,0.015), cmap=colormaps["coolwarm"]) # vlim=(-4e-9,+4e-9)
axis[0].set_title("(a) Mean(last 5 sec) of physical tongue movement trial (paired).", y=-0.1)
cax = figure.colorbar(im, ax=axis[0], fraction=0.046, pad=0.04)
#cax.set_label(r"Impedance (k$\Omega$)")

mean_stimuli_features = np.mean(stimuli_features, axis=0)
print(mean_stimuli_features.shape)
im, cn = mne.viz.plot_topomap( mean_stimuli_features[:,1].T, mne.pick_info(hemo.info, sel=ch_idx_by_type["hbo"]), show=False, axes=axis[1], vlim=(-1,1), cmap=colormaps["coolwarm"]) # 2e-7,2e-7
print(mean_stimuli_features[:,1])
axis[1].set_title("(b) Slope of physical tongue movement trial (paired).", y=-0.1)
cax = figure.colorbar(im, ax=axis[1], fraction=0.046, pad=0.04)
plt.show()

hc_path = "L:\\LovbeskyttetMapper\\CONNECT-ME\\DTU\\Alex_Data\\ICUPatients\\data_initial\\"
dataloader = DataLoader(data_path=hc_path+"\\data.npy", event_path=hc_path+"\\events.npy", isDoc=False, isPCA=False)

figure, axis = plt.subplots(1, 2)

stimuli_features, _ = dataloader.getFeaturesNormal(channels=[0,1,2,3,4,5,6,7])
separetor = math.floor(stimuli_features.shape[0]/2)
stimuli_features[separetor:,:,:] = -stimuli_features[separetor:,:,:]
print(stimuli_features.shape)

mean_stimuli_features = np.mean(stimuli_features, axis=0)
print(mean_stimuli_features.shape)
im, cn = mne.viz.plot_topomap( mean_stimuli_features[:,0].T, mne.pick_info(hemo.info, sel=ch_idx_by_type["hbo"]), show=False, axes=axis[0], vlim=(-0.015,0.015), cmap=colormaps["coolwarm"])
axis[0].set_title("(a) Mean(last 5 sec) of physical tongue movement trial (paired).", y=-0.1)
cax = figure.colorbar(im, ax=axis[0], fraction=0.046, pad=0.04)
#cax.set_label(r"Impedance (k$\Omega$)")

mean_stimuli_features = np.mean(stimuli_features, axis=0)
print(mean_stimuli_features.shape)
im, cn = mne.viz.plot_topomap( mean_stimuli_features[:,1].T, mne.pick_info(hemo.info, sel=ch_idx_by_type["hbo"]), show=False, axes=axis[1], vlim=(-1,1), cmap=colormaps["coolwarm"])
print(mean_stimuli_features[:,1])
axis[1].set_title("(b) Slope of physical tongue movement trial (paired).", y=-0.1)
cax = figure.colorbar(im, ax=axis[1], fraction=0.046, pad=0.04)
plt.show()


