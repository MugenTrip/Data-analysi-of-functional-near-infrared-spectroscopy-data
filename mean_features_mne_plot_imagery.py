import matplotlib.pyplot as plt
import numpy as np
from DataLoader import DataLoader
from Hemo import HemoData
import mne
from matplotlib import colormaps

hemo = HemoData("L:\\LovbeskyttetMapper\\CONNECT-ME\\DTU\\Alex_Data\\HealthyPatients\\data_initial\\C19_raw.snirf", preprocessing=True).getMneIoRaw()
ch_idx_by_type = mne.channel_indices_by_type(hemo.info)

hc_path = "L:\\LovbeskyttetMapper\\CONNECT-ME\\DTU\\Alex_Data\\HealthyPatients\\data_initial\\"
dataloader = DataLoader(data_path=hc_path+"data.npy", event_path=hc_path+"events.npy", isDoc=False, isPCA=False)

figure, axis = plt.subplots(1, 2)

stimuli_features = dataloader.getFeaturesImageryNoPCA()
print(stimuli_features.shape)

mean_stimuli_features = np.mean(np.mean(stimuli_features, axis=1),axis=0)
print(mean_stimuli_features.shape)
im, cn = mne.viz.plot_topomap( mean_stimuli_features[:,0], mne.pick_info(hemo.info, sel=ch_idx_by_type["hbo"]), show=False, axes=axis[0],  vlim=(-0.015,0.015), cmap=colormaps["coolwarm"])
print(mean_stimuli_features[:,0])
axis[0].set_title("(a) Mean (last 5 sec) of imagery tongue motor period (paired).", y=-0.1)
cax = figure.colorbar(im, ax=axis[0], fraction=0.046, pad=0.04)

im, cn = mne.viz.plot_topomap( mean_stimuli_features[:,1], mne.pick_info(hemo.info, sel=ch_idx_by_type["hbo"]), show=False, axes=axis[1], vlim=(-1,1), cmap=colormaps["coolwarm"])
print(mean_stimuli_features[:,1])
axis[1].set_title("(b) Slope of imagery tongue motor period (paired).", y=-0.1)
cax = figure.colorbar(im, ax=axis[1], fraction=0.046, pad=0.04)
plt.show()