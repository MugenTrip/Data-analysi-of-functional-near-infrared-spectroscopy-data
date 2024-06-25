import mne
import numpy as np
from Hemo import HemoData
from DataPath import DataPath
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import argparse
from itertools import compress


path = "L:\\LovbeskyttetMapper\\CONNECT-ME\\DTU\\Alex_Data\\DoC\\data_initial\\"
datapath = DataPath(path, recursive=False)
total = 0
for id,file in enumerate(datapath.getDataPaths()):
    raw = mne.io.read_raw_snirf(file, optode_frame="mri")
    raw.load_data()

    # Converting from raw intensity to optical density
    data_od = mne.preprocessing.nirs.optical_density(raw)

    # Reject bad channels based on the scalp_coupling_index
    sci = mne.preprocessing.nirs.scalp_coupling_index(data_od)
    data_od.info["bads"] = list(compress(data_od.ch_names, sci < 0.8))
    print(data_od.info)
    print("Bad channels: " + str(data_od.info["bads"]) + ", len: " + str(len(data_od.info["bads"])))
    total += len(data_od.info["bads"])
    if False:
        fig, ax = plt.subplots(layout="constrained")
        ax.hist(sci)
        ax.set(xlabel="Scalp Coupling Index", ylabel="Count", xlim=[0, 1])
        plt.show()
    
    # Interpolate
    data_od.interpolate_bads()


print(f"Interpolated channgels: {total}" )