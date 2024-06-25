import mne
import numpy as np
from Hemo import HemoData
from DataPath import DataPath
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import argparse
from itertools import compress


path = "L:\\LovbeskyttetMapper\\CONNECT-ME\\DTU\\Alex_Data\\HealthyPatients\\data_initial\\"
datapath = DataPath(path, recursive=False)
total = 0
for id,file in enumerate(datapath.getDataPaths()):
    raw = mne.io.read_raw_snirf(file, optode_frame="mri")
    raw.load_data()

    # Converting from raw intensity to optical density
    data_od = mne.preprocessing.nirs.optical_density(raw)
    data_od.plot(n_channels=len(data_od.ch_names), duration=1000, show_scrollbars=False, title="Optical Density Before Pre-Processing", scalings="auto")

    hemo_data = HemoData(file)
    hemo_data.getMneIoRaw().plot(n_channels=len(hemo_data.getMneIoRaw().ch_names), duration=1000, show_scrollbars=False, title="Hbo and Hbr After Pre-Processing", scalings="auto")
    plt.show()