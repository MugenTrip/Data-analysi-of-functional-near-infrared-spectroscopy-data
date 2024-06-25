import mne
import numpy as np
import matplotlib.pyplot as plt

class myRaw():

    def __init__(self, data: np.ndarray, n_channels: int=8, raw: mne.io.Raw=None):
        if raw is None:
            print("Missing mne.io.Raw argument.")
            return None
        self._s_frequency = raw.info['sfreq']
        ch_names = [f"PC_{i}" for i in range(1,n_channels+1)]
        ch_types = ["hbo"] * n_channels
        info = mne.create_info(ch_names, ch_types=ch_types, sfreq=self._s_frequency)
        
        self._raw = mne.io.RawArray(data, info)
        events, event_dict = mne.events_from_annotations(raw)
        print(events)
        print(event_dict)
        self._raw.set_annotations(mne.annotations_from_events(events, self._s_frequency, {2: "Real", 3: "Rest", 1: "Imagery"}))
        self._raw.annotations.set_durations(15)

    def getRaw(self):
        return self._raw
    
    def plot(self):
        self._raw.plot(n_channels=len(self._raw.ch_names), duration=500, show_scrollbars=False, scalings="auto")
        plt.show()