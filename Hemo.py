import numpy as np
import mne
import mne_nirs
from itertools import compress
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class HemoData():
    """
    It reads an fnirs file and stores the hemoglobin signal. 

    Attributes:
        data (mne.io.Raw): The mne.io.Raw object with the hemoglobin data.
        shortChannels (list): A list with the ID (initial raw data) of the short channels. 
        isPloting (bool): True if you want to plot graphs during initialization.
        useShortChannelRegression (bool): True if you want to use short channel regression during preprocessing.
    """
    
    def __init__(self, path: str, preprocessing: bool=True, isPloting: bool=False, useShortChannelRegression: bool=True, usePCA: bool=False, isDoC: bool=False) -> None:
        """
        Initializes a HemoData object.
 
        Parameters:
            path (str): The absolute path of the fnirs file.
            preprocessing (bool): True if data are in raw format, False if they are already hemoglobin. 
            isPloting (bool): True if you want to plot graphs during initialization.
            useShortChannelRegression (bool): True if you want to use short channel regression during preprocessing.
        """
        if path == "" or path is None:
            return None
        raw = self.loadData(path)
        self.ploting = isPloting
        self.DoC = isDoC
        self.useShortChannelRegression = useShortChannelRegression
        if preprocessing:
            self.data = self.preprocess(raw)
        else:
            self.data, self.shortChannels = self.removeShortChannels(raw)
            #self.data = raw


    def loadData(self, path: str):
        """
        Loads the snirf file to mne.io.Raw data
 
        Parameters:
            path (str): The absolute path of the fnirs file.
 
        Returns:
            mne.io.Raw: The object with the data.
        """
        raw = mne.io.read_raw_snirf(path, optode_frame="mri")
        raw.load_data()
        return raw

    def preprocess(self, data: mne.io.Raw):
        """
        Aply the preprocessing routine to the data with the raw signals. The routine includes:
            1.  Convert raw to optical density.
            2.  Reject and interpolate bad channels.
            3.  Short channels regression (optional).
            4.  TDDR for the motion artifacts.
            5.  Convert optical density to hemoglobin (ppf=0.05).
            6.  2nd order butterworth bandpass filter [0.01, 0.2]
 
        Parameters:
            data (mne.io.Ra): The mne.io.Raw object with the raw signals.
 
        Returns:
            The mne.io.Raw object with the hemoglobin data.
        """
        # Converting from raw intensity to optical density
        data_od = mne.preprocessing.nirs.optical_density(data)

        # Reject bad channels based on the scalp_coupling_index
        sci = mne.preprocessing.nirs.scalp_coupling_index(data_od)
        data_od.info["bads"] = list(compress(data_od.ch_names, sci < 0.8))
        print("---------------------------------------------------------")
        print("Bad channels: " + str(data_od.info["bads"]))
        print("---------------------------------------------------------")
        if self.ploting:
            fig, ax = plt.subplots(layout="constrained")
            ax.hist(sci)
            ax.set(xlabel="Scalp Coupling Index", ylabel="Count", xlim=[0, 1])
            plt.show()
        
        # Interpolate
        data_od.interpolate_bads()

        # Short Channel Regression
        if self.useShortChannelRegression:
            print("Perforning short channel regression...")
            data_od = mne_nirs.signal_enhancement.short_channel_regression(data_od)
            print("Done!")

        # Remove short channels 
        data_od, self.shortChannels = self.removeShortChannels(data_od)
        print(data_od.info)

        # Remove motion artifacts
        #if not self.DoC:
        data_od_corrected = mne.preprocessing.nirs.temporal_derivative_distribution_repair(data_od)
        #else:
        #data_od_corrected = data_od

        # Convert optical density to hemoglobin
        data_heamo = mne.preprocessing.nirs.beer_lambert_law(data_od_corrected, ppf=6)

        # Physiological noise removal - Bandpass filtering
        iir_params = dict(order=4, ftype='butter')
        data_heamo.filter(l_freq=0.01, h_freq=None, method='iir',
            iir_params=iir_params, verbose=True)
        data_heamo.filter(l_freq=None, h_freq=0.2, method='iir',
            iir_params=iir_params, verbose=True)
        
        # # Reject bad channels with very low or very high power
        # bads = self.reject_channels_based_on_power(data_heamo)
        # for bad in bads:
        #     if bad not in data_heamo.info["bads"]:
        #         data_heamo.info["bads"].append(bad)

        # print(data_heamo.info["bads"])
        # # Interpolate
        # data_heamo.interpolate_bads()
        
        return data_heamo

    def removeShortChannels(self, data: mne.io.Raw):
        """
        Remove the short channels from <data>.
 
        Parameters:
            data (mne.io.Raw): A mne.io.Raw object. 
 
        Returns:
            mne.io.Raw: The mne.io.Raw object without the short channels.
        """
        picks = mne.pick_types(data.info, meg=False, fnirs=True)
        dists = mne.preprocessing.nirs.source_detector_distances(
        data.info, picks=picks
        )
        data.pick(picks[dists > 0.01])
        return data, picks[dists <= 0.01]
    
    def plot(self, show: bool=True, title: str="Hemoglobin Concentration."):
        """
        Plot the signal/time series of each channel of the data.
        """
        self.data.plot(n_channels=len(self.data.ch_names), duration=1000, show_scrollbars=False, title=title, scalings="auto")
        if show:
            plt.show()

    def getShortChannels(self):
        """
        Get the IDs of the short channels.
        """
        return self.shortChannels

    def getMneIoRaw(self):
        """
        Get the mne.io.Raw object with the hemoglobin data.
        """
        return self.data
    
    def reject_channels_based_on_power(self, raw_od):
        bad_channels = []
        data = raw_od.get_data(picks=['hbo'])
        power = np.sum(np.abs(data)**2, axis=1)
        count_greater = [0] * 8
        count_smaller = [0] * 8
        for i,query in enumerate(power):
            print(f"query: {raw_od.ch_names[i]}")
            for j,test in enumerate(power):
                if i == j:
                    continue
                else:
                    if query / test >= 9:
                        print("onces")
                        count_greater[i] += 1
                    elif query / test <= 0.1112:
                        print("-onces")
                        count_smaller[i] += 1

        for i, power in enumerate(count_greater):
            if power >= 7:
                bad_channels.append(raw_od.ch_names[i])
                bad_channels.append(raw_od.ch_names[i].replace("hbo","hbr"))

        for i, power in enumerate(count_smaller):
            if power >= 7:
                bad_channels.append(raw_od.ch_names[i])
                bad_channels.append(raw_od.ch_names[i].replace("hbo","hbr"))

        print(bad_channels)
        return bad_channels





    def getStartTimeOfImagery(self):
        '''
        Return the index of the last Rest Trigger from the array with the triggers.
    
            Parameters:
                event_array (numpy.ndarray): The array that contains all the triggers of the experiment.
    
            Returns:
                int: An integer which represents an index of an array.
        '''
        events, event_dict = mne.events_from_annotations(self.data)
        imageryID = event_dict["Imagery"]
        time = list()
        for idr,row in enumerate(events):
            if row[2] == imageryID:
                time.append(self.indexAsTime(row[0],self.data.info["sfreq"]))
        return time

    def indexAsTime(self, index: int, sample_frequency: float):
        '''
        Convert index to time with respect to the sample frequency.
    
            Parameters:
                index (int): An integer which represents an index on the array with the raw signal.
                sample_frequency (float): The sample frequency that we used.
            Returns:
                float: A float which corresponds to the time.
        '''
        return index / sample_frequency



#hemo = HemoData("L:\\LovbeskyttetMapper\\CONNECT-ME\\DTU\\Alex_Data\\HealthyPatients\\all_\\C19_raw.snirf", preprocessing=True, usePCA=True)
# hemo.plot()
#hemo.getDataByEpochsPairedRealRest()
    
# Satori file
# hemo = HemoData("L:\\LovbeskyttetMapper\\CONNECT-ME\\DTU\\Alex_Data\\Satori\\HealthyPatients\\C8_raw_Satori_od_regression.snirf", preprocessing=False)
# hemo.plot()