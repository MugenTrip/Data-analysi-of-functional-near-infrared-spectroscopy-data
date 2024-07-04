import math
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA
from scipy.stats import pearsonr



class DataLoader():
    """
    Class to extract the features and prepare the classes for the classificaton task.
    """
    def __init__(self, data_path: str, event_path: str, isDoc: bool=False, plot: bool=False, isPCA: bool=True, isNormalize: bool=True) -> None:
        """
        Initialiaze the class
            Args:
            data_path (str):        Path of the data.npy file. Use run_preprocess.py to create it.
            event_path (int):       Path of the events.npy file. Use run_preprocess.py to create it.
            isDoc (bool):           True is you need the features for a DOC patient. Default (false)
            plot (bool):            True is you want to get plots. Default (false)
            isPCA (bool):           True is you want to apply PCA and reduce the channels/features. Default (true)
            isNormalize (bool):     True is you want per channel normalization. Default (true)
        """
        if (data_path == "" or data_path is None) or (event_path == "" or event_path is None):
            print("Specify the path of the raw data and the events.")
            return None
        
        self.isDoc = isDoc
        self.data = np.load(data_path) # patients X channels X samples
        self.events = np.load(event_path) # patients X events X 3
        self.pca_data = None
        self.plot = plot
        self.isPCA = isPCA
        self.isNormalize = isNormalize
        if self.isPCA:
            self.load_pca_data()
        else:
            for patient_id, patient in enumerate(self.data):
                scaler = StandardScaler()
                patient = scaler.fit_transform(patient.T).T
                if patient_id == 0:
                    self.pca_data = np.expand_dims(patient,axis=0)
                else:
                    self.pca_data = np.concatenate((self.pca_data, np.expand_dims(patient,axis=0)), axis=0)
                
                # This will overdide the per channel normalization
                if not isNormalize:
                    self.pca_data = self.data


    def load_pca_data(self): 
        '''
        Apply PCA in each symmetric left-right channel to obtain only one for each area.
        ''' 
        for patient_id, patient in enumerate(self.data):
            # Initiate some plot variables
            if self.plot:
                figure, axis = plt.subplots(4, 1)
                axis[0].set_title("Suplementery Motor Area")
                axis[1].set_title("Frontal Area")
                axis[2].set_title("Motor Area")
                axis[3].set_title("Rear Area")

            #### First Apply PCA to reduce the symmetric channels (frontal. motor, supplementary motor, rear areas)
            for area in range(4):
                pca = PCA(n_components=1)
                data = patient[2*area:2*area+2,:]
                
                tmp_data = data
                                    
                # Apply PCA
                pca.fit(tmp_data.T)
                # Sign check 
                max_id = np.argmax(np.abs(pca.components_))
                if pca.components_[0,max_id] < 0:
                    pca.components_ = -pca.components_
                #print(pca.components_)
                
                tmp = pca.transform(tmp_data.T).T
                if area == 0:
                    pca_data = tmp
                else:
                    pca_data = np.concatenate((pca_data, tmp), axis=0)
                
                # Add plots 
                if self.plot:
                    axis[area].plot(np.arange(data[0].shape[0]), data[0], label=f"Channel {2*area}")
                    if data.shape[0] == 2:
                        axis[area].plot(np.arange(data[1].shape[0]), data[1], label=f"Channel {2*area+1}")
                    axis[area].plot(np.arange(data[0].shape[0]), tmp.flatten(), color="g", label="Principal Component")
                    axis[area].legend(loc='right')
                    if area== 0:
                        axis[area].text(-0.1 , 2.5, f'weights: {pca.components_[0]}', fontsize = 13)
                    elif area== 1:
                        axis[area].text(-0.1 , 4, f'weights: {pca.components_[0]}', fontsize = 13)
                    elif area== 2:
                        axis[area].text(-0.1 , 3, f'weights: {pca.components_[0]}', fontsize = 13)
                    else:
                        axis[area].text(-0.1 , 3, f'weights: {pca.components_[0]}', fontsize = 13)
        
            if self.plot:
                plt.show()
            
            if  self.isNormalize:
                scaler = StandardScaler()
                pca_data = scaler.fit_transform(pca_data.T).T

            if patient_id == 0: 
                self.pca_data = np.expand_dims(pca_data, axis=0)
            else:
                self.pca_data = np.concatenate((self.pca_data, np.expand_dims(pca_data, axis=0)), axis=0)
 
    def getFeatures(self, feature_window: int= 53, channels: list=[0,1,2,3]):
        '''
        Get all the data in the form Patients X Events X Features. And the label/order of the events Patients X Labels.
        '''
        if not self.isPCA:
            return self.getFeaturesNoPCA(feature_window, [0,1,2,3,4,5,6,7])
        if self.isDoc:
            return self.getFeaturesDoc(feature_window, channels) # patients X trials X channels X features
        else:
            return self.getFeaturesNormal(feature_window, channels) # patients X channels X features
        
    def getFeaturesImagery(self, feature_window: int= 53, channels: list=[0,1,2,3]):
        '''
        Get all the data-points and labels for the imagery task.
        '''
        start = 0
        duration = 153
        end = 306
        for patient_id, patient in enumerate(self.pca_data):
            event_features = None
            for event_id, event in enumerate(self.events[patient_id]):
                if event[2] == 1:
                    features_imagery = self.extractFeaturesDoC(patient,event[0]+start, event[0]+start+duration)
                    features_rest = self.extractFeaturesDoC(patient,event[0]+start+duration, event[0]+start+end)
                    
                    if event_features is None:
                        event_features = np.expand_dims(features_imagery - features_rest, axis=0)
                    else:
                        event_features = np.concatenate((event_features, np.expand_dims(features_imagery - features_rest, axis=0)), axis=0)

            if patient_id == 0:
                patient_features = np.expand_dims(event_features, axis=0)
            else:
                patient_features = np.concatenate((patient_features, np.expand_dims(event_features, axis=0)), axis=0)

        patient_features = patient_features.reshape(patient_features.shape[0]*patient_features.shape[1], patient_features.shape[2], patient_features.shape[3]) # 30 (patients) X 3 (events) X 4 (channels) X 2 (features) ----> 90 (datapoints) X 4 X 2  
        separetor = math.floor(patient_features.shape[0]/2)
        patient_features[separetor:,:,:] = -patient_features[separetor:,:,:]
        labels = [0]*math.floor(patient_features.shape[0]/2) + [1]*math.ceil(patient_features.shape[0]/2)

        return patient_features[:,channels,:], np.expand_dims(np.array(labels), axis=1)
    
    def getFeaturesImageryNoPCA(self, feature_window: int= 53, channels: list=[0,1,2,3]):
        '''
        Similar to getFeaturesImagery, but it does not revert half of the datapoints. It was used for plots.
        '''
        start = 0
        duration = 153
        end = 306
        for patient_id, patient in enumerate(self.pca_data):
            event_features_imagery = None
            event_features_rest = None
            for event_id, event in enumerate(self.events[patient_id]):
                if event[2] == 1:
                    features_imagery = self.extractFeaturesDoC(patient,event[0]+start, event[0]+start+duration)
                    features_rest = self.extractFeaturesDoC(patient,event[0]+start+duration, event[0]+start+end)
                    
                    if event_features_imagery is None:
                        event_features_imagery = np.expand_dims(features_imagery, axis=0)
                        event_features_rest = np.expand_dims(features_rest, axis=0)
                    else:
                        event_features_imagery = np.concatenate((event_features_imagery, np.expand_dims(features_imagery, axis=0)), axis=0)
                        event_features_rest = np.concatenate((event_features_rest, np.expand_dims(features_rest, axis=0)), axis=0)

            if patient_id == 0:
                patient_features_imagery = np.expand_dims(event_features_imagery, axis=0)
                patient_features_rest = np.expand_dims(event_features_rest, axis=0)
            else:
                patient_features_imagery = np.concatenate((patient_features_imagery, np.expand_dims(event_features_imagery, axis=0)), axis=0)
                patient_features_rest = np.concatenate((patient_features_rest, np.expand_dims(event_features_rest, axis=0)), axis=0)

        return patient_features_imagery - patient_features_rest

    def getFeaturesImageryNoPCATtest(self, feature_window: int= 53, channels: list=[0,1,2,3]):
        '''
        Similar to getFeaturesImageryNoPCA, but it does not pair the featutes. It was used for the t-test.
        '''
        start = 0
        duration = 153
        end = 306
        for patient_id, patient in enumerate(self.pca_data):
            event_features_imagery = None
            event_features_rest = None
            for event_id, event in enumerate(self.events[patient_id]):
                if event[2] == 1:
                    features_imagery = self.extractFeaturesDoC(patient,event[0]+start, event[0]+start+duration)
                    features_rest = self.extractFeaturesDoC(patient,event[0]+start+duration, event[0]+start+end)
                    
                    if event_features_imagery is None:
                        event_features_imagery = np.expand_dims(features_imagery, axis=0)
                        event_features_rest = np.expand_dims(features_rest, axis=0)
                    else:
                        event_features_imagery = np.concatenate((event_features_imagery, np.expand_dims(features_imagery, axis=0)), axis=0)
                        event_features_rest = np.concatenate((event_features_rest, np.expand_dims(features_rest, axis=0)), axis=0)

            if patient_id == 0:
                patient_features_imagery = np.expand_dims(event_features_imagery, axis=0)
                patient_features_rest = np.expand_dims(event_features_rest, axis=0)
            else:
                patient_features_imagery = np.concatenate((patient_features_imagery, np.expand_dims(event_features_imagery, axis=0)), axis=0)
                patient_features_rest = np.concatenate((patient_features_rest, np.expand_dims(event_features_rest, axis=0)), axis=0)

        return patient_features_imagery , patient_features_rest

    def getFeaturesNoPCA(self, feature_window: int= 53, channels: list=[0,1,2,3]):
        '''
        Get the features of the PHYSICAL task, separetely for the stimuli and the rest periods. It was used fot t-test.
        '''
        start = 0
        duration = 153
        end = 306
        features_physical = np.concatenate(self.extractFeatures(start, start+duration, feature_window), axis=2)
        features_rest = np.concatenate(self.extractFeatures(start+duration, start+end, feature_window), axis=2)
        return features_physical, features_rest

    def getFeaturesNormal(self, feature_window: int= 53, channels: list=[0,1,2,3]):
        '''
        Get the paired features and the labels for the PHYSICAL task.
        '''
        start = 0
        duration = 153
        end = 306
        features_physical = np.concatenate(self.extractFeatures(start, start+duration, feature_window), axis=2)
        features_rest = np.concatenate(self.extractFeatures(start+duration, start+end, feature_window), axis=2)
        self.features = features_physical - features_rest

        separetor = math.floor(self.features.shape[0]/2)
        self.features[separetor:,:,:] = -self.features[separetor:,:,:]
        labels = [0]*math.floor(self.features.shape[0]/2) + [1]*math.ceil(self.features.shape[0]/2)

        return self.features[:,channels,:], np.expand_dims(np.array(labels), axis=1)
    
    def getFeaturesDoc(self, feature_window: int= 53, channels: list=[0,1,2,3]):
        '''
        Get the paired features and the labels for the DOC patients.
        '''
        start = 0
        duration = 153
        end = 295
        for patient_id, patient in enumerate(self.pca_data):
            for event_id, event in enumerate(self.events[patient_id]):
                features_physical = self.extractFeaturesDoC(patient,event[0]+start, event[0]+start+duration)
                features_rest = self.extractFeaturesDoC(patient,event[0]+start+duration, event[0]+start+end)
                
                if event_id == 0:
                    event_features = np.expand_dims(features_physical - features_rest, axis=0)
                else:
                    event_features = np.concatenate((event_features, np.expand_dims(features_physical - features_rest, axis=0)), axis=0)

            if patient_id == 0:
                patient_features = np.expand_dims(event_features, axis=0)
            else:
                patient_features = np.concatenate((patient_features, np.expand_dims(event_features, axis=0)), axis=0)

        separetor = math.floor(patient_features.shape[1]/2)
        patient_features[:,separetor:,:,:] = -patient_features[:,separetor:,:,:]
        labels = [0]*math.floor(patient_features.shape[1]/2) + [1]*math.ceil(patient_features.shape[1]/2)

        return patient_features[:,:,channels,:], np.expand_dims(np.array(labels), axis=1)

    def extractFeatures(self, start, stop, feature_window: int= 53):
        '''
        Method used to extract the mean and the slope for non DOC participants. You can add more featutes in the tuple if you want.
        '''
        # For the slope we need to do this for every patient seperately
        for id, patient in enumerate(self.pca_data):
            if id == 0:
                slope = np.expand_dims(self.slopePerChannel(patient[:,start:stop]), axis=0)
            else:
                slope = np.concatenate((slope, np.expand_dims(self.slopePerChannel(patient[:,start:stop]), axis=0)), axis=0)
        
        return tuple(
                        (
                         np.expand_dims(slope, axis=2), 
                         #np.expand_dims(np.mean(self.pca_data[:,:,start:start+feature_window], axis=2), axis=2), 
                         np.expand_dims(np.mean(self.pca_data[:,:,stop-feature_window:stop], axis=2), axis=2)
                        )
                    )
    
    def extractFeaturesDoC(self, data, start, stop, feature_window: int= 53):
        '''
        Method used to extract the mean and the slope for DOC participants. You can add more featutes in the tuple if you want.
        '''
        # For the slope we need to do this for every patient seperately
        slope = np.expand_dims(self.slopePerChannel(data[:,start:stop]), axis=0)
        return np.concatenate(
                        (
                         slope.reshape(slope.shape[1],1), 
                         #np.expand_dims(np.mean(self.pca_data[:,:,start:start+feature_window], axis=2), axis=2), 
                         np.expand_dims(np.mean(data[:,stop-feature_window:stop], axis=1), axis=1)
                        ),
                        axis = 1
                    )
    
    def slopePerChannel(self, data):
        '''
        Calculate the slope for each channel of the data array.
        data: n_channels X samples
        '''
        # Calculate slope for each event for each channel: events X channels X 1 (slope)
        return np.polyfit(np.arange(data.T.shape[0]), data.T,1)[0]
