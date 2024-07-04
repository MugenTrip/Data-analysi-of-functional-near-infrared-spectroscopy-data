from DataPath import DataPath
import mne
from Hemo import HemoData
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import math
from DataLoader import DataLoader
from sklearn import svm
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from statistics import mean, stdev
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c" , "--channels", type=list, 
                    help="Define the channels you want to use like 0123 for all channels. (O: SMA, 1: Frontal, 2: Motor, 3: Parietal)", required=True)
args = parser.parse_args()
if args.channels is None:
    channels = [0,1,2,3]
else:
    channels = [int(x) for x in args.channels]

stimuli_duration = 153
total_task_duration = 295

def getName(file):
    return file.split("\\")[-1].replace(".snirf","")

def invertDic(dic: dict):
    '''
    Helping function to invert the keys and the values of a dictionary.
    '''
    inv_dic = {v: k for k, v in dic.items()}
    return inv_dic

def slopePerChannel(data):
    '''
    data: n_channels X samples
    '''
    # Calculate slope for each event for each channel: events X channels X 1 slope
    return np.polyfit(np.arange(data.T.shape[0]), data.T,1)[0]

def extractFeatures(patient, start, stop):
        feature_window = 53 # 5 sec
        # For every event
        for iid, event in enumerate(patient):
            if iid == 0:
                tmp = np.expand_dims(slopePerChannel(event[:,start:stop]), axis=0)
            else:
                tmp = np.concatenate((tmp, np.expand_dims(slopePerChannel(event[:,start:stop]), axis=0)), axis=0)
        
        return tuple(
                        (
                         np.expand_dims(tmp, axis=2), 
                         np.expand_dims(np.mean(patient[:,:,stop-feature_window:stop], axis=2), axis=2)
                        )
                    )

def getSpecialCaseFeature(file):
    raw = HemoData(file, preprocessing=True, isDoC=False, isPloting=False)
    raw_haemo = raw.getMneIoRaw()
    raw_data = raw_haemo.get_data(picks=["hbo"]) # channels x samples

    t_interval = 295 # 29 sec
    events, event_dict = mne.events_from_annotations(raw_haemo)
    reverse_event_dict = invertDic(event_dict)

    # Use PCA to reduce the symetrical channels (frontal. motor, supplementary motor, rear areas)
    #print(normalized_data)    
    for area in range(4):
        pca = PCA(n_components=1)
        data = raw_data[2*area:2*area+2,:]
        
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

        scaler = StandardScaler()
        pca_data = scaler.fit_transform(pca_data.T).T

    t_interval = 295 # 29 sec
    events, event_dict = mne.events_from_annotations(raw_haemo)

    # Take each event separetly for each experiment (healthy control/DoC patient). Ends up experiments X events X channels X samples.
    # Look over all events (e.g. Rest, Tongue Motor, Imagery Motor)
    for idx,event in enumerate(events):
            if idx == 0:
                    event_data = np.expand_dims(pca_data[:,event[0]:int(event[0]+t_interval)], axis=0)
            elif idx == len(events)-1:
                   # Drop last. Usually it is croped.
                   continue
            else:
                    event_data = np.concatenate((event_data, np.expand_dims(pca_data[:,event[0]:int(event[0]+t_interval)], axis=0)),axis=0)

    start = 0
    duration = 153
    end = 295
    features_physical = np.concatenate(extractFeatures(event_data, start, start+duration), axis=2)
    features_rest = np.concatenate(extractFeatures(event_data, start+duration, start+end), axis=2)
    features = features_physical - features_rest # Patients X Events X Areas X Features

    print(features_physical.shape)
    print(features_rest.shape)

    separetor = math.floor(features.shape[0]/2)
    features[separetor:,:] = -features[separetor:,:]
    labels = [0]*math.floor(features.shape[0]/2) + [1]*math.ceil(features.shape[0]/2)

    #print(np.array(labels).shape)
    #print(features)
    return features[:,channels,:], labels


#####################################################################################################################
#### Run on Special case DOC
path = os.path.join(os.path.curdir, 'data\doc\special_cases')
datapath = DataPath(path, recursive=False)
print(datapath.getDataPaths())
# ev = []

acc = {}
patient_mean = []
patient_std = []
for id,file in enumerate(datapath.getDataPaths()):
    patient, patient_labels =  getSpecialCaseFeature(file)
    patient_labels = np.expand_dims(np.array(patient_labels), axis=1)
    lst_accu_stratified = []
    skf = LeaveOneOut()
    patient = patient.reshape(patient.shape[0], patient.shape[2]*patient.shape[1])

    for train_index, test_index in skf.split(patient, patient_labels):
        x_train_fold, x_test_fold = patient[train_index], patient[test_index]
        y_train_fold, y_test_fold = patient_labels[train_index], patient_labels[test_index]

        # Standardize
        sc = StandardScaler()
        x_train_fold = sc.fit_transform(x_train_fold)
        x_test_fold = sc.transform(x_test_fold)

        svm_clf = svm.SVC(kernel='linear') # Linear Kernel
        # train the model
        svm_clf.fit(x_train_fold, y_train_fold.ravel())

        # valiidate model
        svm_clf_pred = svm_clf.predict(x_test_fold)
        lst_accu_stratified.append(svm_clf.score(x_test_fold, y_test_fold))

    patient_mean.append(mean(lst_accu_stratified))
    patient_std.append(stdev(lst_accu_stratified))

print(np.argmax(patient_mean))
print(mean(patient_mean))
for i, patient in enumerate(datapath.getDataPaths()):
    print(f"{patient} accuracy: {patient_mean[i]}")
