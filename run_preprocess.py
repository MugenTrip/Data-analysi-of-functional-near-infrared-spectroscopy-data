import mne
import numpy as np
from Hemo import HemoData
from DataPath import DataPath
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import argparse

MAX_LENGTH_HEALTHY = 1224
MAX_LENGTH_DOC = 7756
BASE_PATH = "L:\\LovbeskyttetMapper\\CONNECT-ME\\DTU\\Alex_Data\\"

def getName(file):
       return file.split("\\")[-1].replace(".snirf","") 

def invertDic(dic: dict):
        '''
        Helping function to invert the keys and the values of a dictionary.
        '''
        inv_dic = {v: k for k, v in dic.items()}
        return inv_dic

def preprocess(datapath: DataPath, max_length: int):
    for id,file in enumerate(datapath.getDataPaths()):
        raw_haemo = HemoData(file, preprocessing=True, isPloting=False).getMneIoRaw()
        raw_data = raw_haemo.get_data(picks=["hbo"]) # channels x samples

        t_interval = 153 # 15 sec
        events, event_dict = mne.events_from_annotations(raw_haemo)
        reverse_event_dict = invertDic(event_dict)

        # Save map with real patient names
        if id == 0:
                patient_map = np.expand_dims(np.array([getName(file=file)]), axis=0)
        else:
                patient_map = np.concatenate((patient_map, np.expand_dims(np.array([getName(file=file)]), axis=0)), axis=0)

        # Simple version Subjects X channels X samples. In our case: 30 healthy controls X 8 channels X 1124
        if id == 0:
                data = np.expand_dims(raw_data[:,events[0][0]:events[-1][0]+2*t_interval][:,:max_length], axis=0)
        else:
                data = np.concatenate((data, np.expand_dims(raw_data[:,events[0][0]:events[-1][0]+2*t_interval][:,:max_length],axis=0)),axis=0)

        # Save event list for all patients
        events[:,0] = events[:,0] - events[0,0]  # Re-center events, since we cropped the intro
        if id == 0:
                labels = np.expand_dims(events, axis=0)
        else:
                labels = np.concatenate((labels, np.expand_dims(events,axis=0)),axis=0)

    return data, labels, patient_map

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d" , "--datatype", type=str, choices=["healthy", "icu", "doc"], help="Determine the data type, could be [healthy, icu, doc].", required=True)
    parser.add_argument("-s", "--session", type=str, choices=["initial", "followup", "test"], help="Session, could be initial or followup.", required=True)
    args = parser.parse_args()

    path = None
    if args.datatype == "healthy":
            path = BASE_PATH + "HealthyPatients\\"
            max_length = MAX_LENGTH_HEALTHY
    elif args.datatype == "icu":
            path = BASE_PATH + "ICUPatients\\"
            max_length = MAX_LENGTH_HEALTHY
    elif args.datatype == "doc":
            path = BASE_PATH + "DoC\\"
            max_length = MAX_LENGTH_DOC

    if args.session == "test":
            path = path + "test\\"
    if args.session == "initial":
            path = path + "data_initial\\"
    if args.session == "followup":
            path = path + "data_followup\\"

    datapath = DataPath(path, recursive=False)
    data, labels, patient_map = preprocess(datapath, max_length)
    np.save(path + "data", data)
    np.save(path + "events", labels)
    np.save(path + "map", patient_map)
    