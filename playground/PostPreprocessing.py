import os
import sys
import path
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)
import mne
import numpy as np
from Hemo import HemoData
from DataPath import DataPath
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


MAX_LENGTH_HEALTHY = 1224
MAX_LENGTH_DOC = 7760

class PostPreprocess:
      
      def __init__(self, path: str, isDoc: bool= False, standardize: bool=True, isPlot= False) -> None:
            self.datapath = DataPath(path, fileType = "snirf", recursive = False).getDataPaths()
            self.standardize = standardize
            self.plot = isPlot
            self.max_lenght = MAX_LENGTH_HEALTHY
            if isDoc:
                  self.max_lenght = MAX_LENGTH_DOC

            for id, file in enumerate(self.datapath):
                  # Read mne file and apply preprocessing
                  raw_haemo = HemoData(file, preprocessing=True, isPloting=False).getMneIoRaw()
                  raw_data = raw_haemo.get_data(picks=["hbo"]) # channels x samples
                  events, _ = mne.events_from_annotations(raw_haemo)
                  
                  # Apply Post preprocessing routines ( Standardize & PCA )
                  if id == 0:
                        self.data = np.expand_dims(self.individual_process(raw_data, events), axis=0)
                  else:
                        self.data = np.concatenate((self.data, np.expand_dims(self.individual_process(raw_data, events), axis=0)), axis=0)
            
            print(self.data.shape)
       
      def individual_process(self, data, events):
            t_interval = 153 # 15 sec
            
            #Standardize
            if self.standardize:
                  ss = StandardScaler()
                  data = ss.fit_transform(data.T).T

            # Initiate some plot variables
            figure, axis = plt.subplots(4, 1)
            axis[0].set_title("Suplementery Motor Area")
            axis[1].set_title("Frontal Area")
            axis[2].set_title("Motor Area")
            axis[3].set_title("Rear Area")

            # Apply PCA in each brain area
            for area in range(4):
                  pca = PCA(n_components=1)
                  data_tmp = data[2*area:2*area+2,events[0][0]:events[-1][0]+2*t_interval][:,:self.max_lenght]
                  # PCA
                  pca.fit(data_tmp.T)
                  # Sign check 
                  max_id = np.argmax(np.abs(pca.components_))
                  if pca.components_[0,max_id] < 0:
                        pca.components_ = -pca.components_
                        print("Sign flipped.")
                  
                  if area == 0:
                        tmp = pca.transform(data_tmp.T).T
                  else:
                        tmp = np.concatenate((tmp, pca.transform(data_tmp.T).T), axis=0)

                  # Add plots 
                  axis[area].plot(np.arange(self.max_lenght), data_tmp[0], label=f"Channel {2*area}")
                  axis[area].plot(np.arange(self.max_lenght), data_tmp[1], label=f"Channel {2*area+1}")
                  axis[area].plot(np.arange(self.max_lenght), pca.transform(data_tmp.T).T.flatten(), color="g", label="Principal Component")
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

            return tmp
            
      def save(self, destination: str):
            # Save data with numpy array
            np.save(path + "std_pca_heamo", self.data)
            # Save numpy array with events
            #np.save(path + "std_pca_heamo_events", self.events)
            # Save numpy array with original names
            #np.save(path + "std_pca_heamo_map", self.map)
            pass 
          
      def getName(self, file: str):
            return file.split("\\")[-1].replace(".snirf","") 

# def invertDic(dic: dict):
#         '''
#         Helping function to invert the keys and the values of a dictionary.
#         '''
#         inv_dic = {v: k for k, v in dic.items()}
#         return inv_dic

path = "L:\\LovbeskyttetMapper\\CONNECT-ME\\DTU\\Alex_Data\\HealthyPatients\\test\\"
sc = PostPreprocess(path=path, isDoc=False, standardize=True, isPlot=True)

##PLOTS



# data = pca.transform(data_tmp.T).T


# plt.show()