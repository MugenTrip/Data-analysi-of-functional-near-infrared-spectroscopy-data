import mne
import numpy as np
from Hemo import HemoData
from DataPath import DataPath
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#from pca import PCA
from scipy.stats import pearsonr


def sum_cor(data, patient, area):
      cor = 0
      for id, test_data in enumerate(patient):
            if (id != 2*area) and (id != 2*area + 1):
                  cor_tmp, _ = pearsonr(data, test_data)
                  cor = cor + cor_tmp 
      return cor

def getName(file):
      return file.split("\\")[-1].replace(".snirf","") 

def invertDic(dic: dict):
      '''
      Helping function to invert the keys and the values of a dictionary.
      '''
      inv_dic = {v: k for k, v in dic.items()}
      return inv_dic

path = "L:\\LovbeskyttetMapper\\CONNECT-ME\\DTU\\Alex_Data\\HealthyPatients\\data_initial\\"
datapath = DataPath(path, recursive=False)
print(len(datapath.getDataPaths()))
MAX_LENGTH_HEALTHY = 1224
MAX_LENGTH_DOC = 7760

for id,file in enumerate(datapath.getDataPaths()):
      raw_haemo = HemoData(file, preprocessing=True, isPloting=False)#.getMneIoRaw()
      if getName(file) == "C18_raw":
            raw_haemo.plot()
            raw_data = raw_haemo.getMneIoRaw().get_data(picks=["hbo"])

            t_interval = 153 # 15 sec
            events, event_dict = mne.events_from_annotations(raw_haemo.getMneIoRaw())
            reverse_event_dict = invertDic(event_dict)

      # print(raw_data.shape)
      # raw_data = raw_data - np.expand_dims(np.mean(raw_data, axis=1), axis=1)
      # print(raw_data.shape)

      #Standardize
      #print(np.std(raw_data, axis=1))
      # ss = StandardScaler()
      # raw_data = ss.fit_transform(raw_data.T).T
      # minn = np.expand_dims(np.min(raw_data, axis=1), axis=1)
      # maxx = np.expand_dims(np.max(raw_data, axis=1), axis=1)
      # raw_data = (raw_data - minn) / (maxx - minn)
      # print(np.std(raw_data, axis=1))
      # print(np.std(raw_data[0]))

      

      # Save data with numpy array
      #print(raw_data.shape)
      #print(raw_data[:,events[0][0]:events[-1][0]+2*t_interval].shape)
      #np.save(path + "clean_hemo_data-" + getName(file), raw_data)
      #Save numpy array with events
      #np.save(path + "hemo_events-" + getName(file), np.array(events))  

            # plt.plot(np.arange(raw_data.shape[1]), raw_data[0])
            # plt.plot(np.arange(raw_data[:,events[0][0]:events[-1][0]+2*t_interval].shape[1]) + events[0][0], raw_data[:,events[0][0]:events[-1][0]+2*t_interval][0])
            # plt.show()

            figure, axis = plt.subplots(4, 1)
            axis[0].set_title("Suplementery Motor Area")
            axis[1].set_title("Frontal Area")
            axis[2].set_title("Motor Area")
            axis[3].set_title("Rear Area")
            for area in range(4):
                  once = [False, False, False]
                  for idx, event in enumerate(events):
                        aggrated_time_of_event = idx*153 # Every 15 sec
                        if idx == 0:
                              if once[0] is False:
                                    axis[area].axvline(x = event[0], color = 'y', label="Physical trigger")
                                    once[0] = True
                              else:
                                    axis[area].axvline(x = event[0], color = 'y')
                              #p2.append(tmp)
                        elif idx in [1,3,5,7]:
                              if once[1] is False:
                                    axis[area].axvline(x = event[0], color = 'c', label="Imagery trigger")
                                    once[1] = True
                              else:
                                    axis[area].axvline(x = event[0], color = 'c')
                              #p3.append(tmp)
                        elif idx in [2,4,6]:
                              if once[2] is False:
                                    axis[area].axvline(x = event[0], color = 'g', label="Rest trigger")
                                    once[2] = True
                              else:
                                    axis[area].axvline(x = event[0], color = 'g')
                              #p4.append(tmp)

                  pca = PCA(n_components=1)
                  data_tmp = raw_data[2*area:2*area+2,:]
                  print(np.std(data_tmp[0]))
                  #print(data_tmp.shape)
                  # Calculate power difference
                  power_1 = np.sum(np.abs(data_tmp[0])**2)
                  power_2 = np.sum(np.abs(data_tmp[1])**2)
                  #print(f"      {power_1} VS {power_2}")
                  if power_1 / power_2 >= 9 or power_1 / power_2 <= 0.11112:
                        print(f"----------------------Area: {area}------------------------------")
                        print(f"Power:      {power_1} VS {power_2}")
                        # To Find the noisy channel, we compare their correlation with the rest of the channels
                        cor1 = sum_cor(data_tmp[0], raw_data, area)
                        cor2 = sum_cor(data_tmp[1], raw_data, area)
                        print(f"Correlation:      {cor1} VS {cor2}")
                        if cor1 >= cor2:
                              tmp_data = np.expand_dims(data_tmp[0], axis=0)
                        else:
                              tmp_data = np.expand_dims(data_tmp[1], axis=0)
                        print(f"----------------------------------------------------------------")
                  else:
                        tmp_data = data_tmp
                  # PCA
                  pca.fit(tmp_data.T)
                  # Sign check 
                  max_id = np.argmax(np.abs(pca.components_))
                  if pca.components_[0,max_id] < 0:
                        pca.components_ = -pca.components_
                  print(pca.components_)
                  #print("")
                  #print(pca.get_covariance())
                  
                  data = pca.transform(tmp_data.T).T
                  # axis[area].plot(np.arange(raw_data.shape[1]), tmp_data[0], label=f"Channel {2*area}")
                  # if tmp_data.shape[0] == 2:
                  #       axis[area].plot(np.arange(raw_data.shape[1]), tmp_data[1], label=f"Channel {2*area+1}")

                  ss = StandardScaler()
                  data = ss.fit_transform(data.T).T
                  axis[area].plot(np.arange(raw_data.shape[1]), data.flatten(), color="g", label="Principal Component")
                  axis[area].legend(loc='right')
                  
                  # if area== 0:
                  #       axis[area].text(-0.1 , 2.5, f'weights: {pca.components_[0]}', fontsize = 13)
                  # elif area== 1:
                  #       axis[area].text(-0.1 , 4, f'weights: {pca.components_[0]}', fontsize = 13)
                  # elif area== 2:
                  #       axis[area].text(-0.1 , 3, f'weights: {pca.components_[0]}', fontsize = 13)
                  # else:
                  #       axis[area].text(-0.1 , 3, f'weights: {pca.components_[0]}', fontsize = 13)
            
            plt.show()