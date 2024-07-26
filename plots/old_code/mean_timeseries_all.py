import os
import sys
import path
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem
from matplotlib.legend_handler import HandlerTuple
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr



def sum_cor(data, patient, area):
        cor = 0
        for id, test_data in enumerate(patient):
            if (id != 2*area) and (id != 2*area + 1):
                cor_tmp, _ = pearsonr(data, test_data)
                cor = cor + cor_tmp 
        return cor

def sum_power_distance(query_power, patient, area):
        cor = 0
        for id, test_data in enumerate(patient):
            if (id != 2*area) and (id != 2*area + 1):
                power = np.sum(np.abs(test_data)**2)
                cor = cor + np.abs(power-query_power)
        return cor

def apply_pca(input):
    for patient_id, patient in enumerate(input):
        #### First Apply PCA to reduce the symmetric channels (frontal. motor, supplementary motor, rear areas)
        for area in range(4):
            pca = PCA(n_components=1)
            data = patient[2*area:2*area+2,:]
            
            # In of very big power difference, we assume that one channels is noisy.
            power_1 = np.sum(np.abs(data[0])**2)
            power_2 = np.sum(np.abs(data[1])**2)
            if power_1 / power_2 >= 9 or power_1 / power_2 <= 0.11112:
                print(f"----------------------Area: {area}------------------------------")
                print(f"Power:      {power_1} VS {power_2}")
                # To Find the noisy channel, we compare their correlation with the rest of the channels
                cor1 = sum_cor(data[0], patient, area)
                cor2 = sum_cor(data[1], patient, area)
                powr_dis_1 = sum_power_distance(power_1, patient, area)
                powr_dis_2 = sum_power_distance(power_2, patient, area)
                print(f"Correlation:      {cor1} VS {cor2}")
                print(f"Power_dist:      {powr_dis_1} VS {powr_dis_2}")
                if cor1 >= cor2:
                    tmp_data = np.expand_dims(data[0], axis=0)
                else:
                    tmp_data = np.expand_dims(data[1], axis=0)
                print(f"----------------------------------------------------------------")
            else:
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
                pca_data_tmp = tmp
            else:
                pca_data_tmp = np.concatenate((pca_data_tmp, tmp), axis=0)
        
        # scaler = StandardScaler()
        # pca_data = scaler.fit_transform(pca_data.T).T

        if patient_id == 0: 
            pca_data = np.expand_dims(pca_data_tmp, axis=0)
        else:
            pca_data = np.concatenate((pca_data, np.expand_dims(pca_data_tmp, axis=0)), axis=0)

    return pca_data

LEN_HEALTHY_ICU = 1224
#LEN_DOC = 2360

healthy = np.load("L:\\LovbeskyttetMapper\\CONNECT-ME\\DTU\\Alex_Data\\HealthyPatients\\data_initial\\data.npy") # 30 x 8 x 1224
icu = np.load("L:\\LovbeskyttetMapper\\CONNECT-ME\\DTU\\Alex_Data\\ICUPatients\\data_initial\\data.npy")  # 29 x 8 x 1224
#doc = np.load("L:\\LovbeskyttetMapper\\CONNECT-ME\\DTU\\Alex_Data\\DoC\\data_initial\\hemo_chnl_norm.npy") # 29 x 3 x 8 x 2360

# for i,patient in enumerate(healthy):
#     scaler = StandardScaler()
#     healthy[i] = scaler.fit_transform(patient.T).T


pca_data = apply_pca(healthy)
for i,patient in enumerate(pca_data):
    scaler = StandardScaler()
    if i ==0:
        pca_normalized = np.expand_dims(scaler.fit_transform(patient.T).T, axis=0)
    else:
        pca_normalized = np.concatenate((pca_normalized, np.expand_dims(scaler.fit_transform(patient.T).T, axis=0)), axis=0)
pca_data_mean = np.mean(pca_data, axis=0)
pca_normalized_mean = np.mean(pca_normalized, axis=0)
healthy_mean = np.mean(healthy, axis=0)
figure, axis = plt.subplots(4, 1)
figure.suptitle("Average principal components over all healthy controls and standard error of the mean. The red line is the average PC if we have applied normalization first.")
axis[0].set_title("Suplementery Motor Area")
axis[1].set_title("Frontal Area")
axis[2].set_title("Motor Area")
axis[3].set_title("Rear Area")
for id, _ in enumerate(healthy_mean):
    if id % 2 == 0:
        p2, p3, p4= [], [], []
        p6, = axis[id//2].plot(np.arange(LEN_HEALTHY_ICU), pca_data_mean[id//2],color='green')
        axis[id//2].fill_between(np.arange(LEN_HEALTHY_ICU), pca_data_mean[id//2], pca_data_mean[id//2] - sem(pca_data[:,id//2,:], axis = 0, ddof = 0),color='gray', alpha=0.2)
        twin_axis = axis[id//2].twinx()
        
        color = 'tab:red'
        twin_axis.set_ylabel('normalized scale', color=color)  # we already handled the x-label with ax1
        p7, = twin_axis.plot(np.arange(LEN_HEALTHY_ICU), pca_normalized_mean[id//2], color=color)
        twin_axis.fill_between(np.arange(LEN_HEALTHY_ICU), pca_normalized_mean[id//2], pca_normalized_mean[id//2] - sem(pca_normalized[:,id//2,:], axis = 0, ddof = 0),color='gray', alpha=0.2)
        for idx in range(8):
            aggrated_time_of_event = idx*153 # Every 15 sec
            if idx == 0:
                tmp = axis[id//2].axvline(x = aggrated_time_of_event, color = 'y')
                p2.append(tmp)
            elif idx in [1,3,5,7]:
                tmp = axis[id//2].axvline(x = aggrated_time_of_event, color = 'c')
                p3.append(tmp)
            elif idx in [2,4,6]:
                tmp = axis[id//2].axvline(x = aggrated_time_of_event, color = 'g')
                p4.append(tmp)
        l = axis[id//2].legend([p2[0], (p3[0], p3[1], p3[2], p3[3]), (p4[0], p4[1], p4[2]), p6, p7], 
                               ['Physical Trigger', 'Rest Trigger', 'Imagery Trigger', "Principal Component", "Principal Component Normalized"], handler_map={tuple: HandlerTuple(ndivide=None)}, loc='right')
figure.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


# icu_pca_data = apply_pca(icu)
# print(icu_pca_data.shape)
# # for i,patient in enumerate(icu_pca_data):
# #     scaler = StandardScaler()
# #     icu_pca_data[i] = scaler.fit_transform(patient.T).T
# icu_pca_data = np.mean(icu_pca_data, axis=0)
# icu_mean = np.mean(icu, axis=0)
# for id, _ in enumerate(icu_mean):
#     if id % 2 == 0:
#         plt.figure()
#         p2, p3, p4= [], [], []
#         p7, = plt.plot(np.arange(LEN_HEALTHY_ICU), icu_pca_data[id//2],color='green')
#         p1, = plt.plot(np.arange(LEN_HEALTHY_ICU), icu_mean[id],color='blue')
#         plt.fill_between(np.arange(LEN_HEALTHY_ICU), icu_mean[id], icu_mean[id] - sem(healthy[:,id,:], axis = 0, ddof = 0),color='gray', alpha=0.2)
#         p5, = plt.plot(np.arange(LEN_HEALTHY_ICU), icu_mean[id+1],color='red')
#         plt.fill_between(np.arange(LEN_HEALTHY_ICU), icu_mean[id+1], icu_mean[id+1] - sem(healthy[:,id+1,:], axis = 0, ddof = 0),color='gray', alpha=0.2)
#         for idx in range(8):
#             aggrated_time_of_event = idx*153 # Every 15 sec
#             if idx == 0:
#                 tmp = plt.axvline(x = aggrated_time_of_event, color = 'y')
#                 p2.append(tmp)
#             elif idx in [1,3,5,7]:
#                 tmp = plt.axvline(x = aggrated_time_of_event, color = 'c')
#                 p3.append(tmp)
#             elif idx in [2,4,6]:
#                 tmp = plt.axvline(x = aggrated_time_of_event, color = 'g')
#                 p4.append(tmp)
#         l = plt.legend([p1, p2[0], (p3[0], p3[1], p3[2], p3[3]), (p4[0], p4[1], p4[2]), p5, p7], [f'Channel {id+1}', 'Physical Trigger', 'Rest Trigger', 'Imagery Trigger', f'Channel {id+2}', "Principal Component"], handler_map={tuple: HandlerTuple(ndivide=None)})
#         plt.title("Standardize mean hemoglobin response and standard error of the mean. (ICU patients)")
# plt.show()

# doc_mean = np.mean(doc, axis=0)
# print(doc_mean.shape)
# for trial_id,trial in enumerate(doc_mean):
#     print(trial.shape)
#     for channel_id, _ in enumerate(trial):
#         if channel_id % 2 == 0:
#             plt.figure()
#             p2, p3, p4= [], [], []
#             p1, = plt.plot(np.arange(LEN_DOC), trial[channel_id],color='blue')
#             plt.fill_between(np.arange(LEN_DOC), trial[channel_id], trial[channel_id] - sem(doc[:,trial_id,channel_id,:], axis = 0, ddof = 0),color='gray', alpha=0.2)
#             p5, = plt.plot(np.arange(LEN_DOC), trial[channel_id+1],color='red')
#             plt.fill_between(np.arange(LEN_DOC), trial[channel_id+1], trial[channel_id+1] - sem(doc[:,trial_id,channel_id+1,:], axis = 0, ddof = 0),color='gray', alpha=0.2)
#             # Add trigers to the plot
#             for idx in range(8):
#                 aggrated_time_of_event = idx*295 # Every 29 sec
#                 tmp = plt.axvline(x = aggrated_time_of_event, color = 'g')
#                 p2.append(tmp)
#                 tmp = plt.axvline(x = aggrated_time_of_event + 153, color = 'y')
#                 p3.append(tmp)
#             # Add legends
#             l = plt.legend([p1, tuple(p2), tuple(p3), p5], [f'Channel {channel_id+1}', 'Motor Trigger', 'Rest Trigger', f'Channel {channel_id+2}'], handler_map={tuple: HandlerTuple(ndivide=None)})
#             plt.title(f"Standardize mean hemoglobin response and standard error of the mean. (DoC patients trial_{trial_id})")
#     plt.show()