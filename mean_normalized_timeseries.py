import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem
from matplotlib.legend_handler import HandlerTuple
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

LEN_HEALTHY_ICU = 1224
#LEN_DOC = 2360

healthy = np.load("L:\\LovbeskyttetMapper\\CONNECT-ME\\DTU\\Alex_Data\\HealthyPatients\\data_initial\\data.npy") # 30 x 8 x 1224
for i,patient in enumerate(healthy):
    scaler = StandardScaler()
    if i ==0:
        healthy_normalized = np.expand_dims(scaler.fit_transform(patient.T).T, axis=0)
    else:
        healthy_normalized = np.concatenate((healthy_normalized, np.expand_dims(scaler.fit_transform(patient.T).T, axis=0)), axis=0)

healthy_normalized_mean = np.mean(healthy_normalized, axis=0)
healthy_mean = np.mean(healthy, axis=0)
figure, axis = plt.subplots(4, 1, figsize=(18, 9))
#figure.suptitle("Average principal components over all healthy controls and standard error of the mean. The red line is the average PC if we have applied normalization first.")
axis[0].set_title("Suplementery Motor Area")
axis[1].set_title("Frontal Area")
axis[2].set_title("Motor Area")
axis[3].set_title("Parietal Area")
for id, _ in enumerate(healthy_normalized_mean):
    if id % 2 == 0:
        p2, p3, p4= [], [], []
        p6, = axis[id//2].plot(np.arange(LEN_HEALTHY_ICU), healthy_normalized_mean[id],color='blue')
        axis[id//2].fill_between(np.arange(LEN_HEALTHY_ICU), healthy_normalized_mean[id], healthy_normalized_mean[id] - sem(healthy_normalized[:,id,:], axis = 0, ddof = 0),color='gray', alpha=0.2)        
        p7, = axis[id//2].plot(np.arange(LEN_HEALTHY_ICU), healthy_normalized_mean[id+1],color='red')
        axis[id//2].fill_between(np.arange(LEN_HEALTHY_ICU), healthy_normalized_mean[id+1], healthy_normalized_mean[id+1] - sem(healthy_normalized[:,id+1,:], axis = 0, ddof = 0),color='gray', alpha=0.2)        
        axis[id//2].set_ylim([-0.8, 1.7])
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
                               ['Physical Trigger', 'Rest Trigger', 'Imagery Trigger', f"Channel {id+1}", f"Channel {id+2}"], handler_map={tuple: HandlerTuple(ndivide=None)}, loc='right')
figure.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()



##########################################################################################################################

healthy = np.load("L:\\LovbeskyttetMapper\\CONNECT-ME\\DTU\\Alex_Data\\HealthyPatients\\data_followup\\data.npy") # 30 x 8 x 1224
for i,patient in enumerate(healthy):
    scaler = StandardScaler()
    if i ==0:
        healthy_normalized = np.expand_dims(scaler.fit_transform(patient.T).T, axis=0)
    else:
        healthy_normalized = np.concatenate((healthy_normalized, np.expand_dims(scaler.fit_transform(patient.T).T, axis=0)), axis=0)

healthy_normalized_mean = np.mean(healthy_normalized, axis=0)
healthy_mean = np.mean(healthy, axis=0)
figure, axis = plt.subplots(4, 1, figsize=(18, 9))
#figure.suptitle("Average principal components over all healthy controls and standard error of the mean. The red line is the average PC if we have applied normalization first.")
axis[0].set_title("Suplementery Motor Area")
axis[1].set_title("Frontal Area")
axis[2].set_title("Motor Area")
axis[3].set_title("Parietal Area")
for id, _ in enumerate(healthy_normalized_mean):
    if id % 2 == 0:
        p2, p3, p4= [], [], []
        p6, = axis[id//2].plot(np.arange(LEN_HEALTHY_ICU), healthy_normalized_mean[id],color='blue')
        axis[id//2].fill_between(np.arange(LEN_HEALTHY_ICU), healthy_normalized_mean[id], healthy_normalized_mean[id] - sem(healthy_normalized[:,id,:], axis = 0, ddof = 0),color='gray', alpha=0.2)        
        p7, = axis[id//2].plot(np.arange(LEN_HEALTHY_ICU), healthy_normalized_mean[id+1],color='red')
        axis[id//2].fill_between(np.arange(LEN_HEALTHY_ICU), healthy_normalized_mean[id+1], healthy_normalized_mean[id+1] - sem(healthy_normalized[:,id+1,:], axis = 0, ddof = 0),color='gray', alpha=0.2)        
        axis[id//2].set_ylim([-0.8, 1.7])
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
                               ['Physical Trigger', 'Rest Trigger', 'Imagery Trigger', f"Channel {id+1}", f"Channel {id+2}"], handler_map={tuple: HandlerTuple(ndivide=None)}, loc='right')
figure.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


###############################################################################################################################################

healthy = np.load("L:\\LovbeskyttetMapper\\CONNECT-ME\\DTU\\Alex_Data\\ICUPatients\\data_initial\\data.npy") # 30 x 8 x 1224

for i,patient in enumerate(healthy):
    scaler = StandardScaler()
    if i ==0:
        healthy_normalized = np.expand_dims(scaler.fit_transform(patient.T).T, axis=0)
    else:
        healthy_normalized = np.concatenate((healthy_normalized, np.expand_dims(scaler.fit_transform(patient.T).T, axis=0)), axis=0)

healthy_normalized_mean = np.mean(healthy_normalized, axis=0)
healthy_mean = np.mean(healthy, axis=0)
figure, axis = plt.subplots(4, 1, figsize=(18, 9))
#figure.suptitle("Average principal components over all healthy controls and standard error of the mean. The red line is the average PC if we have applied normalization first.")
axis[0].set_title("Suplementery Motor Area")
axis[1].set_title("Frontal Area")
axis[2].set_title("Motor Area")
axis[3].set_title("Parietal Area")
for id, _ in enumerate(healthy_normalized_mean):
    if id % 2 == 0:
        p2, p3, p4= [], [], []
        p6, = axis[id//2].plot(np.arange(LEN_HEALTHY_ICU), healthy_normalized_mean[id],color='blue')
        axis[id//2].fill_between(np.arange(LEN_HEALTHY_ICU), healthy_normalized_mean[id], healthy_normalized_mean[id] - sem(healthy_normalized[:,id,:], axis = 0, ddof = 0),color='gray', alpha=0.2)        
        p7, = axis[id//2].plot(np.arange(LEN_HEALTHY_ICU), healthy_normalized_mean[id+1],color='red')
        axis[id//2].fill_between(np.arange(LEN_HEALTHY_ICU), healthy_normalized_mean[id+1], healthy_normalized_mean[id+1] - sem(healthy_normalized[:,id+1,:], axis = 0, ddof = 0),color='gray', alpha=0.2)        
        axis[id//2].set_ylim([-0.8, 1.7])
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
                               ['Physical Trigger', 'Rest Trigger', 'Imagery Trigger', f"Channel {id+1}", f"Channel {id+2}"], handler_map={tuple: HandlerTuple(ndivide=None)}, loc='right')
figure.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()




######################################################################################################
healthy = np.load("L:\\LovbeskyttetMapper\\CONNECT-ME\\DTU\\Alex_Data\\ICUPatients\\data_followup\\data.npy") # 30 x 8 x 1224

for i,patient in enumerate(healthy):
    scaler = StandardScaler()
    if i ==0:
        healthy_normalized = np.expand_dims(scaler.fit_transform(patient.T).T, axis=0)
    else:
        healthy_normalized = np.concatenate((healthy_normalized, np.expand_dims(scaler.fit_transform(patient.T).T, axis=0)), axis=0)

healthy_normalized_mean = np.mean(healthy_normalized, axis=0)
healthy_mean = np.mean(healthy, axis=0)
figure, axis = plt.subplots(4, 1, figsize=(18, 9))
#figure.suptitle("Average principal components over all healthy controls and standard error of the mean. The red line is the average PC if we have applied normalization first.")
axis[0].set_title("Suplementery Motor Area")
axis[1].set_title("Frontal Area")
axis[2].set_title("Motor Area")
axis[3].set_title("Parietal Area")
for id, _ in enumerate(healthy_normalized_mean):
    if id % 2 == 0:
        p2, p3, p4= [], [], []
        p6, = axis[id//2].plot(np.arange(LEN_HEALTHY_ICU), healthy_normalized_mean[id],color='blue')
        axis[id//2].fill_between(np.arange(LEN_HEALTHY_ICU), healthy_normalized_mean[id], healthy_normalized_mean[id] - sem(healthy_normalized[:,id,:], axis = 0, ddof = 0),color='gray', alpha=0.2)        
        p7, = axis[id//2].plot(np.arange(LEN_HEALTHY_ICU), healthy_normalized_mean[id+1],color='red')
        axis[id//2].fill_between(np.arange(LEN_HEALTHY_ICU), healthy_normalized_mean[id+1], healthy_normalized_mean[id+1] - sem(healthy_normalized[:,id+1,:], axis = 0, ddof = 0),color='gray', alpha=0.2)        
        axis[id//2].set_ylim([-0.8, 1.7])
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
                               ['Physical Trigger', 'Rest Trigger', 'Imagery Trigger', f"Channel {id+1}", f"Channel {id+2}"], handler_map={tuple: HandlerTuple(ndivide=None)}, loc='right')
figure.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()