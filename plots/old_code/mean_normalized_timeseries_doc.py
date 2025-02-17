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

LEN_HEALTHY_ICU = 7756
#LEN_DOC = 2360

doc = np.load("L:\\LovbeskyttetMapper\\CONNECT-ME\\DTU\\Alex_Data\\DoC\\data_initial\\data.npy") # 37 x 24  x 8 x 7756
print(doc.shape)

# for i,patient in enumerate(doc):
#     scaler = StandardScaler()
#     doc[i] = scaler.fit_transform(patient.T).T

for i,patient in enumerate(doc):
    scaler = StandardScaler()
    if i ==0:
        healthy_normalized = np.expand_dims(scaler.fit_transform(patient.T).T, axis=0)
    else:
        healthy_normalized = np.concatenate((healthy_normalized, np.expand_dims(scaler.fit_transform(patient.T).T, axis=0)), axis=0)

healthy_normalized_mean = np.mean(healthy_normalized, axis=0)
# healthy_mean = np.mean(healthy, axis=0)
figure, axis = plt.subplots(4, 1, figsize=(20, 9))
figure.suptitle("Average principal components over all healthy controls and standard error of the mean. The red line is the average PC if we have applied normalization first.")
axis[0].set_title("Suplementery Motor Area")
axis[1].set_title("Frontal Area")
axis[2].set_title("Motor Area")
axis[3].set_title("Rear Area")
for id, _ in enumerate(healthy_normalized_mean):
    if id % 2 == 0:
        p2, p3 = [], []
        p6, = axis[id//2].plot(np.arange(LEN_HEALTHY_ICU), healthy_normalized_mean[id],color='blue')
        axis[id//2].fill_between(np.arange(LEN_HEALTHY_ICU), healthy_normalized_mean[id], healthy_normalized_mean[id] - sem(healthy_normalized[:,id,:], axis = 0, ddof = 0),color='gray', alpha=0.2)        
        p7, = axis[id//2].plot(np.arange(LEN_HEALTHY_ICU), healthy_normalized_mean[id+1],color='red')
        axis[id//2].fill_between(np.arange(LEN_HEALTHY_ICU), healthy_normalized_mean[id+1], healthy_normalized_mean[id+1] - sem(healthy_normalized[:,id+1,:], axis = 0, ddof = 0),color='gray', alpha=0.2)        
        axis[id//2].set_ylim([-0.6, 0.6])
        for idx in range(24):
            aggrated_time_of_event = idx*(153+142) + (idx // 8) * 336 # Every 15 sec 
            tmp = axis[id//2].axvline(x = aggrated_time_of_event, color = 'y')
            p2.append(tmp)
            tmp = axis[id//2].axvline(x = aggrated_time_of_event + 153, color = 'c')
            p3.append(tmp)

        
        l = axis[id//2].legend([tuple(p2), tuple(p3), p6, p7], 
                               ['Physical Trigger', 'Rest Trigger', f"Channel {id+1}", f"Channel {id+2}"], handler_map={tuple: HandlerTuple(ndivide=None)}, loc='right')
figure.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()