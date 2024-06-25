import numpy as np
from DataPath import DataPath
from sklearn.preprocessing import StandardScaler
from matplotlib.legend_handler import HandlerTuple
from scipy.stats import sem
import matplotlib.pyplot as plt



LEN_HEALTHY_ICU = 7756

doc_path = "L:\\LovbeskyttetMapper\\CONNECT-ME\\DTU\\Alex_Data\\DoC\\data_initial\\"
doc = np.load("L:\\LovbeskyttetMapper\\CONNECT-ME\\DTU\\Alex_Data\\DoC\\data_initial\\data.npy") # 29  x 8 x 7756
doc_map = np.load(doc_path + "map.npy")

good_doc_normalized = None
for idx,patient in enumerate(doc_map):
    if patient in ["P29_1", "P28_1", "P36_1"]:
        print(patient)
        scaler = StandardScaler()
        if good_doc_normalized is None:
            #good_doc_normalized = np.expand_dims(scaler.fit_transform(doc[idx].T).T, axis=0)
            good_doc_normalized = np.expand_dims(doc[idx], axis=0)
        else:
            # good_doc_normalized = np.concatenate((good_doc_normalized, np.expand_dims(scaler.fit_transform(doc[idx].T).T, axis=0)), axis=0)
            good_doc_normalized = np.concatenate((good_doc_normalized, np.expand_dims(doc[idx], axis=0)), axis=0)

print(good_doc_normalized.shape)
healthy_normalized_mean = np.mean(good_doc_normalized, axis=0)
# healthy_mean = np.mean(healthy, axis=0)
figure, axis = plt.subplots(4, 1)
figure.suptitle("Average principal components over all healthy controls and standard error of the mean. The red line is the average PC if we have applied normalization first.")
axis[0].set_title("Suplementery Motor Area")
axis[1].set_title("Frontal Area")
axis[2].set_title("Motor Area")
axis[3].set_title("Rear Area")
for id, _ in enumerate(healthy_normalized_mean):
    if id % 2 == 0:
        p2, p3 = [], []
        p6, = axis[id//2].plot(np.arange(LEN_HEALTHY_ICU), healthy_normalized_mean[id],color='blue')
        axis[id//2].fill_between(np.arange(LEN_HEALTHY_ICU), healthy_normalized_mean[id], healthy_normalized_mean[id] - sem(good_doc_normalized[:,id,:], axis = 0, ddof = 0),color='gray', alpha=0.2)        
        p7, = axis[id//2].plot(np.arange(LEN_HEALTHY_ICU), healthy_normalized_mean[id+1],color='red')
        axis[id//2].fill_between(np.arange(LEN_HEALTHY_ICU), healthy_normalized_mean[id+1], healthy_normalized_mean[id+1] - sem(good_doc_normalized[:,id+1,:], axis = 0, ddof = 0),color='gray', alpha=0.2)        
        for idx in range(24):
            aggrated_time_of_event = idx*(153+142) + (idx // 8) * 336 # Every 15 sec 
            tmp = axis[id//2].axvline(x = aggrated_time_of_event, color = 'y')
            p2.append(tmp)
            tmp = axis[id//2].axvline(x = aggrated_time_of_event + 153, color = 'c')
            p3.append(tmp)

        
        l = axis[id//2].legend([tuple(p2), tuple(p3), p6, p7], 
                               ['Physical Trigger', 'Rest Trigger', f"Channel {id}", f"Channel {id+1}"], handler_map={tuple: HandlerTuple(ndivide=None)}, loc='right')
figure.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


for patient in good_doc_normalized:
    figure, axis = plt.subplots(4, 1)
    figure.suptitle("Average principal components over all healthy controls and standard error of the mean. The red line is the average PC if we have applied normalization first.")
    axis[0].set_title("Suplementery Motor Area")
    axis[1].set_title("Frontal Area")
    axis[2].set_title("Motor Area")
    axis[3].set_title("Rear Area")
    for id, _ in enumerate(patient):
        if id % 2 == 0:
            p2, p3 = [], []
            p6, = axis[id//2].plot(np.arange(LEN_HEALTHY_ICU), patient[id],color='blue')
            p7, = axis[id//2].plot(np.arange(LEN_HEALTHY_ICU), patient[id+1],color='red')
            for idx in range(24):
                aggrated_time_of_event = idx*(153+142) + (idx // 8) * 336 # Every 15 sec 
                tmp = axis[id//2].axvline(x = aggrated_time_of_event, color = 'y')
                p2.append(tmp)
                tmp = axis[id//2].axvline(x = aggrated_time_of_event + 153, color = 'c')
                p3.append(tmp)

            
            l = axis[id//2].legend([tuple(p2), tuple(p3), p6, p7], 
                                ['Physical Trigger', 'Rest Trigger', f"Channel {id}", f"Channel {id+1}"], handler_map={tuple: HandlerTuple(ndivide=None)}, loc='right')
    figure.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()