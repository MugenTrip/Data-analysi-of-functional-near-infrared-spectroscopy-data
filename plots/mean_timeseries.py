import sys
import path
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)
import argparse
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import sem
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple



BASE_PATH = os.path.join(os.path.curdir, 'data/')
MAX_LENGTH_HEALTHY = 1224
MAX_LENGTH_DOC = 7756

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d" , "--datatype", type=str, choices=["healthy", "icu", "doc"], help="Determine the data type, could be [healthy, icu, doc].", required=True)
    parser.add_argument("-s", "--session", type=str, choices=["initial", "followup"], help="Session, could be initial or followup.", required=True)
    args = parser.parse_args()

    data_path = None
    if args.datatype == "healthy":
            data_path = os.path.join(BASE_PATH, 'healthy')
            max_length = MAX_LENGTH_HEALTHY
    elif args.datatype == "icu":
            data_path = os.path.join(BASE_PATH, 'icu')
            max_length = MAX_LENGTH_HEALTHY
    elif args.datatype == "doc":
            data_path = os.path.join(BASE_PATH, 'doc')
            max_length = MAX_LENGTH_DOC

    if args.session == "initial":
            data_path = os.path.join(data_path, 'initial')
    elif args.session == "followup":
            data_path = os.path.join(data_path, 'followup')

    # Load data
    data = np.load(os.path.join(data_path, 'data.npy')) # 30 x 8 x 1224

    # Normalize each channel separetely
    for i,patient in enumerate(data):
        scaler = StandardScaler()
        if i ==0:
            data_normalized = np.expand_dims(scaler.fit_transform(patient.T).T, axis=0)
        else:
            data_normalized = np.concatenate((data_normalized, np.expand_dims(scaler.fit_transform(patient.T).T, axis=0)), axis=0)

    data_normalized_mean = np.mean(data_normalized, axis=0)
    data_mean = np.mean(data, axis=0)
    figure, axis = plt.subplots(4, 1, figsize=(18, 9))
    axis[0].set_title("Suplementery Motor Area")
    axis[1].set_title("Frontal Area")
    axis[2].set_title("Motor Area")
    axis[3].set_title("Parietal Area")
    
    if args.datatype != "doc":
        for id, _ in enumerate(data_normalized_mean):
            if id % 2 == 0:
                p2, p3, p4= [], [], []
                p6, = axis[id//2].plot(np.arange(max_length), data_normalized_mean[id],color='blue')
                axis[id//2].fill_between(np.arange(max_length), data_normalized_mean[id], data_normalized_mean[id] - sem(data_normalized[:,id,:], axis = 0, ddof = 0),color='gray', alpha=0.2)        
                p7, = axis[id//2].plot(np.arange(max_length), data_normalized_mean[id+1],color='red')
                axis[id//2].fill_between(np.arange(max_length), data_normalized_mean[id+1], data_normalized_mean[id+1] - sem(data_normalized[:,id+1,:], axis = 0, ddof = 0),color='gray', alpha=0.2)        
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
    else:
        for id, _ in enumerate(data_normalized_mean):
            if id % 2 == 0:
                p2, p3 = [], []
                p6, = axis[id//2].plot(np.arange(max_length), data_normalized_mean[id],color='blue')
                axis[id//2].fill_between(np.arange(max_length), data_normalized_mean[id], data_normalized_mean[id] - sem(data_normalized[:,id,:], axis = 0, ddof = 0),color='gray', alpha=0.2)        
                p7, = axis[id//2].plot(np.arange(max_length), data_normalized_mean[id+1],color='red')
                axis[id//2].fill_between(np.arange(max_length), data_normalized_mean[id+1], data_normalized_mean[id+1] - sem(data_normalized[:,id+1,:], axis = 0, ddof = 0),color='gray', alpha=0.2)        
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
