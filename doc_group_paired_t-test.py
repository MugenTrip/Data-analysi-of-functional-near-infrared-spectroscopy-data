import scipy.stats as stats 
from DataLoader import DataLoader

hc_path = "L:\\LovbeskyttetMapper\\CONNECT-ME\\DTU\\Alex_Data\\DOC\\data_initial"
dataloader = DataLoader(data_path=hc_path+"\\data.npy", event_path=hc_path+"\\events.npy", isDoc=False, isPCA=False)
stimuli_features, imagery_features = dataloader.getFeaturesImageryNoPCATtest()
print(stimuli_features.shape)
stimuli_features = stimuli_features.reshape(stimuli_features.shape[0] * stimuli_features.shape[1], stimuli_features.shape[2], stimuli_features.shape[3])
imagery_features = imagery_features.reshape(imagery_features.shape[0] * imagery_features.shape[1], imagery_features.shape[2], imagery_features.shape[3])
print(stimuli_features.shape)
for channel in range(8):
    # Performing the paired sample t-test 
    results = stats.ttest_rel(stimuli_features[:,channel,1], imagery_features[:,channel,1])
    print(f"Channel: {channel}: {results.pvalue}, corrected: {results.pvalue*8}")