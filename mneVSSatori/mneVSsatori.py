
from Hemo import HemoData
import numpy as np
from sklearn.preprocessing import StandardScaler
from DataPath import DataPath

# Loop over all files
files_mne = DataPath("L:\\LovbeskyttetMapper\\CONNECT-ME\\DTU\\Alex_Data\\HealthyPatients\\data_initial\\", recursive=False).getDataPaths()
files_satori = DataPath("L:\\LovbeskyttetMapper\\CONNECT-ME\\DTU\\Alex_Data\\Satori\\HealthyPatients\\", recursive=False).getDataPaths()

def get_from_list(lst: list, string: str):
    for item in lst:
        if item.find(string) != -1:
            return item
    return None
        
mean_of_all = list()
names = list()
for idx,mne_file in enumerate(files_mne):
    # if mne_file.split("\\")[-1].replace(".snirf","") != "C9_raw":
    #     continue
    satori_file = get_from_list(files_satori, mne_file.split("\\")[-1].replace(".snirf",""))
    print("Opening : " + mne_file + " and " + satori_file)
    hemo_mne = HemoData(mne_file, preprocessing=True, isPloting=False)
    hemo_satori = HemoData(satori_file, preprocessing=False)

    #print(hemo_satori.getMneIoRaw().get_data(picks=["hbo"]).shape)

    hemo_mne_np = hemo_mne.getMneIoRaw().get_data(picks=["hbo"])
    hemo_satori_np = hemo_satori.getMneIoRaw().get_data(picks=["hbo"])
    #Ensure_same_length
    min_len = min(hemo_mne_np.shape[-1], hemo_satori_np.shape[-1])
    hemo_mne_np = hemo_mne_np[:,:min_len]
    hemo_satori_np = hemo_satori_np[:,:min_len]
    print(hemo_mne_np.shape)
    print(hemo_satori_np.shape)

    # Standardize
    # sc = StandardScaler()
    # hemo_mne_np = sc.fit_transform(hemo_mne_np)
    # hemo_satori_np = sc.fit_transform(hemo_satori_np)

    correlation_per_channel = list()
    for idx, _ in enumerate(hemo_mne_np):
        cor = np.corrcoef(hemo_mne_np[idx],hemo_satori_np[idx])
        print("Channel " + str(idx+1) + " correlation: " + str(cor[0,1]))
        correlation_per_channel.append(cor[0,1])

    print(correlation_per_channel)
    print("mean: " +  str(np.average(correlation_per_channel)))
    mean_of_all.append(np.average(correlation_per_channel))
    names.append(mne_file.split("\\")[-1].replace(".snirf",""))

    #hemo_mne.plot(show=False, title="MNE")
    #hemo_satori.plot(show=True, title="Satori")
    print("_________________________\n")

print(mean_of_all)



for i,n in enumerate(names):
    print(f"{n}: {mean_of_all[i]}")

print(f"Average: {np.average(mean_of_all)}")