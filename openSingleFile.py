import mne
from Hemo import HemoData


path_29 = "L:\LovbeskyttetMapper\CONNECT-ME\DTU\Alex_Data\DoC\data_initial\P29_1.snirf"
path_36 = "L:\LovbeskyttetMapper\CONNECT-ME\DTU\Alex_Data\DoC\data_initial\P36_1.snirf"
path_40 = "L:\LovbeskyttetMapper\CONNECT-ME\DTU\Alex_Data\DoC\data_initial\P40_1.snirf"
path_36_2 = "L:\LovbeskyttetMapper\CONNECT-ME\DTU\Alex_Data\DoC\data_followup\P36_3.snirf"
paths = [path_29, path_40, path_36, path_36_2]
for path in paths:
    raw_haemo = HemoData(path, preprocessing=True, isPloting=False, isDoC=True)
    raw_haemo.plot()
