import argparse
import numpy as np
from DataLoader import DataLoader
from model_svm import model_SVM
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn import svm
from statistics import mean, stdev



if __name__ == '__main__':
    doc_path = "L:\\LovbeskyttetMapper\\CONNECT-ME\\DTU\\Alex_Data\\DoC\\data_followup\\"
    doc_map = np.load(doc_path + "map.npy")
    dataloader = DataLoader(data_path = doc_path + "data.npy", event_path = doc_path + "events.npy", isDoc=True)
    doc_data, doc_label = dataloader.getFeaturesDoc(channels=[0,1,2,3])
    doc_data = doc_data.reshape(doc_data.shape[0], doc_data.shape[1], doc_data.shape[2]*doc_data.shape[3])

    acc = {}
    patient_mean = []
    patient_std = []
    for doc_patient_id, doc_patient in enumerate(doc_data):
        lst_accu_stratified = []
        skf = LeaveOneOut()

        for train_index, test_index in skf.split(doc_patient, doc_label):
            x_train_fold, x_test_fold = doc_patient[train_index], doc_patient[test_index]
            y_train_fold, y_test_fold = doc_label[train_index], doc_label[test_index]

            # Standardize
            sc = StandardScaler()
            x_train_fold = sc.fit_transform(x_train_fold)
            x_test_fold = sc.transform(x_test_fold)

            svm_clf = svm.SVC(kernel='linear') # Linear Kernel
            # train the model
            svm_clf.fit(x_train_fold, y_train_fold.ravel())

            # valiidate model
            svm_clf_pred = svm_clf.predict(x_test_fold)
            lst_accu_stratified.append(svm_clf.score(x_test_fold, y_test_fold))

        patient_mean.append(mean(lst_accu_stratified))
        patient_std.append(stdev(lst_accu_stratified))

    print(np.argmax(patient_mean))
    print(mean(patient_mean))
    for i, patient in enumerate(doc_map):
        print(f"{patient} accuracy: {patient_mean[i]}")









