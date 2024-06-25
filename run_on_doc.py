import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold
from DataLoader import DataLoader
from model_svm import model_SVM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

if __name__ == '__main__':
    hc_path = "L:\\LovbeskyttetMapper\\CONNECT-ME\\DTU\\Alex_Data\\HealthyPatients\\data_initial\\"
    dataloader = DataLoader(data_path=hc_path+"data.npy", event_path=hc_path+"events.npy", isDoc=False)
    train_data, train_labels = dataloader.getFeatures(channels=[0,3])
    train_data = train_data.reshape(train_data.shape[0], train_data.shape[1]*train_data.shape[2])

    
    doc_path = "L:\\LovbeskyttetMapper\\CONNECT-ME\\DTU\\Alex_Data\\DoC\\data_initial\\"
    doc_map = np.load(doc_path + "map.npy")
    dataloader = DataLoader(data_path = doc_path + "data_no_tddr.npy", event_path = doc_path + "events.npy", isDoc=True)
    doc_data, doc_label = dataloader.getFeaturesDoc(channels=[1,2])
    doc_data = doc_data.reshape(doc_data.shape[0], doc_data.shape[1], doc_data.shape[2]*doc_data.shape[3])
    #print("Here")

    my_model = model_SVM()
    my_model.train(train_data, train_labels)

    acc = {}
    for doc_patient_id, doc_patient in enumerate(doc_data):
        my_model.predict(doc_patient)
        # make predictions
        svm_clf_pred = my_model.predict(doc_patient)
        print(doc_label.shape)
        # print the accuracy
        print("Accuracy of Support Vector Machine: ",
        accuracy_score(doc_label, svm_clf_pred))
        # print other performance metrics
        print("Precision of Support Vector Machine: ",
        precision_score(doc_label, svm_clf_pred, average='weighted'))
        print("Recall of Support Vector Machine: ",
        recall_score(doc_label, svm_clf_pred, average='weighted'))
        print("F1-Score of Support Vector Machine: ",
        f1_score(doc_label, svm_clf_pred, average='weighted'))
        print("___________________________\n")

        accuracy = accuracy_score(doc_label, svm_clf_pred)
        f1 = f1_score(doc_label, svm_clf_pred, average='weighted')
        
        acc[doc_map[doc_patient_id][0]] = accuracy

    print(acc)
    for key in acc:
        print(f"{key}: {acc[key]}")









