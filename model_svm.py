import math
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class model_SVM():
    """"""
    def __init__(self, percentile: float=15) -> None:
        self.sc = StandardScaler()
        self.svm_clf = svm.SVC(kernel='linear') # Linear Kernel

    def predict(self, data: np.ndarray):
        # Standardize
        data = self.sc.transform(data)
        # make predictions
        return self.svm_clf.predict(data)
         
    def train(self, training_data: np.ndarray, training_labels: np.ndarray):
        # Standardize
        training_data = self.sc.fit_transform(training_data)
        # train the model
        self.svm_clf.fit(training_data, training_labels)
        print(self.svm_clf)


    def train_validate(self, training_data: np.ndarray, training_labels: np.ndarray, validation_data: np.ndarray, validation_labels: np.ndarray):
        # Standardize
        training_data = self.sc.fit_transform(training_data)
        validation_data = self.sc.transform(validation_data)
        
        # train the model
        self.svm_clf.fit(training_data, training_labels)

        print(self.svm_clf)
        
        # make predictions
        svm_clf_pred = self.svm_clf.predict(validation_data)
        # print other performance metrics
        print("___________________________\n")
        print("Loss of Support Vector Machine: ", accuracy_score(validation_labels, svm_clf_pred))
        print("Accuracy of Support Vector Machine: ", accuracy_score(validation_labels, svm_clf_pred))
        print("Precision of Support Vector Machine: ", precision_score(validation_labels, svm_clf_pred, average='weighted'))
        print("Recall of Support Vector Machine: ", recall_score(validation_labels, svm_clf_pred, average='weighted'))
        print("F1-Score of Support Vector Machine: ", f1_score(validation_labels, svm_clf_pred, average='weighted'))
        print("___________________________\n")

        cm = confusion_matrix(validation_labels,svm_clf_pred)
        cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ["Class_0", "Class_1"])
        cm_display.plot()
        #plt.show()

        return accuracy_score(validation_labels, svm_clf_pred)