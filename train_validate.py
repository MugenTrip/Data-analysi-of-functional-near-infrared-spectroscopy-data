import argparse
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from DataLoader import DataLoader
from model_svm import model_SVM

BASE_PATH = os.path.join(os.path.curdir, 'data/icu')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-v" , "--validation", type=str, choices=["cross_validation", "icu_initial", "icu_followup", "healthy_followup"], 
                        help="Determine the validation type, it could be [cross_validation, icu_initial, icu_followup].", required=True)
    parser.add_argument("-t" , "--task", type=str, choices=["physical", "imagery"], 
                        help="Determine the validation type, it could be [cross_validation, icu_initial, icu_followup].", required=True)
    parser.add_argument("-c" , "--channels", type=list, 
                        help="Define the channels you want to use like 0123 for all channels.", required=True)
    args = parser.parse_args()

    validation_path = None
    if args.validation == "cross_validation":
            pass
    elif args.validation == "icu_initial":
            validation_path = os.path.join(BASE_PATH, 'initial')
    elif args.validation == "icu_followup":
            validation_path = os.path.join(BASE_PATH, 'followup')
    elif args.validation == "healthy_followup":
            validation_path = os.path.join(BASE_PATH.replace("icu", "healthy"), 'followup')

    if args.channels is None:
          channels = [0,1,2,3]
    else:
          print(type(args.channels))
          print(args.channels)
          channels = [int(x) for x in args.channels]

    
    hc_path = os.path.join(os.path.curdir, 'data/healthy/initial')
    dataloader = DataLoader(data_path=os.path.join(hc_path, 'data.npy'), event_path=os.path.join(hc_path, 'events.npy'), isDoc=False)
    if args.task == "physical":
        train_data, train_labels = dataloader.getFeatures(channels=channels)
    else:
        train_data, train_labels = dataloader.getFeaturesImagery(channels=channels) 
        print(train_data.shape)
    train_data = train_data.reshape(train_data.shape[0], train_data.shape[1]*train_data.shape[2])
          
    if validation_path == None:
            ss  = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
            ss.get_n_splits(train_data, train_labels)
            accuracy = []
            f1 = []
            for i, (train, test) in enumerate(ss.split(train_data, train_labels)):
                X_train, y_train = train_data[train], train_labels[train]
                X_test, y_test = train_data[test], train_labels[test]

                my_model = model_SVM()
                accuracy.append(my_model.train_validate(X_train, y_train, X_test, y_test))
            print(f"Accuracy list: {accuracy}, with mean: {np.mean(accuracy)}")
    else:
        dataloader = DataLoader(data_path=os.path.join(validation_path, 'data.npy'), event_path=os.path.join(validation_path, 'events.npy'), isDoc=False)
        if args.task == "physical":
            validate_data, validate_labels = dataloader.getFeatures(channels=channels)
        else:
            validate_data, validate_labels = dataloader.getFeaturesImagery(channels=channels)
            print(validate_data.shape)
        validate_data = validate_data.reshape(validate_data.shape[0], validate_data.shape[1]*validate_data.shape[2])

        my_model = model_SVM()
        accuracy = my_model.train_validate(train_data, train_labels, validate_data, validate_labels)
        print(f"Accuracy list: {accuracy}, with mean: {np.mean(accuracy)}")


