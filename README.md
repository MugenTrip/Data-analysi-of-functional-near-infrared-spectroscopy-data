# Data-analysi-of-functional-near-infrared-spectroscopy-data.

Place the files with the .snirf extension under the folder data, with respect to the participant group (healthy, icu, doc) and the session (initial, followup).

## Preprocessing 

The first step is to run the preprocessing pipeline. You can do this by using the run_preprocess.py script specifing the paricipant groupd using -d and the session using -s. E.g:

```
python run_preprocess.py -d healthy -s initial
```

After that 3 files (data.npy, events.npy, map.npy) should have been created at the corresponding folder.


## Train and validate group level classifier

To train and validate the group level classifier use the train_validate.py script. The model will be trained using the data located under ./data/healthy/initial therefore you need to run the preprocessing pipeline for those data (described above). In addition, if you want to validate your model using a different dataset, you need to run the preprocessing for this dataset as well. The arguments you need to run this script are -v or --validation to define the validation method (cross_validation, healthy_followup, icu_initial, icu_followup), -t or --task to define the task (physical,imagery) and -c to definr the areas (0: SMA, 1: Frontal, 2: Motor, 3: Parietal). E.g:

```
python train_validate.py -v cross_validation -t physical -c 02
```


## Train and validate individual level classifier

To train and validate the individual level classifier use the doc_single_classifier.py script. You can run it as: 

```
python doc_single_classifier.py -s initial -c 02
```