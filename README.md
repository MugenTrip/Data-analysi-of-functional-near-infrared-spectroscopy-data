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

## Plots and Figures

### Group level figures

The group level figures include one figure with the average "paired" features and one with the average time-series, across all participants of a participant group (heathy, icu, doc). You can run each script as follows:

```
python plots\mean_features_topo.py -d healthy -s initial -t physical
```

```
python plots\mean_timeseries.py -d healthy -s initial
```

The -d argument could be [healthy, icu, doc] and defines the participant group, the -s argument refers to the session and it could be [initial, followup], and finally, the -t argument defines the task and it could be [physical, imagery].

### Individual level figures

At the individual level, we can plot either the time-series of each individual or the average features across all trials of each individual. To generate those plots, run the following commands: 

```
python plots\features_topo_for_each_individual.py -d healthy -s initial -t physical
```

```
python plots\timeseries_for_each_individual.py -d healthy -s initial
```

For the average time-series you will get the plot of each individual one by one.

## T-test

You can run a two tailed paired t-test to compare the stimuli period with the resting period using the following commands: 

```
python t-test\physical_paired_t_test.py -d healthy -s initial
```

```
python t-test\imagery_paired_t_test.py -d healthy -s initial
```
The first one is for the physical task while the second one is for the imagery task.

For the DOC patients, the commands are similar but they have only one argument (-s) which is the session ([initial, followup]). You can run the t-test in group level (first command) or in individual level (second command):

```
python t-test\t-test_doc_group_paired.py -s initial 
```

```
python t-test\t-test_doc_individual.py -s initial 
```

