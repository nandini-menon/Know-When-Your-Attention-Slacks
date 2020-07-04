# Know-When-Your-Attention-Slacks
Neural network to classify brainwaves to know whether a person is attentive or not

## Prerequisites

This project uses Python 3.x and pip.

Installing python and pip
### Debain based systems
```
apt install python3 python3-pip
```
### Arch based systems
```
pacman -S python python-pip
```
### Solus
```
eopkg it python3 python3-pip
```
## Installing
External packages used in the python program are: `Numpy`, `Pandas`, `PyEDFLib` and `TensorFlow`.
It is recommended to install these in a python virtual environment. The instructions for installing these packages from the project root folder with a python virtual environment are given below:-
```
python -m venv .
source bin/activate
pip install -r requirements.txt
```

To deactivate the virtual environment just execute the following command:-
```
deactivate
```

## Dataset

The dataset used is 'EEG During Mental Arithmetic Tasks' which can be found at https://physionet.org/content/eegmat/1.0.0/.

The original publication for this resource : [Zyma I, Tukaev S, Seleznov I, Kiyono K, Popov A, Chernykh M, Shpenkov O. Electroencephalograms during Mental Arithmetic Task Performance. Data. 2019; 4(1):14. https://doi.org/10.3390/data4010014](https://www.mdpi.com/2306-5729/4/1/14)

The dataset is downloaded and extracted to `data/edf_inputs`

### Understanding the dataset

* The dataset is provided in a format with extension `.edf`.
* It contains the brainwave readings of 36 people labelled from Subject00 to Subject35.
* Readings taken before the arithmetic task is stored in files labelled with suffix `_1` and readings taken during the arithmetic task is stored in files labelled with suffix `_2`.

## Preprocessing the Dataset

### Step 1
The edf format data has to be converted to csv.

The code for this is available in `src/edf_to_csv.py`.
This can be run by executing the following commands from the project root folder:-
```
cd src/
python edf_to_csv.py
```
After running this program, the corresponding csv file for each edf file will be available in `data/csv_inputs`

### Step 2
The data has to be separated into training set and test set.

The code for this is available in `src/split_train_test.py`
This can be run by executing the following commands from the project root folder:-
```
cd src/
python split_train_test.py
```
After running this program, the training set will be stored in `data/training_set.csv` and the test_set will be stored in `data/test_set.csv`.

> Note: Before running `split_train_test.py`, please make sure that there are no files named `training_set.csv` and `test_set.csv` in the folder `data`. If these files exist, then the data that you are processing will be appended at the end of these files.

## Classification

As of now, the following classifiers have been implemented:-
* [Neural Network](src/neural_network)
* [Naive Bayes Classifier](src/naive_bayes_classifier)
* [Logistic Regression Classifier](src/logistic_regression)
* [Support Vector Machine](src/support_vector_machine)
* [Decision Tree](src/decision_tree)
* [Random Forest Classifier](src/random_forest)

To build these classifiers and make predictions, head on to the respective folders and checkout the README

> More classifiers will be added later and a comparison will be done to know which method works best :smile: