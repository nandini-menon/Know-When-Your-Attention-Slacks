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

## Building the Model

All models built have 21 nodes in the input layer (corresponding to the 21 columns in input data) and 1 node in the output layer which gives an output of:-
* 0 - meaning the person is not attentive
* 1 - meaning the person is attentive

Four models have been built each with difference in the number of hidden layers and the nodes in each hidden layer:-

### Model 1

| **Hidden Layer** | **No. of nodes** |
|:----------------:|:----------------:|
|         1        |        11        |

**Accuracy**: 74.9%

#### To build this model

Execute the following commands from the project folder:-
```
cd src/neural_network/model_1
python model_1.py
```
After this program finishes execution, the model will be saved as `model_1.h5` to the `src/neural-network/model_1` folder

## Model 2

| **Hidden Layer** | **No. of nodes** |
|:----------------:|:----------------:|
|         1        |        21        |
|         2        |        10        |
|         3        |         5        |

**Accuracy**: 75.8%

#### To build this model

Execute the following commands from the project folder:-
```
cd src/neural_network/model_2
python model_2.py
```
After this program finishes execution, the model will be saved as `model_2.h5` to the `src/neural-network/model_2` folder

## Model 3

| **Hidden Layer** | **No. of nodes** |
|:----------------:|:----------------:|
|         1        |        30        |
|         2        |        20        |
|         3        |        10        |
|         4        |         5        |

**Accuracy**: 76.6%

#### To build this model

Execute the following commands from the project folder:-
```
cd src/neural_network/model_3
python model_3.py
```
After this program finishes execution, the model will be saved as `model_3.h5` to the `src/neural-network/model_3` folder

## Model 4

| **Hidden Layer** | **No. of nodes** |
|:----------------:|:----------------:|
|         1        |        21        |
|         2        |        11        |

**Accuracy**: 75.7%

#### To build this model

Execute the following commands from the project folder:-
```
cd src/neural_network/model_4
python model_4.py
```
After this program finishes execution, the model will be saved as `model_4.h5` to the `src/neural-network/model_4` folder

The rest of this README.md will be completed later :wink: