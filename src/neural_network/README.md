# Neural Network

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

## Making Predictions

As of now, this is what the code for making predictions does:-
1. Samples random rows from `test_set.csv`
2. Predicts the output for each row
3. Shows the predictions alongside the actual output
4. Shows the accuracy

To run this program, execute the following command from the `src\neural_network` folder
```
python make_prediction.py
```
