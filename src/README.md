# Evaluation of Classifiers

To evaluate your neural network, run the following program from the current folder:-
```
python evaluate_nn.py
```
Make sure to change line `32` to indicate which model to evaluate.

To evaluate any other classifier, run the following command from the current folder:-
```
python evaluate.py
```
Depending on which classifier you want to evaluate, you need to replace `'path/to/classifier/classifier.sav'` in line `32` in `evaluation.py` with the classifier you want. Don't forget to add the relative path to the location where the classifier is stored.

For example, if you want to evaluate your support vector machine, line `32` in `evaluation.py` should be
```
model = pickle.load(open('support_vector_machine/svm_model.sav', 'rb'))
```

All the models created have been evaluated and their details are shown below:-

## Random Forest

### Confusion Matrix

|              | Predicted **0** | Predicted **1** |
|:------------:|:---------------:|:---------------:|
| Actual **0** |      643858     |       542       |
| Actual **1** |      104472     |      118728     |

### Classification Report

|                  | **Precision** | **Recall** | **f1-score** | **Support** |
|:----------------:|:-------------:|:----------:|:------------:|:-----------:|
|       **0**      |      0.86     |    1.00    |     0.92     |    644400   |
|       **1**      |      1.00     |    0.53    |     0.69     |    223200   |
|                  |               |            |              |             |
|   **Accuracy**   |               |            |   **0.88**   |    867600   |
|   **Macro Avg**  |      0.93     |    0.77    |     0.81     |    867600   |
| **Weighted Avg** |      0.90     |    0.88    |     0.87     |    867600   |


## Decision Tree

### Confusion Matrix

|              | Predicted **0** | Predicted **1** |
|:------------:|:---------------:|:---------------:|
| Actual **0** |      617229     |      27171      |
| Actual **1** |      210462     |      12738      |

### Classification Report

|                  | **Precision** | **Recall** | **f1-score** | **Support** |
|:----------------:|:-------------:|:----------:|:------------:|:-----------:|
|       **0**      |      0.75     |    0.96    |     0.84     |    644400   |
|       **1**      |      0.32     |    0.06    |     0.10     |    223200   |
|                  |               |            |              |             |
|   **Accuracy**   |               |            |   **0.73**   |    867600   |
|   **Macro Avg**  |      0.53     |    0.51    |     0.47     |    867600   |
| **Weighted Avg** |      0.64     |    0.73    |     0.65     |    867600   |


## Logistic Regression

### Confusion Matrix

|              | Predicted **0** | Predicted **1** |
|:------------:|:---------------:|:---------------:|
| Actual **0** |      644400     |        0        |
| Actual **1** |      223200     |        0        |

### Classification Report

|                  | **Precision** | **Recall** | **f1-score** | **Support** |
|:----------------:|:-------------:|:----------:|:------------:|:-----------:|
|       **0**      |      0.74     |    1.00    |     0.85     |    644400   |
|       **1**      |      0.00     |    0.00    |     0.00     |    223200   |
|                  |               |            |              |             |
|   **Accuracy**   |               |            |   **0.74**   |    867600   |
|   **Macro Avg**  |      0.37     |    0.50    |     0.43     |    867600   |
| **Weighted Avg** |      0.55     |    0.74    |     0.63     |    867600   |


## Naive Bayes Classifier

### Confusion Matrix

|              | Predicted **0** | Predicted **1** |
|:------------:|:---------------:|:---------------:|
| Actual **0** |      644398     |        2        |
| Actual **1** |      223181     |       19        |

### Classification Report

|                  | **Precision** | **Recall** | **f1-score** | **Support** |
|:----------------:|:-------------:|:----------:|:------------:|:-----------:|
|       **0**      |      0.74     |    1.00    |     0.85     |    644400   |
|       **1**      |      0.90     |    0.00    |     0.00     |    223200   |
|                  |               |            |              |             |
|   **Accuracy**   |               |            |   **0.74**   |    867600   |
|   **Macro Avg**  |      0.82     |    0.50    |     0.43     |    867600   |
| **Weighted Avg** |      0.78     |    0.74    |     0.63     |    867600   |


## Support Vector Machine

### Confusion Matrix

|              | Predicted **0** | Predicted **1** |
|:------------:|:---------------:|:---------------:|
| Actual **0** |      393869     |     250531      |
| Actual **1** |      137021     |      86179      |

### Classification Report

|                  | **Precision** | **Recall** | **f1-score** | **Support** |
|:----------------:|:-------------:|:----------:|:------------:|:-----------:|
|       **0**      |      0.74     |    0.61    |     0.67     |    644400   |
|       **1**      |      0.26     |    0.39    |     0.31     |    223200   |
|                  |               |            |              |             |
|   **Accuracy**   |               |            |   **0.55**   |    867600   |
|   **Macro Avg**  |      0.50     |    0.50    |     0.49     |    867600   |
| **Weighted Avg** |      0.62     |    0.55    |     0.58     |    867600   |


## Neural Network - Model 1

### Confusion Matrix

|              | Predicted **0** | Predicted **1** |
|:------------:|:---------------:|:---------------:|
| Actual **0** |      644400     |        0        |
| Actual **1** |      223200     |        0        |

### Classification Report

|                  | **Precision** | **Recall** | **f1-score** | **Support** |
|:----------------:|:-------------:|:----------:|:------------:|:-----------:|
|       **0**      |      0.74     |    1.00    |     0.85     |    644400   |
|       **1**      |      0.00     |    0.00    |     0.00     |    223200   |
|                  |               |            |              |             |
|   **Accuracy**   |               |            |   **0.74**   |    867600   |
|   **Macro Avg**  |      0.37     |    0.50    |     0.43     |    867600   |
| **Weighted Avg** |      0.55     |    0.74    |     0.63     |    867600   |


## Models Arranged in Decreaing Order of Accuracy

|            **Model**           | **Accuracy** |
|:------------------------------:|:------------:|
|    Random Forest Classifier    |     0.88     |
|         Neural Network         |     0.74     |
|     Naive Bayes Classifier     |     0.74     |
| Logistic Regression Classifier |     0.74     |
|    Decision Tree Classifier    |     0.73     |
|     Support Vector Machine     |     0.55     |