# Random Forest Classifier

## Building the Model

To build this classifier, execute the following command from the current folder:-
```
python create_classifier.py
```
After this program is executed, the model is created, trained and saved as `random_forest.sav` in this folder.

## Making Predictions

As of now, this is what the code for making predictions does:-
1. Samples random rows from `test_set.csv`
2. Predicts the output for each row
3. Shows the predictions alongside the actual output
4. Shows the accuracy

To run this program, execute the following command from the current folder:-
```
python make_predictions.py
```

> **Note:** If you haven't built the model, the model `random_forest.sav` can be downloaded from [here](https://drive.google.com/drive/folders/1ENGLS1iYebGZJKPndEneSfW3ZhvUABnz?usp=sharing). Please note that the size of this file is `4.87 GB`.

## Output Screenshot

![Random Forest Classifier - Screenshot of Output](screenshot_1.png?raw=true "Random Forest Classifier - Screenshot of Output")

![Random Forest Classifier - Screenshot of Output](screenshot_2.png?raw=true "Random Forest Classifier - Screenshot of Output")