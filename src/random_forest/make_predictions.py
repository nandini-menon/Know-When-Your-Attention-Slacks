import pickle
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


def feature_scaling(X_new):
    sc = StandardScaler()
    X_new = sc.fit_transform(X_new)
    return X_new


def sample_dataset():
    root_path = '../../data/'
    df = pd.read_csv(f'{root_path}test_set.csv')
    df = df.sample(n=20)
    dataset = df.values

    X_new = dataset[:, 0:21].astype(float)
    Y_new = dataset[:, 21].astype("int32")

    X_new = feature_scaling(X_new)
    return (X_new, Y_new)


def get_predictions(X_new, Y_new):
    model = pickle.load(open('random_forest.sav', 'rb'))
    Y_pred = model.predict(X_new)

    print("Prediction\t Actual Result")
    print("------------------------------")
    for i in range(len(Y_new)):
        print(f'{Y_pred[i]}\t\t\t     {Y_new[i]}')

    print("Accuracy:", metrics.accuracy_score(Y_new, Y_pred))


def main():
    X_new, Y_new = sample_dataset()
    get_predictions(X_new, Y_new)

if __name__ == '__main__':
    main()
