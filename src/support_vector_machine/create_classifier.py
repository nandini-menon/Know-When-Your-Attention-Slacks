#Import svm model
import pickle
from sklearn import svm
from sklearn.kernel_approximation import Nystroem
from pandas import read_csv
from sklearn import metrics


def get_data():
    root_path = '../../data/'
    train_df = read_csv(f'{root_path}training_set.csv')
    test_df = read_csv(f'{root_path}test_set.csv')

    train_dataset = train_df.values
    test_dataset = test_df.values

    X_train = train_dataset[:, 0:21].astype(float)
    Y_train = train_dataset[:, 21].astype('int32')

    X_test = test_dataset[:, 0:21].astype(float)
    Y_test = test_dataset[:, 21].astype('int32')

    return (X_train, Y_train, X_test, Y_test)


def create_classifier(X_train, Y_train):
    clf = svm.LinearSVC()
    feature_map_nystroem = Nystroem(gamma=.2,
                                    random_state=1,
                                    n_components=21)
    data_transformed = feature_map_nystroem.fit_transform(X_train)
    clf.fit(data_transformed, Y_train)
    print('Score: ', clf.score(data_transformed, Y_train))

    pickle.dump(clf, open('svm_model.sav', 'wb'))


def evaluate_classifier(X_test, Y_test):
    model = pickle.load(open('svm_model.sav', 'rb'))
    Y_pred = model.predict(X_test)

    print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
    print("Precision:",metrics.precision_score(Y_test, Y_pred))
    print("Recall:",metrics.recall_score(Y_test, Y_pred))


def main():
    X_train, Y_train, X_test, Y_test = get_data()
    create_classifier(X_train, Y_train)
    evaluate_classifier(X_test, Y_test)

if __name__ == '__main__':
    main()
