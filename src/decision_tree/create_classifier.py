import pickle
from pandas import read_csv
from sklearn.tree import DecisionTreeClassifier
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
    clf = DecisionTreeClassifier(max_depth=30)
    clf.fit(X_train, Y_train)
    pickle.dump(clf, open('decision_tree.sav', 'wb'))


def evaluate_classifier(X_test, Y_test):
    model = pickle.load(open('decision_tree.sav', 'rb'))
    Y_pred = model.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))


def main():
    X_train, Y_train, X_test, Y_test = get_data()
    create_classifier(X_train, Y_train)
    evaluate_classifier(X_test, Y_test)

if __name__ == '__main__':
    main()
