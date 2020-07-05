import pandas as pd
from tensorflow import keras


def sample_dataset():
    root_path = '../../data/'
    df = pd.read_csv(f'{root_path}test_set.csv')

    df_0 = df[df['Label'] == 0]
    df_1 = df[df['Label'] == 1]

    df_0 = df_0.sample(n=15)
    df_1 = df_1.sample(n=5)

    df = pd.concat([df_0, df_1], sort=True)
    df = df.sample(frac=1)

    dataset = df.values
    X_new = dataset[:, 0:21].astype(float)
    Y_new = dataset[:, 21].astype("int32")

    return (X_new, Y_new)


def get_predictions(X_new, Y_new):
    trained_model = keras.models.load_model(f'model_3/model_3')
    result = (trained_model.predict(X_new) > 0.5).astype("int32")

    print("Prediction\t Actual Result")
    print("------------------------------")
    for i in range(len(Y_new)):
        print(f'{result[i][0]}\t\t\t     {Y_new[i]}')

    _, acc = trained_model.evaluate(X_new, Y_new)
    print('Accuracy: {:5.2f}%'.format(100*acc))


def main():
    X_new, Y_new = sample_dataset()
    get_predictions(X_new, Y_new)

if __name__ == '__main__':
    main()
