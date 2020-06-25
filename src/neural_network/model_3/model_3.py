import tensorflow as tf
from pandas import read_csv
from tensorflow import keras
from tensorflow.keras import layers


class NeuralNet(keras.Model):

    def __init__(self):
        super(NeuralNet, self).__init__()
        self.layer_1 = layers.Dense(30, activation='relu')
        self.layer_2 = layers.Dense(20, activation='relu')
        self.layer_3 = layers.Dense(10, activation='relu')
        self.layer_4 = layers.Dense(5, activation='relu')
        self.output_layer = layers.Dense(1, activation='sigmoid')

        self.train_X = None
        self.train_Y = None
        self.test_X = None
        self.test_Y = None
        self.val_X = None
        self.val_Y = None

    def call(self, inputs):
        assert inputs.dtype == tf.float32
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        return self.output_layer(x)

    def neuralnet_compile(self):
        self.compile(optimizer=keras.optimizers.Adam(),
                     loss=keras.losses.BinaryCrossentropy(),
                     metrics=['accuracy'])

    def neuralnet_fit(self, train_dataset, test_dataset):
        self.train_X = train_dataset[:, 0:21].astype(float)
        self.train_Y = train_dataset[:, 21]

        self.test_X = test_dataset[:, 0:21].astype(float)
        self.test_Y = test_dataset[:, 21]

        self.val_X = self.train_X[-10000:]
        self.val_Y = self.train_Y[-10000:]
        self.train_X = self.train_X[:-10000]
        self.train_Y = self.train_Y[:-10000]

        self.fit(self.train_X, self.train_Y,
                 batch_size=64,
                 epochs=100,
                 validation_data=(self.val_X, self.val_Y))

    def neuralnet_evaluate(self):
        results = self.evaluate(self.test_X, self.test_Y, batch_size=128)
        print('Test loss, Test accuracy:', results)


def get_dataset():
    root_path = '../../../data/'
    train_df = read_csv(f'{root_path}training_set.csv')
    test_df = read_csv(f'{root_path}test_set.csv')

    train_dataset = train_df.values
    test_dataset = test_df.values
    return (train_dataset, test_dataset)


def main():

    train_dataset, test_dataset = get_dataset()

    model = NeuralNet()
    model.neuralnet_compile()
    model.neuralnet_fit(train_dataset, test_dataset)
    model.neuralnet_evaluate()
    model.save('model_3')

if __name__ == '__main__':
    main()
