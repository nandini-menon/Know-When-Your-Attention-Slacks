from pandas import read_csv
from tensorflow import keras
from tensorflow.keras import layers

# from google.colab import drive

# drive.mount('/content/gdrive')
# root_path = 'gdrive/My Drive/MainProject/'

root_path = '../../data/'

train_df = read_csv(f'{root_path}training_set.csv')
test_df = read_csv(f'{root_path}test_set.csv')

train_dataset = train_df.values
test_dataset = test_df.values

train_X = train_dataset[:, 0:21].astype(float)
train_Y = train_dataset[:, 21]

test_X = test_dataset[:, 0:21].astype(float)
test_Y = test_dataset[:, 21]

# For validation
x_val = train_X[-10000:]
y_val = train_Y[-10000:]
train_X = train_X[:-10000]
train_Y = train_Y[:-10000]

inputs = keras.Input(shape=(21,))
x = layers.Dense(21, activation='relu')(inputs)
x = layers.Dense(10, activation='relu')(x)
x = layers.Dense(5, activation='relu')(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=keras.optimizers.Adam(),  # Optimizer
              # Loss function to minimize
              loss=keras.losses.BinaryCrossentropy(),
              # List of metrics to monitor
              metrics=['accuracy'])

print('# Fit model on training data')
history = model.fit(train_X, train_Y,
                    batch_size=64,
                    epochs=100,
                    # We pass some validation for
                    # monitoring validation loss and metrics
                    # at the end of each epoch
                    validation_data=(x_val, y_val))

print('\nhistory dict:', history.history)

# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
results = model.evaluate(test_X, test_Y, batch_size=128)
print('test loss, test acc:', results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print('\n# Generate predictions for 3 samples')
predictions = model.predict(test_X[:3])
print('predictions: ', predictions)

model.save('model_2.h5')