from pandas import read_csv
from tensorflow import keras
from tensorflow.keras import layers


root_path = '../../../data/'

train_df = read_csv(f'{root_path}training_set.csv')
test_df = read_csv(f'{root_path}test_set.csv')

train_dataset = train_df.values
test_dataset = test_df.values

train_X = train_dataset[:, 0:21].astype(float)
train_Y = train_dataset[:, 21]

test_X = test_dataset[:, 0:21].astype(float)
test_Y = test_dataset[:, 21]

x_val = train_X[-10000:]
y_val = train_Y[-10000:]
train_X = train_X[:-10000]
train_Y = train_Y[:-10000]

inputs = keras.Input(shape=(21,))
x = layers.Dense(11, activation='relu')(inputs)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

print('# Fit model on training data')
model.fit(x=train_X,
          y=train_Y,
          batch_size=64,
          epochs=100,
          validation_data=(x_val, y_val))

print('\n# Evaluate on test data')
results = model.evaluate(test_X, test_Y, batch_size=128)
print('test loss, test acc:', results)

model.save('model_1.h5')
