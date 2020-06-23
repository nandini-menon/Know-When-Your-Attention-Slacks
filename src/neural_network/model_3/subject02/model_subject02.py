from pandas import read_csv
from tensorflow import keras
from tensorflow.keras import layers


train_df = read_csv(f'training_set2.csv')
test_df = read_csv(f'test_set2.csv')

train_dataset = train_df.values
test_dataset = test_df.values

train_X = train_dataset[:, 0:21].astype(float)
train_Y = train_dataset[:, 21]

test_X = test_dataset[:, 0:21].astype(float)
test_Y = test_dataset[:, 21]

x_val = train_X[-100:]
y_val = train_Y[-100:]
train_X = train_X[:-100]
train_Y = train_Y[:-100]

inputs = keras.Input(shape=(21,))
x = layers.Dense(30, activation='relu')(inputs)
x = layers.Dense(20, activation='relu')(x)
x = layers.Dense(10, activation='relu')(x)
x = layers.Dense(5, activation='relu')(x)
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

model.save('model_subject02.h5')
