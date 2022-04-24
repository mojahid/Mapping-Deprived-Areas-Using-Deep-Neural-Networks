import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

epochs = 150
batch_size = 32
img_height = 10
img_width = 10
SEED= 42
LR = 1e-3
DROPOUT=0.1

train_data = r'../data/train/png'
test_data = r'../data/test/png'

#Import Data
train_ds = tf.keras.utils.image_dataset_from_directory(train_data,
                                                       validation_split=0.2,
                                                       subset="training",
                                                       seed=123,
                                                       image_size=(img_height, img_width),
                                                       batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(train_data,
                                                     validation_split=0.2,
                                                     subset="validation",
                                                     seed=123,
                                                     image_size=(img_height, img_width),
                                                     batch_size=batch_size)

test_ds = tf.keras.utils.image_dataset_from_directory(test_data,
                                                      shuffle=False,
                                                      image_size=(img_height, img_width),
                                                      batch_size=10)

# Strat Images Classificatio
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(10, 10,3)))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(125, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dropout(DROPOUT, seed=SEED))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(25, activation='relu'))
model.add(tf.keras.layers.Dropout(DROPOUT, seed=SEED))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(2, activation='softmax'))
print(model.summary())

# compute the model
model.compile(optimizer='adagrad',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# fit model
early_stop = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=5)
check_point = tf.keras.callbacks.ModelCheckpoint('model_{}.h5'.format('adagrad_1'), monitor='accuracy',save_best_only=True)
history = model.fit(train_ds,validation_data=val_ds,epochs=epochs,callbacks=[early_stop, check_point],batch_size=38)

# plot loss during training
fig, axs = plt.subplots(2, 1, figsize=(15, 15))
axs[0].plot(history.history['loss'])
axs[0].plot(history.history['val_loss'])
axs[0].title.set_text('Training Loss vs Validation Loss')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].legend(['Train','Val'])
# plot accuracy during training
axs[1].plot(history.history['accuracy'])
axs[1].plot(history.history['val_accuracy'])
axs[1].title.set_text('Training Accuracy vs Validation Accuracy')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Accuracy')
axs[1].legend(['Train', 'Val'])
plt.show()

# predict probabilities for test set
#yhat_probs = model.predict(test_data, verbose=0)
#yhat_probs = yhat_probs[:, 0]