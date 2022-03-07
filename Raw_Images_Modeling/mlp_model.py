import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential

from keras.preprocessing.image import load_img

import keras
import tensorflow as tf

img_szie = 10
epochs = 4

directory = '../data/png_image'
class_types = os.listdir(directory)
print(class_types)
print('Classes found:', len(class_types))

# List all images with correspondent labels.

images = []

for item in class_types:
    # Get all the file names
    all_images = os.listdir(directory + '/' + item)

    # create a list of all imag lables
    for image in all_images:
        images.append((item, str(directory + '/' + item) + '/' + image))
        # print(images[:1])

# Build a DF
images_df = pd.DataFrame(data=images, columns=['class', 'image'])
print(images_df.head())
# Count of all images
print(len(images_df))
# Count of images for each class
print(images_df['class'].value_counts())

imgs = []
labels = []
for i in class_types:
    imgs_path = directory + '/' + str(i)
    # print(i)
    # print(imgs_path)
    filenames = [i for i in os.listdir(imgs_path)]
    # print(filenames)

    for f in filenames:
        # print(imgs_path + '/' + f)
        # img = cv2.imread(imgs_path + '/' + f)  # reading image as array
        img = cv2.imread(imgs_path + '/' + f)  # reading image as array
        # print(img)
        # print(img.shape)
        img = cv2.resize(img, (img_szie, img_szie))
        # img = cv2.resize(img, (10, 10), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
        imgs.append(img)
        # labels.append(i)

# print(len(imgs))
# print((imgs[0]))
# print(((imgs[0]).shape))
# transform the image arry to numpy array
imgs = np.array(imgs)
print('**************************')
print(imgs.shape)

# normlize the list
imgs = imgs.astype('float32') / 255.0
print(imgs.shape)

print('*****forma the lables*****')
# forma the lables
y = images_df['class'].values
print(y[:5])

y_labelencoder = LabelEncoder()
y = y_labelencoder.fit_transform(y)
print(y)

imgs, y = shuffle(imgs, y, random_state=123)
train_x, test_x, train_y, test_y = train_test_split(imgs, y, test_size=0.05, random_state=123)

print('#check the shape of the training and testing images')
# check the shape of the training and testing images
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

# Strat Images Classificatio

model = keras.Sequential([keras.layers.Flatten(input_shape=(10, 10, 3)),
                          keras.layers.Dense(256, activation=tf.nn.tanh),
                          keras.layers.Dense(3, activation=tf.nn.softmax)
                          ])

print(model.summary())

# compute the model

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# train the model with 4 epochs
model.fit(train_x, train_y, epochs=epochs)

y_pred = model.predict(test_x)
print(y_pred)

# Test images
image = load_img('../data/png_image/1/clipped_232.png', target_size=(10, 10))
print(image)
image = np.array(image)
print(image.shape)
# '../data/png_image/1/clipped_232.png' class 0

image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

yhat = model.predict(image)
print(yhat)
