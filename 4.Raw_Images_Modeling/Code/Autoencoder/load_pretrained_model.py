import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import cv2
from keras import layers, losses
from PIL import Image

MODEL = 'encoder_model'
MODEL2 = 'pretrained_encoder_model'
data_dir = r'../data/test/png'
models = r'../code/ae_models/{}.h5'
model_path=r'../code/ae_models/'

model = tf.keras.models.load_model(models.format(MODEL))
model.add(tf.keras.layers.BatchNormalization(name='batch_normalization_3'))
model.add(tf.keras.layers.Dense(2, activation='softmax',name='dense_4'))
#print(model.layers.pop())
# Check its architecture
model.summary()

model.save(model_path+'model_pretrained_encoder_model.h5')

model2 = tf.keras.models.load_model(models.format(MODEL2))
model2.summary()