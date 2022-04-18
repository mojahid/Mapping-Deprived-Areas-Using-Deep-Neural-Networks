import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import cv2
from keras import layers, losses
from PIL import Image


MODEL = 'encoder_model'
data_dir = r'../data/test/png'
models = r'../code/ae_models/{}.h5'

decoder_model = tf.keras.models.load_model(models.format(MODEL))

# Check its architecture
decoder_model.summary()