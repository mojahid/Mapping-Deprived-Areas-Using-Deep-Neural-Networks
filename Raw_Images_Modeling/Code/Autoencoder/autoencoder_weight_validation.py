import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import cv2
from keras import layers, losses
from PIL import Image
import h5py

#this file use to compare the weights of auoencoder and submodel encoder
#so we can use the encoder model for image classification

MODEL = 'auoencoderadam_latent_dim_10_'
MODEL2 = 'encoder_model_weights'
data_dir = r'../data/test/png'
models = r'../code/ae_models/{}.h5'
model_path=r'../code/ae_models/'


f1 = h5py.File(models.format(MODEL))
f2= h5py.File(models.format(MODEL2))
w1= f1["sequential"]["batch_normalization"]

#for key, value in f.attrs.items():
 #   print(key, value)

def print_structure(weight_file_path):
    """
    Prints out the structure of HDF5 file.

    Args:
      weight_file_path (str) : Path to the file to analyze
    """
    f = h5py.File(weight_file_path)
    try:
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")

        print("  f.attrs.items(): ")
        for key, value in f.attrs.items():
            print("  {}: {}".format(key, value))

        if len(f.items())==0:
            print("  Terminate # len(f.items())==0: ")
            return

        print("  layer, g in f.items():")
        for layer, g in f.items():
            print("  {}".format(layer))
            print("    g.attrs.items(): Attributes:")
            for key, value in g.attrs.items():
                print("      {}: {}".format(key, value))

            print("    Dataset:")
            for p_name in g.keys():
                param = g[p_name]
                subkeys = param.keys()
                print("    Dataset: param.keys():")
                for k_name in param.keys():
                    print("      {}/{}: {}".format(p_name, k_name, param.get(k_name)[:]))
    finally:
        f.close()
#print_structure(models.format(MODEL2))
print_structure(models.format(MODEL))
# etc.


#print(f)
#print(list(f.keys()))
#print(list(f1.keys()))
#print(w1)
'''
model = tf.keras.models.load_model(models.format(MODEL))
for layer in model.layers:
    weights = layer.get_weights()
    print(weights)

model.add(tf.keras.layers.BatchNormalization(name='batch_normalization_3'))
model.add(tf.keras.layers.Dense(2, activation='softmax',name='dense_4'))
#print(model.layers.pop())
# Check its architecture
model.summary()

#model.save(model_path+'model_pretrained_encoder_model.h5')

model2 = tf.keras.models.load_model(models.format(MODEL2))
model2.summary()
'''