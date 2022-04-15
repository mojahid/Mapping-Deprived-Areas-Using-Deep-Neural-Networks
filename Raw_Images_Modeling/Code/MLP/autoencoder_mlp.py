import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, losses


epochs =150
batch_size = 32
img_height = 10
img_width = 10
SEED= 42
LR = 1e-3
DROPOUT=0.1
input_dim = (10,10,3) # Image dimension
output_dim= 2
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats


#Import Data
path = r'../data/train/png/0'

def image_path(path):

    labels_paths = []

    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if(file_path.endswith('png')):
            labels_paths.append(file_path)
    return labels_paths


def images(images_paths):
    x = []
    for img in images_paths:
        x.append(cv2.imread(img))
    x = np.array(x)
    return x

builtup_images_path_lst= image_path(path)
#print(builtup_images_path_lst)
builtup_input= images(builtup_images_path_lst)
builtup_label=np.zeros((len(builtup_input,)))
#print(builtup_input[0:2])
#print(builtup_label[0:10])
#print(builtup_input.shape)
#print(builtup_label.shape)

path_2=r'../dog-cat/train/1'#cat
deprived_images_path_lst= image_path(path_2)
#print(deprived_images_path_lst)
deprived_input= images(deprived_images_path_lst)
deprived_label=np.ones((len(deprived_input,)))
for i in deprived_images_path_lst:
    builtup_images_path_lst.append(i)
x_train= images(builtup_images_path_lst)
x_train=x_train/255
y_train= np.append(builtup_label,deprived_label)


test_path=r'../dog-cat/test/0'#dog
test_builtup_images_path_lst= image_path(test_path)
#print(builtup_images_path_lst)
test_builtup_input= images(test_builtup_images_path_lst)
test_builtup_label=np.zeros((len(test_builtup_input,)))

test_path_2=r'../dog-cat/test/1'#cat
test_deprived_images_path_lst= image_path(test_path_2)
#print(deprived_images_path_lst)
test_deprived_input= images(test_deprived_images_path_lst)
test_deprived_label=np.ones((len(test_deprived_input,)))
for i in test_deprived_images_path_lst:
    test_builtup_images_path_lst.append(i)

x_test= images(test_builtup_images_path_lst)
x_test= x_test/255
y_test= np.append(test_builtup_label,test_deprived_label)

print(x_train.shape)
print(x_test.shape)


#Define the Autoencoder
latent_dim = 10
class Autoencoder(tf.keras.Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential([
         tf.keras.layers.Flatten(input_shape=(10, 10,3)),
         tf.keras.layers.Dense(100, activation='relu'),
         tf.keras.layers.BatchNormalization(),
         tf.keras.layers.Dense(50, activation='relu'),
         tf.keras.layers.BatchNormalization(),
         tf.keras.layers.Dense(25, activation='relu'),
         tf.keras.layers.BatchNormalization(),
         tf.keras.layers.Dense(latent_dim, activation='relu')
    ])
    self.decoder = tf.keras.Sequential([
         tf.keras.layers.Dense(latent_dim, activation='relu'),
         tf.keras.layers.BatchNormalization(),
         tf.keras.layers.Dense(25, activation='relu'),
         tf.keras.layers.BatchNormalization(),
         tf.keras.layers.Dense(50, activation='relu'),
         tf.keras.layers.BatchNormalization(),
         tf.keras.layers.Dense(100, activation='relu'),
         tf.keras.layers.BatchNormalization(),
         tf.keras.layers.Dense(300, activation='linear'),
         tf.keras.layers.Reshape((10,10,3))
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = Autoencoder(latent_dim)
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

autoencoder.fit(x_train,x_train,epochs=10,shuffle=True,validation_data=(x_test, x_test))

encoded_imgs = autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
  # display original
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(x_test[i])
  plt.title("original")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # display reconstruction
  ax = plt.subplot(2, n, i + 1 + n)
  plt.imshow(decoded_imgs[i])
  plt.title("reconstructed")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()