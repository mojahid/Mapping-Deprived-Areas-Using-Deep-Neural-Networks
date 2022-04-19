import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import cv2
from keras import layers, losses
from PIL import Image


epochs =5
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
save_model_path =r'../code/ae_models/'
save_images_path = r'../data/ae_data/reconstructed/'

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
builtup_input= images(builtup_images_path_lst)
builtup_label=np.zeros((len(builtup_input,)))


path_2=r'../data/train/png/1'
deprived_images_path_lst= image_path(path_2)
#print(deprived_images_path_lst)
deprived_input= images(deprived_images_path_lst)
deprived_label=np.ones((len(deprived_input,)))
for i in deprived_images_path_lst:
    builtup_images_path_lst.append(i)
x_train= images(builtup_images_path_lst)
x_train=x_train/255
y_train= np.append(builtup_label,deprived_label)


test_path=r'../data/test/png/0'
test_builtup_images_path_lst= image_path(test_path)
#print(builtup_images_path_lst)
test_builtup_input= images(test_builtup_images_path_lst)
test_builtup_label=np.zeros((len(test_builtup_input,)))

test_path_2=r'../data/test/png/1'
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
  def __init__(self,input_dim,latent_dim):
    super(Autoencoder, self).__init__()
    self.input_dim = input_dim
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential([
         tf.keras.layers.Flatten(input_shape=input_dim),
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
         tf.keras.layers.Reshape(input_dim)
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = Autoencoder(input_dim,latent_dim)
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError(),metrics=['accuracy'])

autoencoder.build((None,10,10,3))
autoencoder.summary()

#define the enconder model
encoder_model=autoencoder.encoder
encoder_model.summary()

#define the decoder model
decoder_model=autoencoder.decoder
decoder_model.summary()


#fit the autoencoder model
early_stop = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=20)
checkpoint = tf.keras.callbacks.ModelCheckpoint(save_model_path+'weights{}.h5'.format('adam_latent_dim_10'), save_weights_only = True, verbose=0)
history=autoencoder.fit(x_train,x_train,epochs=epochs,shuffle=True,validation_data=(x_test, x_test),callbacks=[early_stop,checkpoint],batch_size=38)

#save the encoder model
encoder_model.save('encoder_model.h5')

#create a set od decoded images
encoded_imgs = autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

# plot loss
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


n = 16
plt.figure(figsize=(20, 4))
for i in range(n):
  # display original
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(x_test[i])
  plt.title('original')
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # display reconstruction
  ax = plt.subplot(2, n, i + 1 + n)
  plt.imshow(decoded_imgs[i])
  decoded_img=decoded_imgs[i]*255
  decoded_img=np.uint8(decoded_img)
  #print(decoded_img)
  PIL_image = Image.fromarray(decoded_img, 'RGB')
  #PIL_image.save(save_images_path+'reconstructed_{}.png'.format(i))
  plt.title('reconstructed')
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()