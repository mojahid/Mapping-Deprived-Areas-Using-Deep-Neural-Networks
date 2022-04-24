import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import os
os.system("sudo pip install keras-tuner")
from kerastuner import RandomSearch



#os.system("sudo unzip Raw_Images.zip")
#os.system("sudo unzip Google_Images.zip")


#ata_dir = r"/home/ubuntu/cnn_2/Raw_Images/Train/png/"
data_dir = r"/home/ubuntu/cnn_2/google/Google_Images/Train/png/"


epochs = 100
batch_size = 32
img_height = 100
img_width = 100
MODEL_SERIAL = 'Google_Tuned'


def scheduler(epoch, lr):
  lr_0 = lr
  lr_max = 0.001
  lr_0_steps = 5
  lr_max_steps = 5
  lr_min = 0.0005
  lr_decay = 0.9
  if epoch < lr_0_steps:
    lr = lr_0 + ((lr_max - lr_min) / lr_0_steps) * (epoch - 1)
  elif epoch < lr_0_steps + lr_max_steps:
    lr = lr_max
  else:
    lr = max(lr_max * lr_decay ** (epoch - lr_0_steps - lr_max_steps), lr_min)
  return lr



train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  label_mode="categorical",
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  label_mode="categorical",
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


#class_names = train_ds.class_names
#print(class_names)

data_augmentation = keras.Sequential(
  [
    layers.RandomFlip(input_shape=(img_height,
                                  img_width,
                                  3)),
    #layers.RandomRotation(0.01),
    #layers.RandomZoom(0.1),
  ])


print(train_ds)

def build_model(hp):
    # create model object
    model = keras.Sequential([
    data_augmentation,
    #adding first convolutional layer
    keras.layers.Conv2D(
        #adding filter
        filters=hp.Int('conv_1_filter', min_value=32, max_value=256,step=32),
        # adding filter size or kernel size
        kernel_size=hp.Choice('conv_1_kernel', values = [3,5]),
        #activation function
        activation='relu',
        input_shape=(100,100,3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size= (3,3)),

    # adding second convolutional layer
    keras.layers.Conv2D(
        #adding filter
        filters=hp.Int('conv_2_filter', min_value=32, max_value=128, step=32),
        #adding filter size or kernel size
        kernel_size=hp.Choice('conv_2_kernel', values = [3,5]),
        #activation function
        activation='relu'
    ),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size= (3,3)),

   keras.layers.Conv2D(
            # adding filter
        filters=hp.Int('conv_3_filter', min_value=32, max_value=128,step=32),
            # adding filter size or kernel size
        kernel_size=hp.Choice('conv_3_kernel', values=[3, 5]),
            # activation function
        activation='relu'
    ),
   keras.layers.BatchNormalization(),
   keras.layers.MaxPooling2D(),
        # adding flatten layer
    keras.layers.Flatten(),
        # adding dense layer
    keras.layers.Dense(
            units=hp.Int('dense_1_units', min_value=64, max_value=128, step=16),
            activation='relu'
        ),
        # output layer
    keras.layers.Dense(2, activation='sigmoid')
    ])
    # compilation of model
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


#creating randomsearch object
tuner = RandomSearch(build_model,
                    objective='val_accuracy',
                    max_trials = 10)
# search best parameter
tuner.search(train_ds,epochs=10,validation_data=(val_ds))

model=tuner.get_best_models(num_models=1)[0]
#summary of best model
print(model.summary())
#print(model)
#model.fit(train_ds,
         # epochs=10,
          #validation_split=0.1,
          #nitial_epoch=3)
check_point = tf.keras.callbacks.ModelCheckpoint('model_{}.h5'.format(MODEL_SERIAL),
                                                 monitor='accuracy',
                                                 save_best_only=True)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=5)
#class_weight = {0:1 , 1:500}#, 2:0.5 }

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks=[early_stop, check_point],

)

#file_name = 'CNN'
#model.save('file_name')