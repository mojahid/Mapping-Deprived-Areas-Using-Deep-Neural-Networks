import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import os
from keras.models import Model

#os.system("sudo pip install cv2")
from kerastuner import RandomSearch
from sklearn.model_selection import GridSearchCV, KFold
import numpy as np
import cv2
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from PIL import Image
from numpy import asarray
import pandas as pd
import itertools
from sklearn.metrics import roc_auc_score,confusion_matrix, classification_report

import seaborn as sns

from project_root import get_project_root
root = get_project_root()



#os.system("sudo unzip mixed.zip")
#os.system("sudo unzip OB.zip")
#os.system("sudo unzip raw.zip")

#os.system("sudo unzip Train42_png.zip")



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)


    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')






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


def image_path(path):

    labels_paths = []

    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if(file_path.endswith("png")):
            labels_paths.append(file_path)
    return labels_paths



def images(images_paths):
    x = []
    for img in images_paths:
        x.append(cv2.imread(img))
    x = np.array(x)
    return x

MODEL_SERIAL= "OB_dual"

#print(os.getcwd() )



############################################## MLP INPUT & VALIDATION DATA & TEST DATA #######################################################

# MLP Training data

# Read open building csv file
open_building_train_data= pd.read_csv(root / '1.Data' /"Mixed_data"/ "OB_Coordinates_training.csv")

# create dataframe for features
features_df= open_building_train_data[["Mean_Area","Median_Area","Building_Count","Max_Area","Min_Area"]]

# convert dataframe to numpy array ( shape = ( length of data, 5)
x_train_mlp = features_df.to_numpy()

# Create training labels ( Same as Raw image and open building images )
y_train= open_building_train_data["Label"].tolist()

# Convert labels to numpy array ( shape = (length of data, )
y_train= np.array(y_train)

ohe = OneHotEncoder()
y_train=y_train.reshape(-1,1)
y_train = ohe.fit_transform(y_train).toarray()
print(y_train)



print("Shape of MLP Input data ", x_train_mlp.shape)

print("Shape of training label", y_train.shape)
#---------------------------------------------------------------------------------------------------------

# MLP Validation data

# Read open building csv file
open_building_val_data= pd.read_csv(root / '1.Data' /"Mixed_data"/ "OB_Coordinates_validation.csv")

# create dataframe for features
features_val_df= open_building_val_data[["Mean_Area","Median_Area","Building_Count","Max_Area","Min_Area"]]

# convert dataframe to numpy array ( shape = ( length of data, 5)
x_val_mlp = features_val_df.to_numpy()

# Create validation labels ( Same as Raw image and open building images )
y_val= open_building_val_data["Label"].tolist()

# Convert labels to numpy array ( shape = (length of data, )
y_val= np.array(y_val)

ohe = OneHotEncoder()
y_val=y_val.reshape(-1,1)
y_val = ohe.fit_transform(y_val).toarray()
print(y_val)



print("Shape of MLP Validation data ", x_val_mlp.shape)

print("Shape of Validation label", y_val.shape)
#--------------------------------------------------------------------------------------------------------------

# Testing data

open_building_test_data= pd.read_csv(root / '1.Data' /"Mixed_data"/ "OB_Coordinates_test.csv")

# create dataframe for features
features_test_df= open_building_test_data[["Mean_Area","Median_Area","Building_Count","Max_Area","Min_Area"]]

# convert dataframe to numpy array ( shape = ( length of data, 5)
x_test_mlp = features_test_df.to_numpy()

# Create testing labels ( Same as Raw image and open building images )
y_test= open_building_test_data["Label"].tolist()

# Convert labels to numpy array ( shape = (length of data, )
y_test= np.array(y_test)

ohe = OneHotEncoder()
y_test=y_test.reshape(-1,1)
y_test = ohe.fit_transform(y_test).toarray()
print(y_test)

print("Shape of MLP test data ", x_test_mlp.shape)

print("Shape of testing label", y_test.shape)

################################## RAW IMAGE CNN INPUT & VALIDATION DATA & TEST DATA #########################################################


# Training images

raw_path_train="1.Data/Mixed_data/raw_images/train"

# Generate images path list
raw_train_images_path_lst= image_path(raw_path_train)

# Read Images to numpy arrays
x_train_raw= images(raw_train_images_path_lst)

#print(np.amax(x_train_raw, axis=0))
# Scale images
x_train_raw= x_train_raw / 255

# Print shape of raw images training data ( shape = ( length of data , 10 , 10 , 3)
print("Shape of Raw Images CNN Input data ", x_train_raw.shape)
#------------------------------------------------------------------------------------------------------------------------
# Validation images

raw_path_val="1.Data/Mixed_data/raw_images/validation"

# Generate images path list
raw_val_images_path_lst= image_path(raw_path_val)

# Read Images to numpy arrays
x_val_raw= images(raw_val_images_path_lst)

#print(np.amax(x_train_raw, axis=0))
# Scale images
x_val_raw= x_val_raw / 255

# Print shape of raw images validation data ( shape = ( length of data , 10 , 10 , 3)
print("Shape of Raw Images CNN Validation data ", x_val_raw.shape)
#-------------------------------------------------------------------------------------------------------------------

# Testing images


raw_path_test="1.Data/Mixed_data/raw_images/test"

# Generate images path list
raw_test_images_path_lst= image_path(raw_path_test)

# Read Images to numpy arrays
x_test_raw= images(raw_test_images_path_lst)

# Scale images
x_test_raw= x_test_raw / 255

# Print shape of raw images testing data ( shape = ( length of data , 10 , 10 , 3)
print("Shape of Raw Images CNN test data ", x_test_raw.shape)

########################################## OPEN BUILDING CNN INPUT & VALIDATION DATA & TEST DATA ############################################################################

# Training images

OB_path_train="1.Data/Mixed_data/ob_images/train"

# Generate images path list
OB_train_images_path_lst= image_path(OB_path_train)

# Read Images to numpy arrays
x_train_OB= images(OB_train_images_path_lst)

#print(np.amax(x_train_OB, axis=2))

# Scale images
x_train_OB= x_train_OB / 76

# Print shape of raw images training data ( shape = ( length of data , 100 , 100 , 3)
print("Shape of OB Images CNN Input data ", x_train_OB.shape)
#---------------------------------------------------------------------------------------------------------------------

# Validation images

OB_path_val="1.Data/Mixed_data/ob_images/validation"

# Generate images path list
OB_val_images_path_lst= image_path(OB_path_val)

# Read Images to numpy arrays
x_val_OB= images(OB_val_images_path_lst)

#print(np.amax(x_train_OB, axis=2))

# Scale images
x_val_OB= x_val_OB / 76

# Print shape of raw images validation data ( shape = ( length of data , 100 , 100 , 3)
print("Shape of OB Images CNN Validation data ", x_val_OB.shape)

#------------------------------------------------------------------------------------------------------------------

# Testing images


OB_path_test="1.Data/Mixed_data/ob_images/test"

# Generate images path list
OB_test_images_path_lst= image_path(OB_path_test)

# Read Images to numpy arrays
x_test_OB= images(OB_test_images_path_lst)

# Scale images
x_test_OB= x_test_OB / 76

# Print shape of raw images testing data ( shape = ( length of data , 100 , 100 , 3)
print("Shape of OB Images CNN test data ", x_test_OB.shape)

################################################# RAW IMAGE CNN MODEL ###############################################


def RAW_CNN(input_img):
    model = layers.Conv2D(100, 3, padding='same', input_shape= (10,10,3))(input_img)
    model = layers.ReLU()(model)
    model= layers.BatchNormalization()(model)
    model = layers.MaxPooling2D()(model)
    model = layers.Conv2D(120, 3, padding='same')(model)
    model = layers.ReLU()(model)
    model = layers.BatchNormalization()(model)
    model = layers.MaxPooling2D()(model)
    model = layers.Conv2D(150, 3, padding='same')(model)
    model = layers.ReLU()(model)
    model = layers.BatchNormalization()(model)
    model = layers.MaxPooling2D()(model)
    model= layers.Flatten()(model)
    model = layers.Dense(100)(model)


    return model


######################################### OPEN BUILDING CNN MODEL #########################################################
def OB_CNN(input_img):
    model = layers.Conv2D(100, 3, padding='same', input_shape= (100,100,3))(input_img)
    model = layers.ReLU()(model)
    model= layers.BatchNormalization()(model)
    model = layers.MaxPooling2D()(model)
    model = layers.Conv2D(150, 3, padding='same')(model)
    model = layers.ReLU()(model)
    model = layers.BatchNormalization()(model)
    model = layers.MaxPooling2D()(model)
    model = layers.Conv2D(200, 3, padding='same')(model)
    model = layers.ReLU()(model)
    model = layers.BatchNormalization()(model)
    model = layers.MaxPooling2D()(model)
    model = layers.Conv2D(200, 3, padding='same')(model)
    model = layers.ReLU()(model)
    model = layers.BatchNormalization()(model)
    model = layers.MaxPooling2D()(model)
    model= layers.Flatten()(model)
    model = layers.Dense(100)(model)


    return model

################################# OPEN BUILDING MLP MODEL ################################################

def MLP(input):
    model = layers.Dense(20, input_shape= (5,))(input)
    model = layers.ReLU()(model)
    model= layers.Dense(15)(model)
    model = layers.ReLU()(model)
    model= layers.Dense(10)(model)
    model = layers.ReLU()(model)

    return model

#######################################################################################################

# RAW CNN INPUT & MODEL
raw_cnn_input= layers.Input( shape= (10,10,3))
raw_cnn_model= RAW_CNN(raw_cnn_input)

# OB CNN INPUT & MODEL
ob_cnn_input= layers.Input( shape= (100,100,3))
ob_cnn_model= OB_CNN(ob_cnn_input)

# OB MLP INPUT & MODEL
mlp_input= layers.Input(shape= (5,))
mlp_model= MLP(mlp_input)
print(mlp_model)

combined= layers.concatenate([raw_cnn_model,ob_cnn_model, mlp_model])

x = layers.Dense(210, activation= "relu")(combined)
x = layers.Dense(50, activation= "relu")(x)
x = layers.Dense(25, activation= "relu")(x)
x = layers.Dense(10, activation= "relu")(x)
output = layers.Dense(2, activation='softmax')(x)


model = Model(inputs=[raw_cnn_input, ob_cnn_input, mlp_input], outputs=[output])

model.compile(optimizer='adamax',
              loss=tf.keras.losses.categorical_crossentropy ,
              metrics=['accuracy'])

print(model.summary())

early_stop = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=6)
check_point = tf.keras.callbacks.ModelCheckpoint('model_best{}.h5'.format(MODEL_SERIAL),
                                                 monitor='accuracy',
                                                 save_best_only=True)
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

batch_size= 64
epochs= 100
history = model.fit(
  [x_train_raw, x_train_OB, x_train_mlp], y_train,
    batch_size= batch_size,
    shuffle=True ,
    verbose=1 ,
  validation_data=([x_val_raw, x_val_OB, x_val_mlp], y_val),
  epochs=epochs,
  callbacks=[early_stop, check_point, callback]
)


############################################ TESTING ###################################################
#get the predictions for the test data
predicted_classes = model.predict([x_test_raw, x_test_OB, x_test_mlp])

open_building_test_data["Prediction"] = np.argmax(predicted_classes,axis=1)
y_test_pred= np.array(open_building_test_data["Prediction"].tolist())
y_test= np.array(open_building_test_data["Label"].tolist())

sns.set(style="white")
cnf_matrix = confusion_matrix(y_test,y_test_pred)
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
class_names = ["Built-up","Deprived"]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()

print(classification_report(y_test, y_test_pred))


open_building_test_data.to_csv('Prediction.csv')
