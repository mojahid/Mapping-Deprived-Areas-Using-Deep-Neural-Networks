import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import os
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




#os.system("sudo unzip Raw_Images.zip")
#os.system("sudo unzip Train42_png.zip")


#ata_dir = r"/home/ubuntu/cnn_2/Raw_Images/Train/png/"N
#data_dir = r"/home/ubuntu/CNN_MODEL/png/"

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



#print(os.getcwd() )

path=r"1.Data /raw_images/train/png/0/"
builtup_images_path_lst= image_path(path)
#print(builtup_images_path_lst)
builtup_input= images(builtup_images_path_lst)
builtup_label=np.zeros((len(builtup_input,)))
#print(builtup_input[0:2])
#print(builtup_label[0:10])
#print(builtup_input.shape)
#print(builtup_label.shape)

path_2=r"1.Data /raw_images/train/png/1/"
deprived_images_path_lst= image_path(path_2)
#print(deprived_images_path_lst)
deprived_input= images(deprived_images_path_lst)
deprived_label=np.ones((len(deprived_input,)))
#print(deprived_input[0:2])
#print(deprived_label[0:10])
#print(deprived_input.shape)
#print(deprived_label.shape)

for i in deprived_images_path_lst:
    builtup_images_path_lst.append(i)

x_train= images(builtup_images_path_lst)
x_train=x_train/255
y_train= np.append(builtup_label,deprived_label)
#print(train_x.shape)
#print(train_y.shape)

# one hot encode label for categorical_crossentropy ([0,1], [1,0] labels)
#ohe = OneHotEncoder()
#train_y=y_train.reshape(-1,1)
#train_y = ohe.fit_transform(train_y).toarray()
#print(train_y)

#x_train, x_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.2, random_state=42)


train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))

#val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))


# The GridSearchCV with 5 folds and this hyper-parameters takes ...
LR = [5e-4,3e-4 ]
N_NEURONS = [(50, 300, 300, 200, 100),(50, 300, 512, 512, 200) ]
N_EPOCHS = [40]
#BATCH_SIZE = [32, 64]
DROPOUT = [0.2]
ACTIVATION = ["relu"]


def build_clf(dropout=0.3,
                    activation='relu',
                    n_neurons=(50, 300, 300, 200, 100),
                    lr=1e-3):
    model = Sequential([
  layers.Conv2D(n_neurons[0], 3,input_shape= (10,10,3),  padding='same', activation=activation),
  layers.BatchNormalization(),
  layers.Conv2D(n_neurons[1], 3, padding='same', activation=activation),
  layers.BatchNormalization(),
  layers.MaxPooling2D(),
  layers.Conv2D(n_neurons[2], 3, padding='same', activation=activation),
  layers.BatchNormalization(),
  layers.MaxPooling2D(),
  layers.Conv2D(n_neurons[3], 3, padding='same', activation=activation),
  layers.Conv2D(n_neurons[4], 3, padding='same', activation=activation),
  layers.BatchNormalization(),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation=activation),
  layers.Dropout(dropout),
  layers.Dense(128, activation=activation),
  layers.Dropout(dropout),
  layers.Dense(1, activation="sigmoid")
])
    model.compile(optimizer=keras.optimizers.Adagrad(learning_rate=lr), loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

scoring="f1_macro"
gs=GridSearchCV(estimator=tf.keras.wrappers.scikit_learn.KerasClassifier(
        build_fn= build_clf
    ),
    scoring= scoring,
    param_grid={
        'epochs': N_EPOCHS,  # The param grid must contain arguments of the usual Keras model.fit
        #"batch_size": BATCH_SIZE,  # and/or the arguments of the construct_model function
        'dropout': DROPOUT,
        "activation": ACTIVATION,
        'n_neurons': N_NEURONS,
        'lr': LR,
    },
    n_jobs=1,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    verbose=10,
return_train_score=True
)

def display_cv_results(search_results):
    print('Best score = {:.4f} using {}'.format(search_results.best_score_, search_results.best_params_))
    means = search_results.cv_results_['mean_test_score']
    stds = search_results.cv_results_['std_test_score']
    params = search_results.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print('mean test accuracy +/- std = {:.4f} +/- {:.4f} with: {}'.format(mean, stdev, param))

grid_search_results = gs.fit(x_train, y_train)


display_cv_results(grid_search_results)

best_score_params_estimator_gs = []
# Update best_score_params_estimator_gs
best_score_params_estimator_gs.append([grid_search_results.best_score_, grid_search_results.best_params_, grid_search_results.best_estimator_])

# Sort cv_results in ascending order of 'rank_test_score' and 'std_test_score'
cv_results = pd.DataFrame.from_dict(grid_search_results.cv_results_).sort_values(by=['rank_test_score', 'std_test_score'])

# Get the important columns in cv_results
important_columns = ['rank_test_score',
                     'mean_test_score',
                     'std_test_score',
                     'mean_train_score',
                     'std_train_score',
                     'mean_fit_time',
                     'std_fit_time',
                     'mean_score_time',
                     'std_score_time']

# Move the important columns ahead
cv_results = cv_results[important_columns + sorted(list(set(cv_results.columns) - set(important_columns)))]

# Write cv_results file
cv_results.to_csv(path_or_buf='/home/ubuntu/CNN_MODEL/' +'CNN_GridSearchCV_binary_3.csv', index=False)

# Sort best_score_params_estimator_gs in descending order of the best_score_
best_score_params_estimator_gs = sorted(best_score_params_estimator_gs, key=lambda x: x[0], reverse=True)

# Print best_score_params_estimator_gs
best = pd.DataFrame(best_score_params_estimator_gs, columns=['best_score', 'best_param', 'best_estimator'])





best_model = grid_search_results.best_estimator_

grid_search_results.best_estimator_.model.save("CNN_GridSearch_BESTMODEL.hdf5")

#callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
#early_stop = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=5)
#check_point = tf.keras.callbacks.ModelCheckpoint('model_{}.h5'.format("Best_categorical_2"),
            #                                     monitor='accuracy',
             #                                    save_best_only=True)

#history = best_model.fit(
 #   (x_train, y_train),
  #  validation_data = (x_val, y_val),
   # epochs = 100,
    #callbacks = [early_stop, check_point]
#)
##################################################### MODEL TESTING####################################################################

test_path=r"1.Data /raw_images/test/png/0/"
test_builtup_images_path_lst= image_path(test_path)
#print(builtup_images_path_lst)
test_builtup_input= images(test_builtup_images_path_lst)
test_builtup_label=np.zeros((len(test_builtup_input,)))

test_path_2=r"1.Data /raw_images/test/png/0/"
test_deprived_images_path_lst= image_path(test_path_2)
#print(deprived_images_path_lst)
test_deprived_input= images(test_deprived_images_path_lst)
test_deprived_label=np.ones((len(test_deprived_input,)))

for i in test_deprived_images_path_lst:
    test_builtup_images_path_lst.append(i)

x_test= images(test_builtup_images_path_lst)
x_test= x_test/255
y_test= np.append(test_builtup_label,test_deprived_label)
#print(train_x.shape)


print(y_test)
y_test_pred= best_model.predict(x_test)
print(y_test_pred)
print(len(y_test_pred))
#y_test_pred=[]
#for i in y_pred:
 #   if i < 0.5:
  #      y_test_pred.append(0)
   # else:
    #    y_test_pred.append(1)

y_test_pred=np.array(y_test_pred)
# Plotting confusion matrix obtained from the testing data predictions
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


