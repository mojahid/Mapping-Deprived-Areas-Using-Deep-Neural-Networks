import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


# This file contains the necessary functions to evaluate specific model against the test dataset
# it uses the model naming sequence to load the model and evaluate prediction and measure performance

MODEL = 'adagrad_1'
data_dir = r'../data/test/png'
models = r'../code/model_{}.h5'

dir0 =  r'../data/test/png/0'
dir1 = r'../data/test/png/1'


IMAGE_SIZE = 10
THRESHOLD = 0.5

def predict_func(test_ds,df):
    """ This function loads model (based on file name sequence) evaluate model against the data set given in test_ds
    Keyword arguments:
    test_ds -- tensor dataset holding the images
    model -- file sequence for the model to be loaded
    """
    final_model = tf.keras.models.load_model(models.format(MODEL))
    # Since models are using softmax with three output parameters, the predict function
    # will always take the second prediction of the softmaxprint(final_model.summary)
    # (representing photoshopped images) if results has a shape of 2
    # if not then it takes the results as is
    res = final_model.predict(test_ds)

    df['Class_0'] = res[:, 0]
    df['Class_1'] = res[:, 1]
    #df['Class_2'] = res[:, 2]

    df.loc[(df.Class_0 > df.Class_1), "results"] = 0
    df.loc[(df.Class_1 >= df.Class_0), "results"] = 1
    #df.loc[(df.Class_2 > df.Class_0) & (df.Class_2 > df.Class_1), "results"] = 2

    #df['result1'] = res2
    df.to_excel('results_{}.xlsx'.format(MODEL), index=False)
    return df

def load_data(data_dir):
  """ This function utilizes the built-in function image_dataset_from_directory() to load
  the images.
  """
  test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        shuffle=False,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=10)

  return test_ds


def prepare_dataframe():
  """ This function prepares a new dataframe to hold information about the dataset (train or test)
  The dataframe will contain image_name, size and shape
  Keyword arguments:
  orig_dir -- directory for original photos (which can be train or test)
  ps_dir -- directory for photoshopped photos (which can be train or test)
  """
  files1 = []
  files2 = []
  #files3 = []

  # This function will loop each folder separately and will populate the dataframe
  # with the necessary information and then merge dataframe created out of each folder
  for file in sorted(os.listdir(dir0)):
    files1.append((file, 0))

  for file in sorted(os.listdir(dir1)):
    files2.append((file, 1))

  #for file in sorted(os.listdir(dir2)):
    #files3.append((file, 2))

  df1 = pd.DataFrame(files1, columns=['FileName', 'Class'])
  df2 = pd.DataFrame(files2, columns=['FileName', 'Class'])
  #df3 = pd.DataFrame(files3, columns=['FileName', 'Class'])

  df = pd.concat([df1, df2])#,df3])
  return df

def measure_model(y_true, y_pred):
    """ This function is used in the test MODE and it uses the predicted and original targets to measure
    model perfomace and draw a heatmap for the confusion matrix
    Keyword arguments:
    y_true -- original target list
    y_pred -- predicted target list from the two phased prediction
    """
    confusion = confusion_matrix(y_true, y_pred)
    print(classification_report(y_true, y_pred))
    print('****************')
    # print(confusion)
    ax = sns.heatmap(confusion, annot=True, cmap='Blues', fmt='d')
    ax.set_title(' Confusion Matrix for model_'+ MODEL)
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')
    ax.xaxis.set_ticklabels(['0', '1'])
    ax.yaxis.set_ticklabels(['0', '1'])
    plt.savefig('C_M_model_{}_rg.png'.format(MODEL))
    plt.show()
    return

## MAIN
# Load training or test data
# This data will be used against the individual models predictions

test_ds = load_data(data_dir)

# Prepre the dataframe that will aggregate all model predictions
df = prepare_dataframe()

# If the model uses ELA then use test_ds_02 if the model use normal image then use test_ds_01
df = predict_func(test_ds, df)

y_true = np.array(df['Class'])
y_pred = np.array(df['results'])

# Measure the model
measure_model(y_true, y_pred)