import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


# This file contains the necessary functions to evaluate different models and use hard voting to evaluate the
# final prediction of deprived area mapping based on the best models

raw_image_test_directory = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Test_Ensemble\raw_test"
OB_image_test_directory  = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Test_Ensemble\ob_test"


# Paths for the raw images
raw_image_dir0 = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Test_Ensemble\raw_test\0"
raw_image_dir1 = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Test_Ensemble\raw_test\1"

# paths for the open building images
OB_image_dir0 = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Test_Ensemble\ob_test\0"
OB_image_dir1 = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Test_Ensemble\ob_test\1"

# Paths for the models
models = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Test_Ensemble\models\model_{}.h5"


def predict_function(test_ds,MODEL_NAME):
     """ This function loads model (based on file name sequence) evaluate model against the data set given in test_ds
     Keyword arguments:
     test_ds -- tensor dataset holding the images
     model -- file sequence for the model to be loaded
     """
     df_res = pd.DataFrame()
     final_model = tf.keras.models.load_model(models.format(MODEL_NAME))
     # Since models are using softmax with three output parameters, the predict function
     # will always take the second prediction of the softmaxprint(final_model.summary)

     res = final_model.predict(test_ds)
     df_res['Class_0'] = res[:, 0]
     df_res['Class_1'] = res[:, 1]
     # df['Class_2'] = res[:, 2]

     df_res.loc[(df_res.Class_0 > df_res.Class_1), "results"] = 0
     df_res.loc[(df_res.Class_1 >= df_res.Class_0), "results"] = 1

     mapping = {df_res.columns[0]: '{}_Class_0'.format(MODEL_NAME), df_res.columns[1]: '{}_Class_1'.format(MODEL_NAME),
                df_res.columns[2]: '{}_Results'.format(MODEL_NAME)}
     df_res = df_res.rename(columns=mapping)


     #df_res.rename(columns={'Class_0': '{}_Class_0'.format(MODEL_NAME), 'Class_1': '{}_Class_1'.format(MODEL_NAME)})

     return df_res

def prepare_dataframe(label_0_dir_raw,label_1_dir_raw, label_0_dir_ob, label_1_dir_ob):
  """ This function prepares a new dataframe to hold information about the dataset
  The dataframe will contain the image name to reference for further analyis if needed
  Keyword arguments:
  label_0_dir -- directory for built-up photos
  label_1_dir -- directory for deprived photos
  """
  files1 = []
  files2 = []
  files3 = []
  files4 = []


  # This function will loop each folder separately and will populate the dataframe
  # with the necessary information and then merge dataframe created out of each folder

  # This will repeated 4 times:
  # 1- raw images for label 0
  # 2- raw images for label 1
  # 3- Open Building images for label 0
  # 4- Open Building images for label 1

  for file in sorted(os.listdir(label_0_dir_raw)):
    files1.append((file, 0))

  for file in sorted(os.listdir(label_1_dir_raw)):
    files2.append((file, 1))

  for file in sorted(os.listdir(label_0_dir_ob)):
    files3.append(file)

  for file in sorted(os.listdir(label_1_dir_ob)):
    files4.append(file)

  # Created 4 dataframes from the files list
  # No need to specify the class for open building images because we can use the one from raw image

  df1 = pd.DataFrame(files1, columns=['raw_image_name', 'Class'])
  df2 = pd.DataFrame(files2, columns=['raw_image_name', 'Class'])
  df3 = pd.DataFrame(files3, columns=['ob_image_name'])
  df4 = pd.DataFrame(files4, columns=['ob_image_name'])

  # Concat Open Building image column to the raw image column
  df5 = pd.concat([df1, df3], 1)
  df6 = pd.concat([df2, df4], 1)

  # Concat rows of both label 0 and label 1
  df = pd.concat([df5, df6], ignore_index=True)#,df3])
  return df

def load_data(label_dir,process_mode):
  """ This function utilizes the built-in function image_dataset_from_directory() to load
  the images. It will load different imagew Based on the model (Satellite or Open Building)
  Keyword arguments:
  label_dir -- parent directory for both label data
  process_mode -- to identify size if needed
  """
  IMAGE_SIZE =0
  if process_mode =="RAW1":
      IMAGE_SIZE = 10
  elif process_mode =="RAW2":
      IMAGE_SIZE = 32
  elif process_mode =="OB1":
      IMAGE_SIZE = 100
  elif process_mode == "OB2":
      IMAGE_SIZE = 32

  test_ds = tf.keras.utils.image_dataset_from_directory(
    label_dir,
    label_mode="categorical",
    shuffle=False,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=10)
  return test_ds

def load_and_predict_models(df,raw_dataset,raw_dataset_pre,OB_dataset,OB_dataset_pre):
  """ This function is custom built to call the predict_func for each of the selected models
  The function returns the same input dataframe with individual model prediction added
  each model prediction is in a seperate column with the model name
  Keyword arguments:
  df -- input dataframe that comes pre-populated with image name, size and shape
  """
  # Best Models:
  # 1. CNN on Open Building
  # 2. Pre-trained VGG16 CNN on Open Building
  # 3. CNN on raw images
  # 4. Pre-trained VGG16 CNN on raw images
  # 5. MLP on raw images
  print(">>>> Model # 1...")
  df1 = predict_function(OB_dataset, 'CNN_OB_01_1')
  print(">>>> Model # 2...")
  df2 = predict_function(OB_dataset_pre, 'CNN_OB_PRE_01')
  print(">>>> Model # 3...")
  df3 = predict_function(raw_dataset, 'CNN_RAW_12')
  print(">>>> Model # 4...")
  df4 = predict_function(raw_dataset_pre, 'CNN_RAW_PRE_09')
  print(">>>> Model # 5...")
  df5 = predict_function(raw_dataset, 'MLP_RAW_02')
  print(">>>> Model # 6...")
  df6 = predict_function(OB_dataset, 'CNN_OB_PRE_02')
  print(">>>> Model # 7...")
  df7 = predict_function(raw_dataset, 'CNN_RAW_14')
  #df['Model-11'] = predict_func(test_ds_02, '11')

  df_all = pd.concat([df, df1, df2, df3, df4, df5, df6, df7], 1)

  return df_all

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
    ax.set_title(' Confusion Matrix for hard voted models')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')
    ax.xaxis.set_ticklabels(['Built-up', 'Deprived'])
    ax.yaxis.set_ticklabels(['Built-up', 'Deprived'])
    plt.savefig('C_M_model_{}_rg.png'.format("Ensemble"))
    plt.show()
    return


## MAIN
# Load test data
# This data will be used against the individual models predictions

# First prepare four datasets for test (Open building 32x32, Open Building 100x100, raw image for 10x10 model and raw images for 32x32 pretrained model)

print(">>>> loading data...")
ob_dataset_100 = load_data(OB_image_test_directory, "OB1")
ob_dataset_32 = load_data(OB_image_test_directory, "OB2")
raw_dataset_10 = load_data(raw_image_test_directory,"RAW1")
raw_dataset_32 = load_data(raw_image_test_directory, "RAW2")
print("*********** done loading data")


# Prepre the dataframe that will aggregate all model predictions
print(">>>> preparing dataframe ...")
df = prepare_dataframe(raw_image_dir0, raw_image_dir1,OB_image_dir0, OB_image_dir1)
print("*********** done preparing data frame")

# Now we have the test data ready and the dataframe that will hold different model prediction
df_all = load_and_predict_models(df, raw_dataset_10, raw_dataset_32, ob_dataset_100, ob_dataset_32)
print("*********** done predicting all models")

# Create a new column to hold the sum of all models
df_all['SUM'] = df_all['CNN_OB_01_1_Results'] + df_all['CNN_OB_PRE_01_Results'] + df_all['CNN_RAW_12_Results'] \
                + df_all['CNN_RAW_PRE_09_Results'] + df_all['MLP_RAW_02_Results'] + df_all['CNN_OB_PRE_02_Results'] \
                + df_all['CNN_RAW_14_Results']

# Create a new column that checks if the SUM of all models >= 4
# which means (4 or more models predicted the image as deprived)
df_all['Final'] = 0
df_all.loc[df_all.SUM >= 4, "Final"] = 1

# Save the results for analysis
df_all.to_excel('results_{}.xlsx'.format('ALL'), index=False)

# Export the two lists that will be used for the confusion matrix
y_true = np.array(df_all['Class'])
y_pred = np.array(df_all['Final'])

# Measure the model
measure_model(y_true, y_pred)