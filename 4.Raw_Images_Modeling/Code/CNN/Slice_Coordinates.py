import rasterio
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
#import ConvertToPNG

def normalize(array,band):
    """
    Normalizes numpy arrays into scale 0.0 - 1.0 to be used in RGB mapping
    Inputs:
        array: array to be normalized between 0-1
    return:
        normalized array
    """
    #get min and max to normalize
    min = [352.5, 422, 504, 228]
    max = [5246, 4056, 3918, 3577]
    array_min, array_max = array.min(), array.max()
    return ((array - min[band-1])/(max[band-1] - min[band-1]))

def load_data(data_dir):
  """ This function utilizes the built-in function image_dataset_from_directory() to load
  the images.
  """
  test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        label_mode="categorical",
        shuffle=False,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=10)

  return test_ds

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

    #df['Class_0'] = res[:, 0]
    df['Class_1'] = res[:, 1]
    #df['Class_2'] = res[:, 2]

    df.loc[(df.Class_1 < 0.5), "results"] = 0
    df.loc[(df.Class_1 > 0.5) & (df.Class_1 < 0.75), "results"] = 1
    df.loc[(df.Class_1 >= 0.75), "results"] = 2

    return df

def slice_image():
    for i in range(110):
        x = X_CORNER + i * STEP
        print("..... Long loop : {}".format(i))
        for j in range(110):
            y = Y_CORNER - j * STEP

            py, px = mapData.index(x, y)

            window = rasterio.windows.Window(px, py, IMAGE_SIZE, IMAGE_SIZE)
            clip = mapData.read(window=window)

            # copy same meta data
            metaData = mapData.meta
            metaData['width'], metaData['height'] = IMAGE_SIZE, IMAGE_SIZE
            metaData['transform'] = rasterio.windows.transform(window, mapData.transform)
            image = np.zeros((10, 10, 4), dtype=float)

            # read the 4 bands in each geoTiff
            band1 = clip[0]
            band2 = clip[1]
            band3 = clip[2]
            band4 = clip[3]

            # assign each band to the corresponding location in the array inline with RGB requirements
            image[:, :, 0] = band1
            image[:, :, 1] = band2
            image[:, :, 2] = band3
            image[:, :, 3] = band4

            a_image2 = np.empty_like(image)
            a_image2[:, :, 0] = normalize(image[:, :, 0], 1)
            a_image2[:, :, 1] = normalize(image[:, :, 1], 2)
            a_image2[:, :, 2] = normalize(image[:, :, 2], 3)
            a_image2[:, :, 3] = normalize(image[:, :, 3], 4)

            # get RGB relevant value by multiplying * 255
            a_image3 = a_image2 * 255

            # remove any decimal point
            a_image4 = np.around(a_image3, decimals=0)
            # print(a_image4)

            a_image4[:, :, 3] = 255
            a_image4[:, :, 3]
            a_image5 = np.copy(a_image4)
            a_image4[:, :, 0] = a_image5[:, :, 0]
            a_image4[:, :, 1] = a_image5[:, :, 1]
            a_image4[:, :, 2] = a_image5[:, :, 2]

            imr = Image.fromarray(np.uint8(a_image4))
            imr.save(SAVE_PATH.format(BATCH, i, j))
            image_name = SAVE_PATH.format(BATCH, i, j)
            records.append((image_name, x, y))
    df1 = pd.DataFrame(records, columns=['filename', 'Long', 'Lat'])
    df2 = df1.sort_values(by=['filename'])
    # df2 = df1.reindex(sorted(df1.columns), axis=1)
    df2.to_csv(r"C:\Users\minaf\Desktop\Sample\All_Labeled_Batch_{}.csv".format(BATCH))
    return df2

DO_SLICE = False

IMAGE_PATH = r"C:\Users\minaf\Desktop\Sample\test_map.tif"
IMAGE_SIZE = 10

BATCH = 1
X_CORNER = 3.2993274
Y_CORNER = 6.5889620
STEP = 0.000899

SAVE_PATH = r"C:\Users\minaf\Desktop\Sample\Generated\label\Generated_{}_{}_{}.png"
IMAGES_PATH = r"C:\Users\minaf\Desktop\Sample\Generated"

mapData = rasterio.open(IMAGE_PATH)
records = []

MODEL = 'bn12'
models = r"C:\Users\minaf\Data-Science-Capstone\model_{}.h5"

if DO_SLICE:
    print("************************** starting the image slicing *******************")
    df2 = slice_image()
    print("************************** completed slicing *******************")
else:
    df2 = pd.read_csv(r"C:\Users\minaf\Desktop\Sample\All_Labeled_Batch_{}.csv".format(BATCH))
    print("************************** Loaded the pre-generated batch *******************")

print("************************** starting model inference *******************")

test_ds = load_data(IMAGES_PATH)

print("************************** Data Loaded now running the model *******************")

df3 = predict_func(test_ds, df2)

df4 = df3.drop('filename',1)
df4 = df4.drop('Class_1',1)
df4.to_csv(r"C:\Users\minaf\Desktop\Sample\Labeled_Batch_{}.csv".format(BATCH), sep=' ', index=False)