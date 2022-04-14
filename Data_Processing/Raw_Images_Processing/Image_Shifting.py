import rasterio
import pandas as pd


# This code loop through the coordinates csv file for the minority class label (deprived) and generate more images
# Images will be generated by shifting the center coordinates 2 pixels in 4 directions
# To generate even more images images were shifted 1 or 3 pixels as well

# Base path setting
BASE_PATH = r"C:\Users\minaf\Documents\GWU\Capstone\Data\Accra"

# Raw image path setting that will be used to clip new images
RAW_FILE_PATH = BASE_PATH + r"\Maps\Cloud_free_Accra_img.tif"

# Coordinate file (mainly the training file) that will be checked for adjacent images
COORDINATE_PATH_TRAIN = BASE_PATH + r"\train_ac42.csv"

# Path where new images will be processed
path = BASE_PATH + r'\Raw_Images'
SHIFTING_PATH   = path + r"\train\tif\1\clipped_s{}{}{}.tif"

IMAGE_SIZE = 10

# read the coordinates csv file

res = pd.read_csv(COORDINATE_PATH_TRAIN)
res = res[res['Label'] == 1]
mapData = rasterio.open(RAW_FILE_PATH)

for i in range(len(res)):
    py, px = mapData.index(res.iat[i, 1], res.iat[i, 2])
    px_1 = px + 1
    py_1 = py + 2
    window = rasterio.windows.Window(px_1 - IMAGE_SIZE // 2, py_1 - IMAGE_SIZE // 2, IMAGE_SIZE, IMAGE_SIZE)
    clip = mapData.read(window=window)
    metaData = mapData.meta
    metaData['width'], metaData['height'] = IMAGE_SIZE, IMAGE_SIZE
    metaData['transform'] = rasterio.windows.transform(window, mapData.transform)
    newImage = rasterio.open(SHIFTING_PATH.format(i,px_1,py_1), 'w', **metaData)
    newImage.write(clip)

    px_1 = px + 1
    py_1 = py + 2
    window = rasterio.windows.Window(px_1 - IMAGE_SIZE // 2, py_1 - IMAGE_SIZE // 2, IMAGE_SIZE, IMAGE_SIZE)
    clip = mapData.read(window=window)
    metaData = mapData.meta
    metaData['width'], metaData['height'] = IMAGE_SIZE, IMAGE_SIZE
    metaData['transform'] = rasterio.windows.transform(window, mapData.transform)
    newImage = rasterio.open(SHIFTING_PATH.format(i,px_1,py_1), 'w', **metaData)
    newImage.write(clip)

    px_1 = px - 2
    py_1 = py + 1
    window = rasterio.windows.Window(px_1 - IMAGE_SIZE // 2, py_1 - IMAGE_SIZE // 2, IMAGE_SIZE, IMAGE_SIZE)
    clip = mapData.read(window=window)
    metaData = mapData.meta
    metaData['width'], metaData['height'] = IMAGE_SIZE, IMAGE_SIZE
    metaData['transform'] = rasterio.windows.transform(window, mapData.transform)
    newImage = rasterio.open(SHIFTING_PATH.format(i,px_1,py_1), 'w', **metaData)
    newImage.write(clip)

    px_1 = px - 2
    py_1 = py - 1
    window = rasterio.windows.Window(px_1 - IMAGE_SIZE // 2, py_1 - IMAGE_SIZE // 2, IMAGE_SIZE, IMAGE_SIZE)
    clip = mapData.read(window=window)
    metaData = mapData.meta
    metaData['width'], metaData['height'] = IMAGE_SIZE, IMAGE_SIZE
    metaData['transform'] = rasterio.windows.transform(window, mapData.transform)
    newImage = rasterio.open(SHIFTING_PATH.format(i,px_1,py_1), 'w', **metaData)
    newImage.write(clip)

print("*************")
