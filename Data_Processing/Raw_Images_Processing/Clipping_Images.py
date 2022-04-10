import rasterio
import pandas as pd
import os

# This code use the extracted coordinates and parse thru csv file anc clip images in the specified folder
BASE_PATH = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos"
RAW_FILE_PATH = BASE_PATH + r"\Maps\Cloud_free_lagos_img.tif"
# Mode is TEST or TRAIN which will either clip images extracted from the train csv file or the test csv file
MODE = "TRAIN"

# If PROCESS_NON_BUILDUP is set to false then only two labels will be created for 0 and 1 (deprived and buildup)
# If PROCESS_NON_BUILDUP is set to true then three labels will be processed
PROCESS_NON_BUILTUP = False

# Check and create folders
path = BASE_PATH + r'\Raw_Images30'
path_train = path + r"\train\tif"
path_test = path + r"\test\tif"
# Check whether the specified path exists or not
isExist = os.path.exists(path)
if not isExist:
    os.makedirs(path_train)
    os.makedirs(path_test)

# Set the coordinate path
COORDINATE_PATH_TRAIN = BASE_PATH + r"\train422.csv"
COORDINATE_PATH_Test  = BASE_PATH + r"\test422.csv"

# Make and set paths for clipped images
os.makedirs(path + r"\{}\tif\0".format(MODE))
os.makedirs(path + r"\{}\tif\1".format(MODE))
if PROCESS_NON_BUILTUP:
    os.makedirs(path + r"path\{}\tif\2".format(MODE))

BUILTUP_PATH    = path + r"\{}\tif\0\clipped_{}.tif"
DEPRIVED_PATH   = path + r"\{}\tif\1\clipped_{}.tif"
NONBUILDUP_PATH = path + r"\{}\tif\2\clipped_{}.tif"

IMAGE_SIZE = 10

# read the coordinates csv file

if MODE =="TEST":
    res = pd.read_csv(COORDINATE_PATH_Test)
elif MODE == "TRAIN":
    res = pd.read_csv(COORDINATE_PATH_TRAIN)

mapData = rasterio.open(RAW_FILE_PATH)

for index, row in res.iterrows():
    # Get pixel coordinates
    py, px = mapData.index(row['long'], row['lat'])
    window = rasterio.windows.Window(px - IMAGE_SIZE // 2, py - IMAGE_SIZE // 2, IMAGE_SIZE, IMAGE_SIZE)

    clip = mapData.read(window=window)

    # copy same meta data
    metaData = mapData.meta
    metaData['width'], metaData['height'] = IMAGE_SIZE, IMAGE_SIZE
    metaData['transform'] = rasterio.windows.transform(window, mapData.transform)

    if(row['Label'] == 0):
        newImage =  rasterio.open(BUILTUP_PATH.format(MODE,index), 'w', **metaData)
        newImage.write(clip)
    elif(row['Label'] == 1):
        newImage =  rasterio.open(DEPRIVED_PATH.format(MODE,index), 'w', **metaData)
        newImage.write(clip)
    elif(row['Label'] == 2 and PROCESS_NON_BUILTUP):
        newImage =  rasterio.open(NONBUILDUP_PATH.format(MODE,index), 'w', **metaData)
        newImage.write(clip)