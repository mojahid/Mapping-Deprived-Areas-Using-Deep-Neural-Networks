import rasterio
import pandas as pd

# This file uses the coordinates extracted

BASE_PATH = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos"


RAW_FILE_PATH = BASE_PATH + r"\Maps\lagos_raw_image.tif"
COORDINATE_PATH = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Labeled\coordinates.csv"

BUILTUP_PATH    = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Labeled\0\clipped_{}.tif"
DEPRIVED_PATH   = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Labeled\1\clipped_{}.tif"
NONBUILDUP_PATH = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Labeled\2\clipped_{}.tif"


IMAGE_SIZE = 10

# read the coordinates csv file

res = pd.read_csv(COORDINATE_PATH)
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
        newImage =  rasterio.open(BUILTUP_PATH.format(index), 'w', **metaData)
        newImage.write(clip)
    elif(row['Label'] == 1):
        newImage =  rasterio.open(DEPRIVED_PATH.format(index), 'w', **metaData)
        newImage.write(clip)
    elif(row['Label'] == 2):
        newImage =  rasterio.open(NONBUILDUP_PATH.format(index), 'w', **metaData)
        newImage.write(clip)