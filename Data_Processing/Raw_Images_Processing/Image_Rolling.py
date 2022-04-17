import rasterio
import pandas as pd

# This code will run through the minority class and will perform image rolling and generating more images from adjacent images
# The idea is to parse the coordinates and check if any image coordinates are next to each other and if this is the case, it will
# generate more images by rolling across the two adjacent images

# Base path setting
BASE_PATH = r"C:\Users\minaf\Documents\GWU\Capstone\Data\Accra"

# Raw image path setting that will be used to clip new images
RAW_FILE_PATH = BASE_PATH + r"\Maps\Cloud_free_Accra_img.tif"

# Coordinate file (mainly the training file) that will be checked for adjacent images
COORDINATE_PATH_TRAIN = BASE_PATH + r"\train_ac42.csv"

# Path where new images will be processed
path = BASE_PATH + r'\Raw_Images'
DEPRIVED_PATH   = path + r"\train\tif\0\clipped_n{}{}{}.tif"

IMAGE_SIZE = 10

# read the coordinates csv file

res = pd.read_csv(COORDINATE_PATH_TRAIN)
res = res[res['Label'] == 0]
mapData = rasterio.open(RAW_FILE_PATH)

l =0

for i in range(len(res)):
    for j in range(i+1, len(res)-i):
        long_dif = round(res.iat[i, 1] - res.iat[j, 1], 6)
        lat_dif = round(res.iat[i, 2] - res.iat[j, 2], 6)
        if (long_dif == 0) and (lat_dif == 0.000833):
            l = l+1
            py, px = mapData.index(res.iat[i, 1], res.iat[i, 2])
            for k in range(1,10):
                py_temp = py + (1*k)+1
                window = rasterio.windows.Window(px - IMAGE_SIZE // 2, py_temp - IMAGE_SIZE // 2, IMAGE_SIZE, IMAGE_SIZE)
                clip = mapData.read(window=window)
                metaData = mapData.meta
                metaData['width'], metaData['height'] = IMAGE_SIZE, IMAGE_SIZE
                metaData['transform'] = rasterio.windows.transform(window, mapData.transform)
                newImage = rasterio.open(DEPRIVED_PATH.format(i,"v",k), 'w', **metaData)
                newImage.write(clip)
            #print("vertical: " + str(i) + " - " + str(j) + " " + str(long_dif) + "  " + str(lat_dif))

        elif (lat_dif == 0) and (long_dif == -0.000833):
            l = l+1
            py, px = mapData.index(res.iat[i, 1], res.iat[i, 2])
            for k in range(1,10):
                px_temp = px + (1*k)+1
                window = rasterio.windows.Window(px_temp - IMAGE_SIZE // 2, py - IMAGE_SIZE // 2, IMAGE_SIZE, IMAGE_SIZE)
                clip = mapData.read(window=window)
                metaData = mapData.meta
                metaData['width'], metaData['height'] = IMAGE_SIZE, IMAGE_SIZE
                metaData['transform'] = rasterio.windows.transform(window, mapData.transform)
                newImage = rasterio.open(DEPRIVED_PATH.format(i,"h",k), 'w', **metaData)
                newImage.write(clip)
            #print("Horizontal: " + str(i) + " - " + str(j) + " " +  str(long_dif) + "  " + str(lat_dif))

print("*************")
print(l)




