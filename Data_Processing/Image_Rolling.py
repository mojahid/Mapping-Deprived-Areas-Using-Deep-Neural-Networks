import rasterio
import pandas as pd

RAW_FILE_PATH = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\lagos_raw_image.tif"
COORDINATE_PATH = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Labeled\coordinates.csv"

BUILTUP_PATH    = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Labeled\0\clipped_{}.tif"
DEPRIVED_PATH   = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Labeled\1\clipped_{}.tif"
NONBUILDUP_PATH = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Labeled\2\clipped_{}.tif"
ROLLING_PATH =  r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Rolling\clipped_{}{}{}.tif"



IMAGE_SIZE = 10

# read the coordinates csv file

res = pd.read_csv(COORDINATE_PATH)
res = res[res['Label'] == 1]
mapData = rasterio.open(RAW_FILE_PATH)

l =0




for i in range(len(res)):
    for j in range(i+1, len(res)-i):
        long_dif = round(res.iat[i, 0] - res.iat[j, 0], 6)
        lat_dif = round(res.iat[i, 1] - res.iat[j, 1], 6)
        if (long_dif == 0) and (lat_dif == 0.000833):
            l = l+1
            py, px = mapData.index(res.iat[i, 0], res.iat[i, 1])
            for k in range(1,10):
                py_temp = py + (1*k)+1
                window = rasterio.windows.Window(px - IMAGE_SIZE // 2, py_temp - IMAGE_SIZE // 2, IMAGE_SIZE, IMAGE_SIZE)
                clip = mapData.read(window=window)
                metaData = mapData.meta
                metaData['width'], metaData['height'] = IMAGE_SIZE, IMAGE_SIZE
                metaData['transform'] = rasterio.windows.transform(window, mapData.transform)
                newImage = rasterio.open(ROLLING_PATH.format(i,"v",k), 'w', **metaData)
                newImage.write(clip)
            #print("vertical: " + str(i) + " - " + str(j) + " " + str(long_dif) + "  " + str(lat_dif))

        elif (lat_dif == 0) and (long_dif == -0.000833):
            l = l+1
            py, px = mapData.index(res.iat[i, 0], res.iat[i, 1])
            for k in range(1,10):
                px_temp = px + (1*k)+1
                window = rasterio.windows.Window(px_temp - IMAGE_SIZE // 2, py - IMAGE_SIZE // 2, IMAGE_SIZE, IMAGE_SIZE)
                clip = mapData.read(window=window)
                metaData = mapData.meta
                metaData['width'], metaData['height'] = IMAGE_SIZE, IMAGE_SIZE
                metaData['transform'] = rasterio.windows.transform(window, mapData.transform)
                newImage = rasterio.open(ROLLING_PATH.format(i,"h",k), 'w', **metaData)
                newImage.write(clip)
            #print("Horizontal: " + str(i) + " - " + str(j) + " " +  str(long_dif) + "  " + str(lat_dif))

print("*************")
print(l)

        #if (res.iat[ ]== res.loc[j, 'lat']) and (res.loc[i, 'long'] - res.loc[j, 'long'] == -0.000833333):
           # print("yes")



