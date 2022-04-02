import rasterio
import pandas as pd

GOOGLE_IMAGE_PATH = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Cloud_free_lagos_img.tif"
COORDINATE_PATH = r"C:\Users\minaf\Documents\GWU\Capstone\train42.csv"

SHIFTING_PATH =  r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Google_Images42\train\Moreshift\Shifted_{}{}{}.tif"



IMAGE_SIZE = 10

# read the coordinates csv file

res = pd.read_csv(COORDINATE_PATH)
res = res[res['Label'] == 1]
mapData = rasterio.open(GOOGLE_IMAGE_PATH)

l =0

for i in range(len(res)):
    py, px = mapData.index(res.iat[i, 1], res.iat[i, 2])
    px_1 = px + 2
    py_1 = py + 1
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

    px_1 = px - 1
    py_1 = py + 2
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
print(l)
