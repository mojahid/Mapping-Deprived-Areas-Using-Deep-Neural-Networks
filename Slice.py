import rasterio
import pandas as pd

RAW_FILE_PATH = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\lag_contextual_features_Spfeas\fourier\fourier_sc31_mean.tif"
##COORDINATE_PATH = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Labeled\coordinates.csv"

BUILTUP_PATH    = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Labeled\0\clipped_{}.tif"
DEPRIVED_PATH   = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Labeled\1\clipped_{}.tif"
NONBUILDUP_PATH = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Labeled\2\TestClip2_{}.tif"


IMAGE_SIZE = 300

# read the coordinates csv file

#res = pd.read_csv(COORDINATE_PATH)
mapData = rasterio.open(RAW_FILE_PATH)

#for index, row in res.iterrows():
    # Get pixel coordinates
py, px = mapData.index(3.401267,6.881582)
window = rasterio.windows.Window(px - IMAGE_SIZE // 2, py - IMAGE_SIZE // 2, IMAGE_SIZE, IMAGE_SIZE)

clip = mapData.read(window=window)

    # copy same meta data
metaData = mapData.meta
metaData['width'], metaData['height'] = IMAGE_SIZE, IMAGE_SIZE
metaData['transform'] = rasterio.windows.transform(window, mapData.transform)


newImage =  rasterio.open(NONBUILDUP_PATH.format(1), 'w', **metaData)
newImage.write(clip)