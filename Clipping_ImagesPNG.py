import os
import pandas as pd
import numpy as np
import rasterio
from osgeo import gdal
from PIL import Image

COORDINATE_PATH = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Labeled\coordinates.csv"

BUILTUP_PATH    = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\sample"
DEPRIVED_PATH   = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Labeled\1"
NONBUILDUP_PATH = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Labeled\2"

BUILTUP_PATH_PNG    = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Labeled_png\0\clipped_{}.jpeg"
DEPRIVED_PATH_PNG   = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Labeled_png\1\clipped_{}.jpeg"
NONBUILDUP_PATH_PNG = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Labeled_png\2\clipped_{}.jpeg"


IMAGE_SIZE = 10

# read the coordinates csv file

#res = pd.read_csv(COORDINATE_PATH)


# iterate over files in
i = 0
#for filename in os.listdir(BUILTUP_PATH):
#    infile = rasterio.open(BUILTUP_PATH  + '\\' +  filename)
#    print(infile.meta)
#    profile = infile.profile
#    profile['driver'] = 'BMP'
#    #png_filename = filename.with_suffix('.png')
#   raster = infile.read()
#    dst = rasterio.open(BUILTUP_PATH_PNG.format(i), 'w', **profile)
#    print(BUILTUP_PATH_PNG.format(i))
#    dst.write(raster)
#    i = i+1


#for filename in os.listdir(BUILTUP_PATH):
#    ds = gdal.Open(BUILTUP_PATH  + '\\' +  filename)
    #driver.CreateCopy(BUILTUP_PATH_PNG.format(i), ds, 0)
#    gdal.Translate(BUILTUP_PATH_PNG.format(i), ds, format = 'JPEG')
#    i = i+1


    #rgb = np.stack((ds.GetRasterBand(b).ReadAsArray() for b in (4,3,2)))
    #print(rgb)
    #im = Image.fromarray(rgb)
    #im.save(BUILTUP_PATH_PNG.format(i))
    #i = i+1

for filename in os.listdir(BUILTUP_PATH):
    im = Image.open(BUILTUP_PATH  + '\\' +  filename)
    ds = im.convert('RGB')
    ds.save(BUILTUP_PATH_PNG.format(i))
    i = i + 1