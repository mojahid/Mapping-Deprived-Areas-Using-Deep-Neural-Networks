import numpy as np
from osgeo import gdal
from PIL import Image
import os

def tif2numpyarray(input_file):
    """
    read GeoTiff and convert to numpy.ndarray.
    Inputs:
        input_file (str) : the name of input tiff file
    return:
        image(np.array) : image for each bands
    """
    # read the file and construct a zero numpy
    dataset = gdal.Open(input_file, gdal.GA_ReadOnly)
    image = np.zeros((dataset.RasterYSize, dataset.RasterXSize, dataset.RasterCount),
                     dtype=float)

    # read the 4 bands in each geoTiff
    band1 = dataset.GetRasterBand(1)
    band2 = dataset.GetRasterBand(2)
    band3 = dataset.GetRasterBand(3)
    band4 = dataset.GetRasterBand(4)

    # assign each band to the corresponding location in the array inline with RGB requirements
    image[:, :, 0] = band1.ReadAsArray()
    image[:, :, 1] = band2.ReadAsArray()
    image[:, :, 2] = band3.ReadAsArray()
    image[:, :, 3] = band4.ReadAsArray()

    return image

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


def convert_to_PNG(path):
    """
    Construct a new image from the normalized numpy array
    Changes bands order to fit the colors in original image
    This function uses the normalize and tif2numpyarray
    Inputs:
        path : string path to the geoTiff file
    return:
        Image from array
    """

    # Convert the image to numpy array using tif2numpyarray function
    im  = tif2numpyarray(path)
    #print(im)

    # Normalize each band separately
    a_image2 = np.empty_like(im)
    a_image2[:,:,0] = normalize(im[:,:,0],1)
    a_image2[:,:,1] = normalize(im[:,:,1],2)
    a_image2[:,:,2] = normalize(im[:,:,2],3)
    a_image2[:,:,3] = normalize(im[:,:,3],4)
    #print(a_image2)

    # get RGB relevant value by multiplying * 255
    a_image3 = a_image2*255

    # remove any decimal point
    a_image4 = np.around(a_image3, decimals=0)
    #print(a_image4)

    a_image4[:,:,3] = 255
    a_image4[:, :, 3]
    a_image5 = np.copy(a_image4)
    a_image4[:, :, 0] = a_image5[:, :, 0]
    a_image4[:, :, 1] = a_image5[:, :, 1]
    a_image4[:, :, 2] = a_image5[:, :, 2]

    #a_image7 = np.empty([10, 10, 3])
    #for i in range(10):
    #    a_image7[i] = np.delete(a_image4[i],3,1)
    #print(a_image7)

    imr = Image.fromarray(np.uint8(a_image4))

    return imr


# Train or Test
MODE = "Train"

# Raw_Images or Google_Images
IMAGE = "Google_Images42"

if MODE =="Train":
    BUILTUP_PATH    = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\{}\Train\tif\0"
    #BUILTUP_PATH = r"C:\Users\minaf\Documents\GWU\Capstone\Playground\{}"
    #DEPRIVED_PATH   = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\{}\Train\tif\1"
    DEPRIVED_PATH = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Google_Images42\Train\Moreshift"
    NONBUILDUP_PATH = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\{}\Train\tif\2"
    BUILTUP_PATH_PNG = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\{}\Train\png\0\{}.png"
    #BUILTUP_PATH_PNG = r"C:\Users\minaf\Documents\GWU\Capstone\Playground\{}{}.png"
    DEPRIVED_PATH_PNG = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\{}\Train\png\1\{}.png"
    #DEPRIVED_PATH_PNG = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\{}\Extra\{}.png"
    NONBUILDUP_PATH_PNG = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\{}\Train\png\2\{}.png"
else:
    BUILTUP_PATH    = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\{}\Test\tif\0"
    DEPRIVED_PATH   = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\{}\Test\tif\1"
    NONBUILDUP_PATH = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\{}\Test\tif\2"
    BUILTUP_PATH_PNG = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\{}\Test\png\0\{}.png"
    DEPRIVED_PATH_PNG = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\{}\Test\png\1\{}.png"
    NONBUILDUP_PATH_PNG = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\{}\Test\png\2\{}.png"


#i=0
#for filename in os.listdir(BUILTUP_PATH.format(IMAGE)):
#    path = BUILTUP_PATH.format(IMAGE)  + '\\' +  filename
#    print(path)
#    tmp_image = convert_to_PNG(path)
#    tmp_image.save(BUILTUP_PATH_PNG.format(IMAGE, filename))
#    i = i + 1

i=0
for filename in os.listdir(DEPRIVED_PATH.format(IMAGE)):
    path = DEPRIVED_PATH.format(IMAGE)  + '\\' +  filename
    print(path)
    tmp_image = convert_to_PNG(path)
    tmp_image.save(DEPRIVED_PATH_PNG.format(IMAGE, filename))
    i = i + 1

#i=0
#for filename in os.listdir(NONBUILDUP_PATH.format(IMAGE)):
#    path = NONBUILDUP_PATH.format(IMAGE)  + '\\' +  filename
#    print(path)
#    tmp_image = convert_to_PNG(path)
#    tmp_image.save(NONBUILDUP_PATH_PNG.format(IMAGE, filename))
#    i = i + 1
