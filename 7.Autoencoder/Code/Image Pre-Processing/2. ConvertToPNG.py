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
    #print("******************************")

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
    """
    Select appropriate min and max to normalize
    """
    # Min-Max values from Lagos TIF file
    min = [352.5, 422, 504, 228]
    max = [5246, 4056, 3918, 3577]

    # Min-Max values from Accra TIF file
    # min = [139, 230, 369.5, 201]
    # max = [9918, 10392, 10400, 6500]

    # Min-Max values from Nairobi TIF file
    # min = [96.5, 50.5, 80.5, 30.5]
    # max = [7110, 7366, 8004, 8042]
    #min = [0, 0, 0, 0]
    #max = [0.2856778204, 0.3260027468, 0.3961989582, 0.4650869071]

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

    # Normalize each band separately
    a_image2 = np.empty_like(im)
    a_image2 = im
    a_image2[:,:,0] = normalize(im[:,:,0],1)
    a_image2[:,:,1] = normalize(im[:,:,1],2)
    a_image2[:,:,2] = normalize(im[:,:,2],3)
    a_image2[:,:,3] = normalize(im[:,:,3],4)

    # get RGB relevant value by multiplying * 255
    a_image3 = a_image2*255

    # remove any decimal point
    a_image4 = np.around(a_image3, decimals=0)

    a_image4[:,:,3] = 255
    a_image4[:, :, 3]
    a_image5 = np.copy(a_image4)
    a_image4[:, :, 0] = a_image5[:, :, 0]
    a_image4[:, :, 1] = a_image5[:, :, 1]
    a_image4[:, :, 2] = a_image5[:, :, 2]

    imr = Image.fromarray(np.uint8(a_image4))

    return imr

# Select parent directory
parent_dir = '/home/ubuntu/Autoencoder/Final'

# Create directory for PNG
name = 'Lagos'
directory = f'{name}/PNG/Train'

# Create folder if doesn't exist
folder_path = os.path.join(parent_dir, directory)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Select TIF folder
PATH = f'/home/ubuntu/Autoencoder/Final/{name}/TIF'

# Select PNG folder
PATH_PNG = f'/home/ubuntu/Autoencoder/Final/{directory}/'+f'{name}' + '_clipped_{}_.png'

# Convert TIF to PNG
i=0
for filename in os.listdir(PATH):
    path = PATH + '/' +  filename
    print(path)
    tmp_image = convert_to_PNG(path)
    tmp_image.save(PATH_PNG.format(filename))
    i = i + 1

print("Conversion Completed")