from itertools import product
import rasterio as rio
from rasterio import windows
import os

""" ===================================== Clip into 10 x 10px TIF images ============================================ """

parent_dir = '/home/ubuntu/Autoencoder/Final'

print(os.getcwd())

# Select Google Earth Engine Tif File
tif_file = 'Cloud_free_Lagos_img.tif'
infile = '/home/ubuntu/Autoencoder/Final/TIF_Files/' + tif_file

# Creating folder for clipped images
name = 'Lagos'
directory = f'{name}/TIF/'

folder_path = os.path.join(parent_dir, directory)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Set output path for 10 x 10 clipped images
out_path = f'/home/ubuntu/Autoencoder/Final/{directory}'
output_filename = f'{name}'+'_{}-{}.tif'

# Extract 10 x 10px  images from tif
def get_tiles(ds, width=10, height=10):
    nols, nrows = ds.meta['width'], ds.meta['height']
    offsets = product(range(0, nols, width), range(0, nrows, height))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
    for col_off, row_off in  offsets:
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        transform = windows.transform(window, ds.transform)
        yield window, transform

with rio.open(os.path.join(infile)) as inds:
    tile_width, tile_height = 10, 10

    meta = inds.meta.copy()

    for window, transform in get_tiles(inds):
        print(window)
        meta['transform'] = transform
        meta['width'], meta['height'] = window.width, window.height
        outpath = os.path.join(out_path,output_filename.format(int(window.col_off), int(window.row_off)))
        with rio.open(outpath, 'w', **meta) as outds:
            outds.write(inds.read(window=window))

print("Extraction Completed")






