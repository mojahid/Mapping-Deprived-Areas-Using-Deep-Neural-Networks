import os.path
import matplotlib.image as mpimg
import os
import numpy as np
import skimage
from PIL import Image


"""============================= Removing images below 10x10px & blank images from PNG Folder ==============================="""
print("Running....")

# remove images less than 10 x 10 px
name = 'Lagos'
img_dir = f"/home/ubuntu/Autoencoder/Final/{name}/PNG/Train"
for filename in os.listdir(img_dir):
    filepath = os.path.join(img_dir, filename)
    with Image.open(filepath) as im:
        x, y = im.size
    totalsize = x*y
    if totalsize < 100:
        os.remove(filepath)

# delete black/blank images i.e. less than 0.25
def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        img = mpimg.imread(filepath)
        img = skimage.img_as_float(img)
        print(np.mean(img))
        if(np.mean(img) <= 0.25):
            os.remove(filepath)

load_images(img_dir)

print("Completed")