from PIL import Image
import os

IMAGE_PATH = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Google_Images42\Train\png\1"
SAVE_PATH = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Google_Images42\Train\png\1\R_{}_{}.png"

for filename in os.listdir(IMAGE_PATH):
    image_file = IMAGE_PATH + '\\' + filename
    im = Image.open(image_file)
    im = im.rotate(90, expand=True)
    im.save(SAVE_PATH.format(filename, '90'))
    im = im.rotate(180, expand=True)
    im.save(SAVE_PATH.format(filename, '180'))
    im = im.rotate(270, expand=True)
    im.save(SAVE_PATH.format(filename, '270'))

