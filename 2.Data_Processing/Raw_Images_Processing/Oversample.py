from PIL import Image
import os


BASE_PATH = r"C:\Users\minaf\Documents\GWU\Capstone\Data\Accra"
# Mode is TEST or TRAIN which will either convert tiff images from the train or the test folders
MODE = "TRAIN"

# If PROCESS_NON_BUILDUP is set to false then only two labels will be created for 0 and 1 (deprived and buildup)
# If PROCESS_NON_BUILDUP is set to true then three labels will be processed
PROCESS_NON_BUILTUP = False

# Check and create folders
path = BASE_PATH + r'\Raw_Images'

IMAGE_PATH = path + r"\{}\png\1".format(MODE)

SAVE_PATH = IMAGE_PATH + r"\R_{}_{}.png"

for filename in os.listdir(IMAGE_PATH):
    image_file = IMAGE_PATH + '\\' + filename
    im = Image.open(image_file)
    im = im.rotate(90, expand=True)
    im.save(SAVE_PATH.format(filename, '90'))
    im = im.rotate(180, expand=True)
    im.save(SAVE_PATH.format(filename, '180'))
    im = im.rotate(270, expand=True)
    im.save(SAVE_PATH.format(filename, '270'))

