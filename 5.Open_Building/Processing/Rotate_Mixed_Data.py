from PIL import Image
import os

def get_next_image_name(global_index):
    """
    provide next name for the image that keep files sorted properly and avoid Lexicographic orders
    Inputs:
        int: global index in the file
    return:
        string new file name
    """
    #Get the number of digits the number is and add preceeding zeros
    length = len(str(global_index))
    zeros = ""
    for i in range(7-length):
        zeros = zeros + "0"
    return zeros + str(global_index)

BASE_PATH = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Open_building\mixed\raw_images\train\{}.png"
# Mode is TEST or TRAIN which will either convert tiff images from the train or the test folders

for i in range (8377,16077,4):
    im_path = BASE_PATH.format(get_next_image_name(i+1))
    im = Image.open(im_path)
    im = im.rotate(90, expand=True)
    im.save(BASE_PATH.format(get_next_image_name(i+1)))

    im = Image.open(BASE_PATH.format(get_next_image_name(i+2)))
    im = im.rotate(270, expand=True)
    im.save(BASE_PATH.format(get_next_image_name(i+2)))

    im = Image.open(BASE_PATH.format(get_next_image_name(i+2)))
    im = im.rotate(180, expand=True)
    im.save(BASE_PATH.format(get_next_image_name(i+2)))


