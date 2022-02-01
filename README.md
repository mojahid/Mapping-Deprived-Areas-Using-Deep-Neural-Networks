# Data-Science-Capstone
This project aims to use satellite images and other data to predict poverty indecies 

Extract_Coordinates.py:
This python code will process the training tif file and read the meta data then extract labels and its coresponding coordinates.
It will then transform the coordinates (in case it is needed in other data capturing) and save all the coordinates and the labels in a csv file.

Notes:
1- Coordintanes represents the top left pixel of the 100x100 meters label 
2- As per the raw image resolution, the 100x100 meter block is represented by 10x10 pixel in the image

Clipping_Images.py:
This file will iterate through the coordinates csv file obtained from the Extract_Coordinates.py and will extract the corresponding clipped 10 pixel x 10 pixel image from the raw image and save the clipped images in three folders (should be created before running the code)
