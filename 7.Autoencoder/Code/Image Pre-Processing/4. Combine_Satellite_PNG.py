# importing required packages
from pathlib import Path
import shutil
import os

# Combine all three cities PNG images into one folder

parent_dir = '/home/ubuntu/Autoencoder/Final'
directory = 'Satellite_Images/Train/'

# Create directory if doesn't exist
folder_path = os.path.join(parent_dir, directory)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Defining source and destination paths
name = 'Lagos'
src = f'/home/ubuntu/Autoencoder/{name}/PNG/Train'
trg = folder_path

for src_file in Path(src).glob('*.*'):
    shutil.copy(src_file, trg)

print("Transfer Complete")

