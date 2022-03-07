import shutil
import numpy as np
import os


BUILTUP_PATH_PNG    = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Labeled_png\0"
NONBUILDUP_PATH_PNG = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Labeled_png\2"

BUILTUP_PATH_U_Sample = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Under_Sample\0"
NONBUILDUP_PATH_U_Sample = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Under_Sample\2"

buildup_files = os.listdir(BUILTUP_PATH_PNG)
nonbuildup_files = os.listdir(NONBUILDUP_PATH_PNG)

array1 = np.random.randint(15000, size=2000)
array2 = np.random.randint(32000, size=2000)

for i in range(2000):
    src1 = BUILTUP_PATH_PNG + '\\' + buildup_files[array1[i]]
    dst1 = BUILTUP_PATH_U_Sample + '\\' + buildup_files[array1[i]]

    src2 = NONBUILDUP_PATH_PNG + '\\' + nonbuildup_files[array2[i]]
    dst2 = NONBUILDUP_PATH_U_Sample + '\\' + nonbuildup_files[array2[i]]

    shutil.copyfile(src1, dst1)
    shutil.copyfile(src2, dst2)

