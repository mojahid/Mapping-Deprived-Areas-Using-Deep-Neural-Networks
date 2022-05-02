import os
from os import walk

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
    for i in range(5-length):
        zeros = zeros + "0"
    return zeros + str(global_index)

ob_directory = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Test_Ensemble\obtest2\play"
raw_directory = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Test_Ensemble\raw_test\play"

ob_file = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Test_Ensemble\obtest2\play\ob_{}.png"
raw_file = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Test_Ensemble\raw_test\play\clipped_{}.tif.png"

ob_list = list()
raw_list = list()


#ob_list = os.listdir(ob_directory)
#raw_list = os.listdir(raw_directory)


#ob_list.sort()

#for i in range(0, len(ob_list)):
#    ob_list[i] = ob_list[i].replace('ob_','').replace('.png','')

#for i in range(0, len(raw_list)):
#    raw_list[i] = raw_list[i].replace('clipped_','').replace('.tif.png','')
#print(ob_list)
#k=0
#for i in range(0, len(raw_list)):
#    if (raw_list[i] not in ob_list):
        #print('found{}',raw_list[i])
   #else:
#        print('not found{}', raw_list[i])
#        k=k+1
#        os.remove(raw_file.format(raw_list[i]))

#print("******************{}",k)

#k=0
#for i in range(0, len(ob_list)):
#    if (ob_list[i] not in raw_list):
#        #print('found{}',raw_list[i])
    #else:
#        print('not found{}', ob_list[i])
#        k=k+1
#        os.remove(ob_file.format(ob_list[i]))
#print("******************{}",k)

#raw_new_name = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Test_Ensemble\raw_test\1\clipped_{}.tif.png"
#for filename in os.listdir(r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Test_Ensemble\raw_test\1"):
#    new_file_name1 = filename.replace('clipped_', '').replace('.tif.png', '')
#    new_file_name2 = get_next_image_name(int(new_file_name1))
#    os.rename(raw_new_name.format(new_file_name1), raw_new_name.format(new_file_name2))


raw_new_name = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Test_Ensemble\ob_test\1\ob_{}.png"
for filename in os.listdir(r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Test_Ensemble\ob_test\1"):
    new_file_name1 = filename.replace('ob_','').replace('.png','')
    new_file_name2 = get_next_image_name(int(new_file_name1))
    os.rename(raw_new_name.format(new_file_name1), raw_new_name.format(new_file_name2))

