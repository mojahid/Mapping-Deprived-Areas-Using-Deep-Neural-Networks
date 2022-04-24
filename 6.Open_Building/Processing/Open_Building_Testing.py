import pandas as pd
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import Generate_OB_Images
import random

# load google open building in a dataframe
all_bldgs = pd.read_csv(r'C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Open_building\Lagos_OpenBuilding.csv')

# Labeled coordinates file
labels = pd.read_csv(r'C:\Users\minaf\Documents\GWU\Capstone\Data\train42.csv')
labels = labels[labels["Label"] != 2]

path = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Open_building\train\{}\ob_aug9_{}.png"

# loop through all coordinates
for index, row in labels.iterrows():
    # Get pixel coordinates and define the 4 boundary coordinates
    py, px = row['long'], row['lat']
    py1 = round(py + 5*0.00008333, 6)
    py2 = round(py - 5*0.00008333, 6)
    px1 = round(px + 5*0.00008333, 6)
    px2 = round(px - 5*0.00008333, 6)

    # Query text to bound longitude and latitude within the boundaries
    query_text = 'latitude  <' +str(px1)+ ' & latitude > '+str(px2)+ ' & longitude < ' + str(py1)+' & longitude > ' +str(py2)

    zone = all_bldgs.query(query_text)
    #zone.to_csv("Lagos_OpenBuilding_{}.csv".format(index))
    image = Generate_OB_Images.generate_OB_Image(py2,px1,zone)

    image = image.convert("L")
    image.save(path.format(labels.at[index,'Label'], index))
    #image.show()
    #print(zone)

