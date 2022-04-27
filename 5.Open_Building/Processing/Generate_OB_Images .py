import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import numpy as np
import pandas as pd

def generate_OB_Image(x,y,polygons):
    """
    generate an image with building polygons
    Inputs:
        input_file (str) : the name of input tiff file
    return:
        image(np.array) : image
    """
    image = Image.new("RGB", (100, 100))
    draw = ImageDraw.Draw(image)
    for index, row in polygons.iterrows():
        polygon = row["geometry"]
        polygon = polygon.replace('MULTIPOLYGON', "").replace('POLYGON', "").replace("(", "").replace(")", "")
        polygon = polygon.replace(',', "").split(" ")
        polygon = np.array(polygon)
        polygon = polygon.astype(float)
        #print(polygon)
        points = ()
        for j in range(0,len(polygon),2):
            element = (polygon[j]-x)/0.0000083333 , (y-polygon[j+1])/0.0000083333
            points = points + (element,)

            #print(polygon)
        #print(points)
        draw.polygon((points), fill=255)
    return image


#polygons = pd.read_csv(r"C:\Users\minaf\Documents\GitHub\Data-Science-Capstone\Raw_Images_Modeling\Code\CNN\Lagos_OpenBuilding_2.csv")
#image = generate_OB_Image(3.389583336,	6.492916840,polygons)
#image.show()

