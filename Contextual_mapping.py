import pandas as pd

LABLES_COORDINATES = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Labeled\coordinates.csv"
CONTEXTUAL_COORDINATES = r"C:\Users\minaf\Documents\GWU\Capstone\Data\lagos\Labeled\coordinates_fourier.csv"


# read the coordinates csv file

labels_pd = pd.read_csv(LABLES_COORDINATES)
fourier_pd = pd.read_csv(CONTEXTUAL_COORDINATES)

for index1, l_row in labels_pd.iterrows():
    count = 0
    for index, c_row in fourier_pd.iterrows():
        if (c_row['long'] >= l_row['long']) and (c_row['long'] < l_row ['long'] + 0.000833333) and (c_row['lat'] <=l_row['lat']) and (c_row['lat'] > l_row['lat'] - 0.000833333):
            count = count + 1
    print(str(index1) + " - " + str(count))





