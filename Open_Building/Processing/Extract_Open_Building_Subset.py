import pandas as pd

all = pd.DataFrame()
df_iterator = pd.read_csv(r'C:\Users\minaf\Downloads\103_buildings.csv', chunksize=500000)
interation = 0

#df_lagos_only
for i, df_chunk in enumerate(df_iterator):
    #2.85, 6.37
    #3.87, 6.94
    interation = interation + 1
    zone = df_chunk.query('latitude  <6.94 & latitude > 6.37 & longitude >2.85 & longitude < 3.87')
    if zone is not None:
        all = all.append(zone)

print("*******************")
print(interation)
print("*******************")
print(all.info())
print(all.describe())
all = all.drop("full_plus_code",1)
print(all.describe())
all.to_csv("Lagos_OpenBuilding.csv")