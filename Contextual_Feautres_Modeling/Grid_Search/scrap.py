import pandas as pd

df = pd.read_csv(r'C:\Users\brear\OneDrive\Desktop\Grad School\Data-Science-Capstone\Contextual_Feautres_Modeling\Grid_Search\Contextual_Features_final.csv')
df = df.drop(columns= ['long_x','lat_x','Label_x','long_y','lat_y','Label_y'])
cols_to_move = ['lat','long','Label','Point']
df = df[ cols_to_move + [ col for col in df.columns if col not in cols_to_move ] ]

# Move Target to first column
target = 'Label'
first_col = df.pop(target)
df.insert(0, target,  first_col)

# set label column equal to only 0 and 1 classes
df = df[df['Label'].isin([0,1])]
print(df['Label'].value_counts())