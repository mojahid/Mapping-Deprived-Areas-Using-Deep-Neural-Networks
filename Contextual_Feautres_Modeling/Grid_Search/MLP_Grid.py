import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings("ignore")

print('------------------------------All Classes------------------------------')

# Read in dataframe and remove merged columns
df = pd.read_csv(r'C:\Users\brear\OneDrive\Desktop\Grad School\Data-Science-Capstone\Contextual_Feautres_Modeling\Grid_Search\Contextual_Features_final.csv')
df = df.drop(columns= ['long_x','lat_x','Label_x','long_y','lat_y','Label_y'])
cols_to_move = ['lat','long','Label','Point']
df = df[ cols_to_move + [ col for col in df.columns if col not in cols_to_move ] ]

# Move Target to first column
target = 'Label'
first_col = df.pop(target)
df.insert(0, target,  first_col)
#print(df.head())
#print(df['Label'].value_counts())

# define target and independent features

# full dataset
# X = df.values[:, 1:]
# y = df.values[:, 0]


# PCA feature selection
# pca_features = pd.read_csv(r'C:\Users\brear\OneDrive\Documents\GitHub\Data-Science-Capstone\Contextual_Feautres_Modeling\feature_selection\best_pca_features.csv')
'''
pca = ['Label']
for row in range(50):
    pca.append(pca_features.iloc[row,0])

df_pca = df[pca]

X = df_pca.values[:, 1:]
y = df_pca.values[:, 0]
'''

# Logistic feature selection
log_features = pd.read_csv(r'C:\Users\brear\OneDrive\Documents\GitHub\Data-Science-Capstone\Contextual_Feautres_Modeling\feature_selection\logistic_feature_importance.csv')

log = ['Label']
for row in range(50):
    log.append(log_features.iloc[row,0])

df_log= df[log]

X = df_log.values[:, 1:]
y = df_log.values[:, 0]

# Random Forest feature selection
# rf_features = pd.read_csv(r'C:\Users\brear\OneDrive\Documents\GitHub\Data-Science-Capstone\Contextual_Feautres_Modeling\feature_selection\random_forest_values.csv')
'''
rf = ['Label']
for row in range(50):
    rf.append(rf_features.iloc[row,0])

df_rf= df[rf]

X = df_rf.values[:, 1:]
y = df_rf.values[:, 0]
'''

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42) # 0.25 x 0.8 = 0.2

# Feature Scaling
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)
X_val = sc.transform(X_val)


# -------------------------------------------------------------MLP-------------------------------------------------------------
# Hyper-parameter space
parameter_space = {
    'hidden_layer_sizes': [(60,100,60), (100,100,100), (50,100,50)],
    'activation': ['identity', 'relu', 'logistic', 'tanh'],
    'solver': ['sgd', 'adam', 'lbfgs'],
    'alpha': [0.0001, 0.00001, 0.000001],
    'learning_rate': ['constant','adaptive', 'invscaling'],
}

parameter_space = {
    'hidden_layer_sizes': [(60,100,60), (100,100,100), (50,100,50)],
    'activation': ['identity', 'relu', 'logistic', 'tanh'],
    'solver': ['adam'],
    'alpha': [0.0001],
    'learning_rate': ['invscaling'],
}
'''
# Create network
clf = MLPClassifier(max_iter=1000000)

# Run Gridsearch
clf = GridSearchCV(clf, parameter_space, n_jobs=-1, cv=3)

clf.fit(X_train, y_train)

clf_pred = clf.predict(X_test)
print("Test Results Using MLP Best Params & Top 50 Log Features: \n")
print("Classification Report: ")
print(classification_report(y_test, clf_pred))

# Best parameter set
print('Best parameters found for MLP:\n', clf.best_params_)

# Save model
filename = 'MLP_model_Top50_Log_Features_AllClasses.sav'
pickle.dump(clf, open(filename, 'wb'))
'''
# load the model from disk
filename = 'MLP_model_Top50_Log_Features_AllClasses.sav'
loaded_model = pickle.load(open(filename, 'rb'))
val_pred = loaded_model.predict(X_val)
print("Validation Results Using MLP Best Params, Top 50 Log Features, All Classes: \n")
print("Classification Report: ")
print(classification_report(y_val, val_pred))

print('------------------------------Classes 0 & 1 Only------------------------------')


# Read in dataframe and remove merged columns
df = pd.read_csv(r'C:\Users\brear\OneDrive\Desktop\Grad School\Data-Science-Capstone\Contextual_Feautres_Modeling\Grid_Search\Contextual_Features_final.csv')
df = df.drop(columns= ['long_x','lat_x','Label_x','long_y','lat_y','Label_y'])
cols_to_move = ['lat','long','Label','Point']
df = df[ cols_to_move + [ col for col in df.columns if col not in cols_to_move ] ]

# Move Target to first column
target = 'Label'
first_col = df.pop(target)
df.insert(0, target,  first_col)
#print(df.head())
#print(df['Label'].value_counts())

# set label column equal to only 0 and 1 classes
df = df[df['Label'].isin([0,1])]

# define target and independent features

# full dataset
# X = df.values[:, 1:]
# y = df.values[:, 0]


# PCA feature selection
# pca_features = pd.read_csv(r'C:\Users\brear\OneDrive\Documents\GitHub\Data-Science-Capstone\Contextual_Feautres_Modeling\feature_selection\best_pca_features.csv')
'''
pca = ['Label']
for row in range(50):
    pca.append(pca_features.iloc[row,0])

df_pca = df[pca]

X = df_pca.values[:, 1:]
y = df_pca.values[:, 0]
'''

# Logistic feature selection
log_features = pd.read_csv(r'C:\Users\brear\OneDrive\Documents\GitHub\Data-Science-Capstone\Contextual_Feautres_Modeling\feature_selection\logistic_feature_importance.csv')

log = ['Label']
for row in range(50):
    log.append(log_features.iloc[row,0])

df_log= df[log]

X = df_log.values[:, 1:]
y = df_log.values[:, 0]

# Random Forest feature selection
# rf_features = pd.read_csv(r'C:\Users\brear\OneDrive\Documents\GitHub\Data-Science-Capstone\Contextual_Feautres_Modeling\feature_selection\random_forest_values.csv')
'''
rf = ['Label']
for row in range(50):
    rf.append(rf_features.iloc[row,0])

df_rf= df[rf]

X = df_rf.values[:, 1:]
y = df_rf.values[:, 0]
'''


# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42) # 0.25 x 0.8 = 0.2

# Feature Scaling
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)
X_val = sc.transform(X_val)


# -------------------------------------------------------------MLP-------------------------------------------------------------
# Hyper-parameter space
parameter_space = {
    'hidden_layer_sizes': [(60,100,60), (100,100,100), (50,100,50)],
    'activation': ['identity', 'relu', 'logistic', 'tanh'],
    'solver': ['sgd', 'adam', 'lbfgs'],
    'alpha': [0.0001, 0.00001, 0.000001],
    'learning_rate': ['constant','adaptive', 'invscaling'],
}

parameter_space = {
    'hidden_layer_sizes': [(60,100,60), (100,100,100), (50,100,50)],
    'activation': ['identity', 'relu', 'logistic', 'tanh'],
    'solver': ['adam'],
    'alpha': [0.0001],
    'learning_rate': ['invscaling'],
}

'''
# Create network
clf = MLPClassifier(max_iter=1000000)

# Run Gridsearch
clf = GridSearchCV(clf, parameter_space, n_jobs=-1, cv=3)

clf.fit(X_train, y_train)

clf_pred = clf.predict(X_test)
print("Test Results Using MLP Best Params & Top 50 Log Features: \n")
print("Classification Report: ")
print(classification_report(y_test, clf_pred))

# Best parameter set
print('Best parameters found for MLP:\n', clf.best_params_)

# Save model
filename = 'MLP_model_Top50_Log_Features_Classes01.sav'
pickle.dump(clf, open(filename, 'wb'))
'''

# load the model from disk
filename = 'MLP_model_Top50_Log_Features_Classes01.sav'
loaded_model = pickle.load(open(filename, 'rb'))
val_pred = loaded_model.predict(X_val)
print("Validation Results Using MLP Best Params, Top 50 Log Features, Classes 0 & 1: \n")
print("Classification Report: ")
print(classification_report(y_val, val_pred))




