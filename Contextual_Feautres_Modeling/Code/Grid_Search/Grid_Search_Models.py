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
import warnings
warnings.filterwarnings("ignore")

# Read in dataframe and remove merged columns
df = pd.read_csv(r'/Contextual_Features_final.csv')
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
X = df.values[:, 1:]
y = df.values[:, 0]

'''
# PCA feature selection
pca = ['Label',
      'gabor_sc3_filter_13',
      'sfs_sc71_max_line_length',
      'hog_sc7_variance',
      'orb_sc51_max',
      'lbpm_sc7_variance',
      'sfs_sc71_min_line_length',
      'ndvi_sc3_variance',
      'hog_sc7_kurtosis',
      'lsr_sc31_line_length',
      'sfs_sc31_w_mean',
      'hog_sc7_kurtosis',
      'orb_sc71_kurtosis',
      'orb_sc31_kurtosis']

df_pca = df[pca]

X = df_pca.values[:, 1:]
y = df_pca.values[:, 0]
'''

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)


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

# Create network
clf = MLPClassifier(max_iter=1000000)

# Run Gridsearch
clf = GridSearchCV(clf, parameter_space, n_jobs=-1, cv=3)

clf.fit(X_train, y_train)
clf_pred = clf.predict(X_test)
print("Results Using MLP Best Params & All Features: \n")
print("Classification Report: ")
print(classification_report(y_test, clf_pred))

sns.heatmap(confusion_matrix(y_test, clf_pred), annot=True, fmt='d')
plt.title('MLP')
plt.show()

# Best parameter set
print('Best parameters found for MLP:\n', clf.best_params_)

# --------------------------------------------------------Gradient Boosting----------------------------------------------------

# Hyper-parameter space
parameter_space = {
    'loss': ['deviance', 'exponential'],
    'criterion': ['friedman_mse', 'squared_error', 'mse', 'mae'],
    'n_estimators': [100, 200, 50],
    'subsample': [1.0, 0.8, 0.6],
    "learning_rate": [0.01, 0.025, 0.05],
    "min_samples_split": np.linspace(0.1, 0.5, 12),
    "min_samples_leaf": np.linspace(0.1, 0.5, 12),
    "max_depth": [3, 5, 8],
    "max_features": ["log2", "sqrt"],
}

parameter_space = {
    'loss': ['deviance', 'exponential'],
    'criterion': ['friedman_mse', 'squared_error', 'mse', 'mae'],
    'n_estimators': [100],
    'subsample': [1.0, 0.6],
    "learning_rate": [0.01, 0.05],
    "min_samples_split": np.linspace(0.1, 0.5, 3),
    "min_samples_leaf": np.linspace(0.1, 0.5, 3),
    "max_depth": [3, 8],
    "max_features": ["log2", "sqrt"],
}

clf = GradientBoostingClassifier()

# Run Gridsearch
clf = GridSearchCV(clf, parameter_space, n_jobs=-1, cv=3)

# Gradient Boosting Predictions
clf.fit(X_train, y_train)
clf_pred = clf.predict(X_test)

# Gradient Boosting Results
print("\n")
print("Results Using Gradient Boosting & All Features: \n")
print("Classification Report: ")
print(classification_report(y_test, clf_pred))

sns.heatmap(confusion_matrix(y_test, clf_pred), annot=True, fmt='d')
plt.title('Gradient Boosting')
plt.show()

print('Best parameters found for Gradient Boosting:\n', clf.best_params_)


# --------------------------------------------------------Logistic Regression----------------------------------------------------

# Hyper-parameter space
parameter_space = {
    'penalty': ['l1', 'l2','elasticnet', 'none'],
    'dual': [True, False],
    'C': [0.001, 0.01, 0.1, 1, 10],
    'class_weight': ['dict', 'balanced', None],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}

clf = LogisticRegression()

# Run Gridsearch
clf = GridSearchCV(clf, parameter_space, n_jobs=-1, cv=3)

# Logistic Regression Predictions
clf.fit(X_train, y_train)
clf_pred = clf.predict(X_test)

# Logistic Regression Results
print("\n")
print("Results Using Logistic Regression & All Features: \n")
print("Classification Report: ")
print(classification_report(y_test, clf_pred))

sns.heatmap(confusion_matrix(y_test, clf_pred), annot=True, fmt='d')
plt.title('Logistic Regression')
plt.show()

print('Best parameters found for Logistic Regression:\n', clf.best_params_)

