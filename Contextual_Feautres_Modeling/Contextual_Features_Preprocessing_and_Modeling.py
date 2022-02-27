import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import warnings
warnings.filterwarnings("ignore")

# Read in dataframe and remove merged columns
df = pd.read_csv('Contextual_Features_final.csv')
df = df.drop(columns= ['long_x','lat_x','Label_x','long_y','lat_y','Label_y'])
cols_to_move = ['lat','long','Label','Point']
df = df[ cols_to_move + [ col for col in df.columns if col not in cols_to_move ] ]

# Move Target to first column
target = 'Label'
first_col = df.pop(target)
df.insert(0, target,  first_col)
print(df.head())
print(df['Label'].value_counts())
# correlation heatmap to check for multicollinearity
'''
plt.figure(figsize=(50, 50))
plt.title('Correlation Heat Map', fontsize= 40)
sns.heatmap(df.corr(), annot=True, vmin=-1, vmax=1, cmap='YlOrRd', square=True)
plt.show()
'''
# define target and independent features
X = df.values[:, 1:]
y = df.values[:, 0]

# Singular Value Decomposition
svd = TruncatedSVD(n_components=25, n_iter=7, random_state=42) # define transform
svd.fit(X) # prepare transform on dataset
X_transformed = svd.transform(X) # apply transform to dataset
print('post-SVD shape: ', X_transformed.shape)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_scaled = sc.fit_transform(X)
sc.fit(X_train)
X_train_scaled = sc.transform(X_train)
X_test_scaled = sc.transform(X_test)

# Decision Tree model
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_scaled, y_train)

# --------------Logistic Regression Predictions-----------------
lr_pred = lr.predict(X_test_scaled)
lr_score = lr.predict_proba(X_test_scaled)

# Logistic Regression Results
print("\n")
print("Results Using Logistic Regression & All Features: \n")
print("Classification Report: ")
print(classification_report(y_test, lr_pred))



# ---------------Gradient Boosting model-------------------
gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05)
gb_clf.fit(X_train_scaled, y_train)

# Gradient Boosting Predictions
gb_pred = gb_clf.predict(X_test_scaled)
gb_score = gb_clf.predict_proba(X_test_scaled)

# Gradient Boosting Results
print("\n")
print("Results Using Gradient Boosting & All Features: \n")
print("Classification Report: ")
print(classification_report(y_test,gb_pred))

