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
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
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

# SVD analysis
H = np.matmul(X.T, X)

u, s, v = np.linalg.svd(H) #, full_matrices=True)
#print('Singular values of original = ', s)
singular = s[:30]
var_list = []
for value in singular:
    var_list =+ singular**2/np.sum(singular**2)
print('------------variance list------------')
print(singular[2])
print(var_list[2])
x_axis = np.arange(30)
plt.plot(x_axis, var_list)
plt.xlabel('Number of Components')
plt.ylabel('Prop. of Variance Explained')
plt.title('SVD: Cumulative Explained Variance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Cross Validating TruncatedSVD paired with Logistic Regression model
def get_models():
    models = dict()
    for i in range(1,140):
        steps = [('svd', TruncatedSVD(n_components=i)), ('m', LogisticRegression())]
        models[str(i)] = Pipeline(steps=steps)
    return models
# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
    scores = evaluate_model(model, X, y)
    results.append(scores)
    names.append(name)
print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))
# plot model performance for comparison
plt.boxplot(results, labels=names, showmeans=True)
plt.xlabel('Number of Components')
plt.ylabel('Accuracy')
plt.title('Truncated SVD: Number of Component Analysis')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# train test split
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_scaled = sc.fit_transform(X)
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

# Logistic Regression model
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train, y_train)

# --------------Logistic Regression Predictions-----------------
lr_pred = lr.predict(X_test)
lr_score = lr.predict_proba(X_test)

# Logistic Regression Results
print("\n")
print("Results Using Logistic Regression & All Features: \n")
print("Classification Report: ")
print(classification_report(y_test, lr_pred))



# ---------------Gradient Boosting model-------------------
gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05)
gb_clf.fit(X_train, y_train)

# Gradient Boosting Predictions
gb_pred = gb_clf.predict(X_test)
gb_score = gb_clf.predict_proba(X_test)

# Gradient Boosting Results
print("\n")
print("Results Using Gradient Boosting & All Features: \n")
print("Classification Report: ")
print(classification_report(y_test,gb_pred))

