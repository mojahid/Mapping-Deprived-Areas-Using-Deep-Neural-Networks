import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import os
import warnings
warnings.filterwarnings("ignore")

from project_root import get_project_root
root = get_project_root()

# parameters for file storage name
dataset = 'contextual'
model = 'Ensemble'
feature_count = 144
features = 'All_Features'
classes = 'all_classes'

if feature_count == 144:
    feature_count = ''
else:
    feature_count = feature_count


# data
df = pd.read_csv(root / '1.Data' / 'Contextual_data.csv')
df = df.drop(columns=['long', 'lat', 'Point'])
cols_to_move = ['Label']
df = df[cols_to_move + [col for col in df.columns if col not in cols_to_move]]
# df = df[df['Label'].isin([0, 1])]

# Move Target to first column
target = 'Label'
first_col = df.pop(target)
df.insert(0, target, first_col)

# set features and target
X = df.values[:, 1:]
y = df.values[:, 0]

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25,
                                                  random_state=42)  # 0.25 x 0.8 = 0.2

# Feature Scaling
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)
X_val = sc.transform(X_val)



# models for ensembling
clf1 = MLPClassifier(hidden_layer_sizes = (100, 100, 100),
                     activation = 'tanh',
                     solver = 'adam',
                     alpha = 0.001,
                     learning_rate = 'invscaling')
clf2 = MLPClassifier(hidden_layer_sizes = (100, 100, 100),
                     activation = 'tanh',
                     solver = 'adam',
                     alpha = 0.001,
                     learning_rate = 'invscaling')
clf3 = MLPClassifier(hidden_layer_sizes = (100, 100, 100),
                     activation = 'tanh',
                     solver = 'adam',
                     alpha = 0.001,
                     learning_rate = 'invscaling')
clf4 = MLPClassifier(hidden_layer_sizes = (100, 100, 100),
                     activation = 'tanh',
                     solver = 'adam',
                     alpha = 0.001,
                     learning_rate = 'invscaling')
clf5 = MLPClassifier(hidden_layer_sizes = (100, 100, 100),
                     activation = 'tanh',
                     solver = 'adam',
                     alpha = 0.001,
                     learning_rate = 'invscaling')
clf6 = MLPClassifier(hidden_layer_sizes = (100, 100, 100),
                     activation = 'tanh',
                     solver = 'adam',
                     alpha = 0.001,
                     learning_rate = 'invscaling')
clf7 = MLPClassifier(hidden_layer_sizes = (100, 100, 100),
                     activation = 'tanh',
                     solver = 'adam',
                     alpha = 0.001,
                     learning_rate = 'invscaling')
clf8 = MLPClassifier(hidden_layer_sizes = (100, 100, 100),
                     activation = 'tanh',
                     solver = 'adam',
                     alpha = 0.001,
                     learning_rate = 'invscaling')
clf9 = MLPClassifier(hidden_layer_sizes = (100, 100, 100),
                     activation = 'tanh',
                     solver = 'adam',
                     alpha = 0.001,
                     learning_rate = 'invscaling')

# ensemble models
eclf1 = VotingClassifier(estimators=[('mlp1', clf1), ('mlp2', clf2), ('mlp3', clf3), ('mlp4', clf4), ('mlp5', clf5), ('mlp6', clf6), ('mlp7', clf7), ('mlp8', clf8), ('mlp9', clf9)], voting='hard')

# Fit model
eclf1.fit(X_train, y_train)



# Predict on validation set
val_pred = eclf1.predict(X_val)
print(f"Validation Results Using Ensembled Contextual Models: \n")
print("Classification Report: ")
print(classification_report(y_val, val_pred))
cf_matrix = confusion_matrix(y_val, val_pred)
print(cf_matrix)
sns.heatmap(cf_matrix, annot=True, fmt="d")
plt.title(f'Contextual {model} Model - {feature_count}{features}, {classes}')
plt.show()

# f1 scores for comparison table output
f1_micro_class0 = f1_score(y_val, val_pred, average=None)[0]
f1_micro_class1 = f1_score(y_val, val_pred, average=None)[1]
f1_macro = f1_score(y_val, val_pred, average='macro')

# Save model
filename = f'{dataset}_{model}_model_{feature_count}{features}_{classes}.sav'
pickle.dump(eclf1, open(root / '1.Data' / filename, 'wb'))