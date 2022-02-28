from sklearn.decomposition import TruncatedSVD
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)


    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Read in dataframe and remove merged columns
df = pd.read_csv(r'C:\Users\brear\OneDrive\Desktop\Grad School\Data-Science-Capstone\Contextual_Features_final.csv')
df = df.drop(columns= ['long_x','lat_x','Label_x','long_y','lat_y','Label_y'])
cols_to_move = ['lat','long','Label','Point']
df = df[ cols_to_move + [ col for col in df.columns if col not in cols_to_move ] ]

# Move Target to first column
target = 'Label'
first_col = df.pop(target)
df.insert(0, target,  first_col)
print(df.head())
print(df['Label'].value_counts())

# define target and independent features
X = df.values[:, 1:]
y = df.values[:, 0]

# train test split
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_scaled = sc.fit_transform(X)
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

# Hyper-parameter space
parameter_space = {
    'hidden_layer_sizes': [(60,100,60), (100,100,100), (50,100,50)],
    'activation': ['identity', 'relu', 'logistic', 'tanh'],
    'solver': ['sgd', 'adam', 'lbfgs'],
    'alpha': [0.0001, 0.00001, 0.000001],
    'learning_rate': ['constant','adaptive', 'invscaling'],
}

# Create network
mlp = MLPClassifier(max_iter=1000000)

# Run Gridsearch
clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)

# Test network
print("============PREDICT TEST SPLIT WITH MLP CLASSIFIER=====================")
mlp.fit(X_train, y_train)
x_predictions = mlp.predict(X_test)
print(classification_report(y_test, x_predictions))

plot_confusion_matrix(mlp, X_test, y_test)
plt.show()

print("============PREDICT TEST SPLIT WITH Best_Params MLP CLASSIFIER=====================")
clf.fit(X_train, y_train)
x_predictions = clf.predict(X_test)
print(classification_report(y_test, x_predictions))

plot_confusion_matrix(clf, X_test, y_test)
plt.show()

# Best parameter set
print('Best parameters found:\n', clf.best_params_)
