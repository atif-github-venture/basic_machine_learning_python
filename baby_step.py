# Source: https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

# check versions

# Python version
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))


# Load The Data

# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load Dataset
# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

# Dimensions of Dataset
# shape
print(dataset.shape)

# Peek at the Data
# head
print(dataset.head(20))

# Statistical Summary
# descriptions
print(dataset.describe())

# Class Distribution
print(dataset.groupby('class').size())


# Data Visualization
# box and whisker plots
# line towards outside is whisker showing extreme values, and box displays 1st, 2nd, 3rd quartile. 2nd quartile
# is computed as mid number on sorted value and 1st and 2nd quartile is computed on either side of 2nd quartile
# for their mid values on leftover subset
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# plt.show()

# histograms
dataset.hist()
# plt.show()

# Multivariate Plots
# This can be helpful to spot structured relationships between input variables.
# scatter plot matrix
scatter_matrix(dataset)
# plt.show()


# Evaluate Some Algorithms
# Create a Validation Dataset
# Split the loaded dataset into two, 80% of which we will use to train our models and 20% that we will hold back as a validation dataset.
# more info on slice array and numpy -> https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/

# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test Harness
# 10-fold cross validation to estimate accuracy
# This will split our dataset into 10 parts, train on 9 and test on 1 and repeat for all combinations of train-test splits.
# The specific random seed does not matter, learn more about pseudorandom number generators here:- >
# https://machinelearningmastery.com/introduction-to-random-number-generators-for-machine-learning/

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'
# We are using the metric of ‘accuracy‘ to evaluate models. This is a ratio of the number of correctly predicted instances in divided
# by the total number of instances in the dataset multiplied by 100 to give a percentage (e.g. 95% accurate). We will be using the
# scoring variable when we run build and evaluate each model next.


# Build Models
# This is a good mixture of simple linear (LR and LDA), nonlinear (KNN, CART, NB and SVM) algorithms.
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# Select Best Model
# In this case, we can see that it looks like Support Vector Machines (SVM) has the largest estimated accuracy score.
# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
# plt.show()

# Make Predictions
# The KNN algorithm is very simple and was an accurate model based on our tests. Now we want to get an idea of the accuracy of the model on our validation set.

# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# https://machinelearningmastery.com/make-predictions-scikit-learn/







print('That will be the end!')