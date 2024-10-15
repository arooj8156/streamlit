# import libraries

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.decomposition import PCA

#  App title and description
st.write(""""
# Explore different ML models and datasets
let's see which one is best .
""")

# sidebar
dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('Iris', 'Breast Cancer', 'Wine')
)

classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('K Nearest Neighbours', 'Support Vector Machine(SVM)', 'Random Forest')
)

# load dataset
def get_dataset(dataset_name):
    data = None
    if dataset_name == 'Iris':
        data = datasets.load_iris()
    elif dataset_name == 'Breast Cancer':
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X, y

# call the function
X, y = get_dataset(dataset_name)

# get shape
st.write('Shape of dataset:', X.shape)
st.write('Number of classes:', len(np.unique(y)))


# add classifier parametes to user input
def add_parameter_ui(classifier_name):
    params = dict()
    if classifier_name == 'K Nearest Neighbours':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    elif classifier_name == 'Support Vector Machine(SVM)':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['max_depth'] = max_depth
        params['n_estimators'] = n_estimators
    return params

# call the function
params = add_parameter_ui(classifier_name)

# get classifier
def get_classifier(classifier_name, params):
    clf = None
    if classifier_name == 'K Nearest Neighbours':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    elif classifier_name == 'Support Vector Machine(SVM)':
        clf = SVC(C=params['C'])
    else:
        clf = RandomForestClassifier(n_estimators=params['n_estimators'],
             max_depth=params['max_depth'])
    return clf

# call the function
clf = get_classifier(classifier_name, params)

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train the classifier
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# calculate accuracy
acc = accuracy_score(y_test, y_pred)
st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', acc)

# plot dataset
pca = PCA(2)

X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, edgecolors='k', cmap='viridis')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()
st.pyplot(fig)
