import random
from utils import mnist_reader
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import csv
import pandas as pd
import scipy.fftpack
import cv2

X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')


def hog_transform(X):
    hog = cv2.HOGDescriptor()
    X_hog = []
    for i in range(X.shape[0]):
        image = X[i].reshape(28, 28)
        features = hog.compute(image)
        X_hog.append(features.flatten())
    return np.array(X_hog)


# use the function to transform the data
X_train = hog_transform(X_train)
X_test = hog_transform(X_test)


base = KNeighborsClassifier(n_neighbors = 3 , metric = "manhattan")
base.fit(X_train, y_train)


predict_set = base.predict(X_test)
acc = accuracy_score(predict_set, y_test)
print("Accuracy: ", acc)
