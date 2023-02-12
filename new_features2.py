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

X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')


def fft_transform(X):
    X_fft = []
    for i in range(X.shape[0]):
        image = X[i].reshape(28, 28)
        fft = np.abs(scipy.fftpack.fft2(image))
        X_fft.append(fft.flatten())
    return np.array(X_fft)

# X_train, y_train =  X_train[:1000], y_train[:1000]
# X_test, y_test = X_test[:200], y_test[:200]
# use the function to transform the data
X_train = fft_transform(X_train)
X_test = fft_transform(X_test)

#bisemodel
# base = KNeighborsClassifier(n_neighbors = 3 , metric = "manhattan")
# base.fit(X_train, y_train)

#best model
clf = SVC(kernel='rbf', C=13)
clf.fit(X_train, y_train)


predict_set = clf.predict(X_test)
acc = accuracy_score(predict_set, y_test)
print("Accuracy: ", acc)


