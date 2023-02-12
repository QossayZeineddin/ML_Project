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
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import normalize
X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

def lbp_transform(X):
    X_lbp = []
    for i in range(X.shape[0]):
        image = X[i].reshape(28, 28)
        lbp = local_binary_pattern(image, 12, 1, method='nri_uniform')
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 784))
        hist = hist.astype("float")
        hist = normalize(hist.reshape(1, -1))
        X_lbp.append(hist.flatten())
    return np.array(X_lbp)
# X_train, y_train =  X_train[:1000], y_train[:1000]
# X_test, y_test = X_test[:200], y_test[:200]
print(X_train[0])
# use the function to transform the data
X_train = lbp_transform(X_train)
X_test = lbp_transform(X_test)


base = KNeighborsClassifier(n_neighbors = 3 , metric = "braycurtis")
base.fit(X_train, y_train)
#
predict_set = base.predict(X_test)
acc = accuracy_score(predict_set, y_test)
print("Accuracy: ", acc)