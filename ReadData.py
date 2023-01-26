import random
from utils import mnist_reader
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

# read the data
X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

# part 1. Validation Set

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state = random.randint(1,10000000),test_size=0.33,  shuffle=True)

#print(X_train[700])
#print("=======================================================================================")
# 2. Features extraction  70% ready
# Reshape the imagesAccuracy:  0.8605333333333334


X_train_reshaped = np.reshape(X_train, (X_train.shape[0], 784))
X_val_reshaped = np.reshape(X_val, (X_val.shape[0], 784))
X_test_reshaped = np.reshape(X_test, (X_test.shape[0], 784))

# Normalize the features
scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train_reshaped)
X_val_norm = scaler.transform(X_val_reshaped)
X_test_norm = scaler.transform(X_test_reshaped)

print(X_train_norm[700])

plot.imshow(X_train[5900].reshape( 28 , 28))
plot.show()

# ========================================================================
# =========================***** KNN  *****===============================
# ========================================================================

# 3. Baseline Model i choise k = 4   ,, 90% ready
# Accuracy:  0.8605333333333334    K = 5 manhattan
# Accuracy:  0.8617333333333334    K = 4 manhattan
# Accuracy:  0.8606666666666667    K = 3 manhattan
# Accuracy:  0.8489333333333333    K = 2 manhattan
# Accuracy:  0.8488888888888889    K = 1 manhattan


#            manhattan              euclidean      sqeuclidean     squareform   braycurtis
#
# k = 1     0.8523232                0.8452020
#
# k = 3     0.855303                 0.846868686                                  0.844393939


#Base Model and part 4.  Improvements  still working here

# base = KNeighborsClassifier(n_neighbors = 3 , metric = "braycurtis")
# base.fit(X_train_norm, y_train)
#
# predict_set = base.predict(X_val_norm)
# acc = accuracy_score(predict_set, y_val)
# print("Accuracy: ", acc)

#
# predict_set = base.predict(X_test_norm)
# acc = accuracy_score(predict_set, y_test)
# print("Accuracy: ", acc)



# ========================================================================
# =========================***** SVM  *****===============================
# ========================================================================


# The 'C' parameter is the Penalty parameter of the error term. It controls the trade off between smooth decision
# boundary and classifying the training points correctly. Large C give low bias and high variance
# , small C give high bias and low variance.

# Create an SVM classifier with a linear kernel
clf = SVC(kernel='sigmoid', C=2)

#          linear      poly      rbf        sigmoid
# c = 1   0.85166
# c = 2   0.84974747
# c = 3
#

# Train the classifier on the training data
clf.fit(X_train_norm, y_train)

# Test the classifier on the testing data
# accuracy = clf.score(X_test, y_test)
# print("Accuracy:", accuracy)


predict_set = clf.predict(X_val_norm)
acc = accuracy_score(predict_set, y_val)
print("Accuracy: ", acc)