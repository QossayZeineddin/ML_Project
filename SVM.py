from utils import mnist_reader
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import train_test_split
import random
from sklearn.preprocessing import MinMaxScaler


# read the data
X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state = random.randint(1,10000000),test_size=0.33,  shuffle=True)


X_train_reshaped = np.reshape(X_train, (X_train.shape[0], 784))
X_val_reshaped = np.reshape(X_val, (X_val.shape[0], 784))
X_test_reshaped = np.reshape(X_test, (X_test.shape[0], 784))

# Normalize the features
scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train_reshaped)
X_val_norm = scaler.transform(X_val_reshaped)
X_test_norm = scaler.transform(X_test_reshaped)

# ========================================================================
# =========================***** SVM  *****===============================
# ========================================================================


# The 'C' parameter is the Penalty parameter of the error term. It controls the trade off between smooth decision
# boundary and classifying the training points correctly. Large C give low bias and high variance
# , small C give high bias and low variance.

# Create an SVM classifier with a linear kernel
clf = SVC(kernel='linear', C=1)

#          linear      poly      rbf        sigmoid
# c = 1   0.85166
# c = 2   0.84974747
# c = 3
#

# Train the classifier on the training data

clf.fit(X_train_norm, y_train)

# Test the classifier on the testing data
predict_set = clf.predict(X_val_norm)
acc = accuracy_score(predict_set, y_val)
print("Accuracy: ", acc)

