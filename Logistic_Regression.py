from sklearn.linear_model import LogisticRegression

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

#c is Inverse of regularization strength
clf = LogisticRegression(C=1000 ,max_iter=100,multi_class='multinomial',fit_intercept=False)
#max_iter=100
# C = 5        0.85222
# C = 2        0.84797
# C = 100      0.85156
# C = 1000     0.85328
# C = 5000     0.85252
#max_iter=1000
# C = 0.5      0.84636
# C = 5        0.83898
# C = 2        0.84388
# C = 100      0.83449
# C = 1000     0.83479
# C = 5000     0.83489
# C = 10000    0.83782
#max_iter=10000
# C = 10000    0.83297
# Fit the model to the training data
# take c=1000 max_iter=100
clf.fit(X_train_norm, y_train)

# Make predictions on the test set
predict_set = clf.predict(X_val)
acc = accuracy_score(predict_set, y_val)
print("Accuracy: ", acc)


# Test the classifier on the testing data
#predict_set = clf.predict(X_test_norm)
#acc = accuracy_score(predict_set, y_test)
#print("Accuracy: ", acc)
#testing accuracy=0.8375



# Test the classifier on the training data
#predict_set = clf.predict(X_train_norm)
#acc = accuracy_score(predict_set, y_train)
#print("Accuracy: ", acc)
#training accuracy=0.86597