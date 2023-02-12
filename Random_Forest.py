from utils import mnist_reader
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
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
# =========================***** Random Forest  *****===============================
# ========================================================================



# Define the Random Forest model
clf = RandomForestClassifier(n_estimators=20)

# n = 100   0.881313131
# n = 10    0.85207


# Fit the model to the training data
clf.fit(X_train_norm, y_train)

# Make predictions on the test set
predict_set = clf.predict(X_test)
acc = accuracy_score(predict_set, y_test)
print("Accuracy: ", acc)

