from utils import mnist_reader
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, shuffle=True)

#print(X_train[700])
#print("=======================================================================================")

# Reshape the images

X_train_reshaped = np.reshape(X_train, (X_train.shape[0], 784))
X_val_reshaped = np.reshape(X_val, (X_val.shape[0], 784))
X_test_reshaped = np.reshape(X_test, (X_test.shape[0], 784))

# Normalize the features
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train_reshaped)
X_val_norm = scaler.transform(X_val_reshaped)
X_test_norm = scaler.transform(X_test_reshaped)

#print(X_train_norm[700])

plot.imshow(X_train[5900].reshape( 28 , 28))
plot.show()

#Base Model
base = KNeighborsClassifier(n_neighbors = 3 , metric = "manhattan")
base.fit(X_train_norm, y_train)

predict_set = base.predict(X_val_norm)
acc = accuracy_score(predict_set, y_val)
print("Accuracy: ", acc)


predict_set = base.predict(X_test_norm)
acc = accuracy_score(predict_set, y_test)
print("Accuracy: ", acc)
