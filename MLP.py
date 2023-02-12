from sklearn.neural_network import MLPClassifier

from utils import mnist_reader
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split
import random
from sklearn.preprocessing import MinMaxScaler



# ========================================================================
# =========================***** MLP  *****===============================
# ========================================================================

# read the data
X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state = random.randint(1,10000000),test_size=0.33,  shuffle=True)

X_train_reshaped = np.reshape(X_train, (X_train.shape[0], 784))
X_val_reshaped = np.reshape(X_val, (X_val.shape[0], 784))
X_test_reshaped = np.reshape(X_test, (X_test.shape[0], 784))

# # Normalize the features
scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train_reshaped)
X_val_norm = scaler.transform(X_val_reshaped)
X_test_norm = scaler.transform(X_test_reshaped)


clf = MLPClassifier(hidden_layer_sizes=300,alpha=0.001, max_iter=300)
clf.fit(X_train_norm, y_train)
#Make predictions on the test set
predict_set = clf.predict(X_val)
acc = accuracy_score(predict_set, y_val)
print("Accuracy: ", acc)
#hidden_layer_sizes=200   alpha=0.0002   max_iter=100   Accuracy=0.8681
#hidden_layer_sizes=300   alpha=0.001    max_iter=300   Accuracy=0.8745
#hidden_layer_sizes=500   alpha=0.01     max_iter=500   Accuracy=0.8616
#hidden_layer_sizes=100   alpha=0.005    max_iter=500   Accuracy=0.8395

# Test the classifier on the testing data
predict_set = clf.predict(X_test_norm)
acc = accuracy_score(predict_set, y_test)
print("Accuracy: ", acc)
#testing accuracy=0.8866

# Test the classifier on the training data
predict_set = clf.predict(X_test)
acc = accuracy_score(predict_set, y_test)
print("Accuracy: ", acc)
#training accuracy=0.9986


# prediction_true = [0,0,0,0,0,0,0,0,0,0]
# prediction_count = [0,0,0,0,0,0,0,0,0,0]
#
# for i in range(len(predict_set)):
#     if y_test[i] == 0:
#         prediction_count[0] +=1
#         if y_test[i] == predict_set[i]:
#             prediction_true[0] +=1
#
#     elif y_test[i] == 1:
#         prediction_count[1] += 1
#         if y_test[i] == predict_set[i]:
#             prediction_true[1] += 1
#     elif  y_test[i] == 2:
#         prediction_count[2] += 1
#         if y_test[i] == predict_set[i]:
#             prediction_true[2] += 1
#     elif  y_test[i] == 3:
#         prediction_count[3] += 1
#         if y_test[i] == predict_set[i]:
#             prediction_true[3] += 1
#     elif y_test[i] == 4:
#         prediction_count[4] += 1
#         if y_test[i] == predict_set[i]:
#             prediction_true[4] += 1
#     elif  y_test[i] == 5:
#         prediction_count[5] += 1
#         if y_test[i] == predict_set[i]:
#             prediction_true[5] += 1
#     elif  y_test[i] == 6:
#         prediction_count[6] += 1
#         if y_test[i] == predict_set[i]:
#             prediction_true[6] += 1
#     elif  y_test[i] == 7:
#         prediction_count[7] += 1
#         if y_test[i] == predict_set[i]:
#             prediction_true[7] += 1
#     elif  y_test[i] == 8:
#         prediction_count[8] += 1
#         if y_test[i] == predict_set[i]:
#             prediction_true[8] += 1
#     elif  y_test[i] == 9:
#         prediction_count[9] += 1
#         if y_test[i] == predict_set[i]:
#             prediction_true[9] += 1
# for i in range(10):
#     print(f"Acc for C{i} =% {(prediction_true[i]/prediction_count[i]) * 100}")

#cv2