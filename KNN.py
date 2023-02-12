from utils import mnist_reader
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
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
# =========================***** KNN  *****===============================
# ========================================================================

# 3. Baseline Model i choise k = 4   ,, 90% ready
# Accuracy:  0.8605333333333334    K = 5 manhattan
# Accuracy:  0.8617333333333334    K = 4 manhattan
# Accuracy:  0.8606666666666667    K = 3 manhattan
# Accuracy:  0.8489333333333333    K = 2 manhattan
# Accuracy:  0.8488888888888889    K = 1 manhattan


#            manhattan              euclidean           cosine         sqeuclidean     squareform   braycurtis
#
# k = 1     0.8523232                0.8452020          0.85257
#
# k = 3     0.855303                 0.846868686        0.85333                                     0.844393939


#Base Model and part 4.  Improvements  still working here

base = KNeighborsClassifier(n_neighbors = 3 , metric = "braycurtis")
base.fit(X_train_norm, y_train)
#
predict_set = base.predict(X_val_norm)
acc = accuracy_score(predict_set, y_val)
print("Accuracy: ", acc)

# prediction_true = [0,0,0,0,0,0,0,0,0,0]
# prediction_count = [0,0,0,0,0,0,0,0,0,0]
# for i in range(len(predict_set)):
#     if y_val[i] == 0:
#         prediction_count[0] +=1
#         if y_val[i] == predict_set[i]:
#             prediction_true[0] +=1
#     elif y_val[i] == 1:
#         prediction_count[1] += 1
#         if y_val[i] == predict_set[i]:
#             prediction_true[1] += 1
#     elif  y_val[i] == 2:
#         prediction_count[2] += 1
#         if y_val[i] == predict_set[i]:
#             prediction_true[2] += 1
#     elif  y_val[i] == 3:
#         prediction_count[3] += 1
#         if y_val[i] == predict_set[i]:
#             prediction_true[3] += 1
#     elif y_val[i] == 4:
#         prediction_count[4] += 1
#         if y_val[i] == predict_set[i]:
#             prediction_true[4] += 1
#     elif  y_val[i] == 5:
#         prediction_count[5] += 1
#         if y_val[i] == predict_set[i]:
#             prediction_true[5] += 1
#     elif  y_val[i] == 6:
#         prediction_count[6] += 1
#         if y_val[i] == predict_set[i]:
#             prediction_true[6] += 1
#     elif  y_val[i] == 7:
#         prediction_count[7] += 1
#         if y_val[i] == predict_set[i]:
#             prediction_true[7] += 1
#     elif  y_val[i] == 8:
#         prediction_count[8] += 1
#         if y_val[i] == predict_set[i]:
#             prediction_true[8] += 1
#     elif  y_val[i] == 9:
#         prediction_count[9] += 1
#         if y_val[i] == predict_set[i]:
#             prediction_true[9] += 1
# for i in range(10):
#     print(f"Acc for C{i} = {prediction_true[i]/prediction_count[i]}")
# #
# predict_set = base.predict(X_test_norm)
# acc = accuracy_score(predict_set, y_test)
# print("Accuracy: ", acc)
