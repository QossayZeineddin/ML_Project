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



#          linear      poly         rbf        sigmoid
# c = 1   0.85166     0.85959     0.88429      0.42085
# c = 2   0.84975     0.86934     0.89298      0.38227
# c = 3   0.84813     0.87323     0.89702      0.46277
# c = 4   0.84399     0.87434     0.89854      0.38525
# c = 5   0.84288     0.87843     0.90025      0.42535
# c = 6   0.84086     0.87985     0.89868      0.38141
# c = 7   0.84106     0.87894     0.89964      0.43661
# c = 8   0.84030     0.88237     0.89919      0.42883
# c = 9   0.84091     0.88066     0.90030      0.44166
# c = 10  0.84086     0.87702     0.90166      0.36893
# c = 11  0.83970     0.88520     0.90186      0.42767
# c = 12  0.84030     0.88086     0.90283      0.44661
# c = 13  0.83833     0.88061     0.90429      0.44308
# c = 14  0.83833     0.87646     0.89888      0.38757
# c = 15  0.83924     0.87763     0.89833      0.40505
# c = 16  0.83672     0.88379     0.90000      0.39752
# c = 17  0.83409     0.87959     0.90162      0.42348
# c = 18  0.83525     0.87985     0.90312      0.36782
# c = 19  0.84056     0.88081     0.89747      0.40691
# Train the classifier on the training data


# Create an SVM classifier with a linear kernel
clf = SVC(kernel='rbf', C=13)


clf.fit(X_train_norm, y_train)
# Test the classifier on the validation data to tune hyper parameter
#take the maximum accuracy eith kernal=rbf and c=13
# predict_set = clf.predict(X_val_norm)
# acc = accuracy_score(predict_set, y_val)
# print("Accuracy: ", acc)

# Test the classifier on the testing data
predict_set = clf.predict(X_test_norm)
acc = accuracy_score(predict_set, y_test)
print("Accuracy of  testing set is : ", acc)
#testing accuracy=0.8946


# Test the classifier on the training data
#predict_set = clf.predict(X_train_norm)
#acc = accuracy_score(predict_set, y_train)
#print("Accuracy: ", acc)
#training accuracy=0.98045


prediction_true = [0,0,0,0,0,0,0,0,0,0]
prediction_count = [0,0,0,0,0,0,0,0,0,0]
error_count = [0,0,0,0,0,0,0,0,0,0]

for i in range(len(predict_set)):
    if y_test[i] == 0:
        prediction_count[0] +=1
        if y_test[i] == predict_set[i]:
            prediction_true[0] +=1
        else:
            error_count[predict_set[i]] = error_count[predict_set[i]] +1
            print (predict_set[1])
    elif y_test[i] == 1:
        prediction_count[1] += 1
        if y_test[i] == predict_set[i]:
            prediction_true[1] += 1
    elif  y_test[i] == 2:
        prediction_count[2] += 1
        if y_test[i] == predict_set[i]:
            prediction_true[2] += 1
    elif  y_test[i] == 3:
        prediction_count[3] += 1
        if y_test[i] == predict_set[i]:
            prediction_true[3] += 1
    elif y_test[i] == 4:
        prediction_count[4] += 1
        if y_test[i] == predict_set[i]:
            prediction_true[4] += 1
    elif  y_test[i] == 5:
        prediction_count[5] += 1
        if y_test[i] == predict_set[i]:
            prediction_true[5] += 1
    elif  y_test[i] == 6:
        prediction_count[6] += 1
        if y_test[i] == predict_set[i]:
            prediction_true[6] += 1
    elif  y_test[i] == 7:
        prediction_count[7] += 1
        if y_test[i] == predict_set[i]:
            prediction_true[7] += 1
    elif  y_test[i] == 8:
        prediction_count[8] += 1
        if y_test[i] == predict_set[i]:
            prediction_true[8] += 1
    elif  y_test[i] == 9:
        prediction_count[9] += 1
        if y_test[i] == predict_set[i]:
            prediction_true[9] += 1
for i in range(10):
    print(f"Accuracy for C{i} =% {(prediction_true[i]/prediction_count[i]) * 100}")
print("")
for i in range(10):
    print(f"preduct type  C{i}  insted of c6 is = " , error_count[i] )
