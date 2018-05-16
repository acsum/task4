import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB

train_labeled = pd.read_hdf("train_labeled.h5", "train")
train_unlabeled = pd.read_hdf("train_unlabeled.h5", "train")
test = pd.read_hdf("test.h5", "test")

y_train_labeled = train_labeled['y']
x_train_labeled = train_labeled._drop_axis(['y'], axis=1)
x_train_unlabeled = train_unlabeled

#Switch to numpy
# Preprocessing X
x_train=[]
x_train_labeled = np.array(x_train_labeled)
x_train_unlabeled = np.array(x_train_unlabeled)
x_train.extend(x_train_labeled)
x_train.extend(x_train_unlabeled)
x_test = np.array(test)

# Preprocessing y
y_train_labeled = np.array(y_train_labeled)
ones = -1*np.ones(21000)
ones = np.array(ones)
y_train = np.concatenate((y_train_labeled, ones))

# Trying Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)

# output results
d={'Id': test.index, 'y': y_pred}
output=pd.DataFrame(d)
output.to_csv('output1.csv', index=False)

# from sklearn.metrics import accuracy_score
# acc = accuracy_score(y, y_pred)