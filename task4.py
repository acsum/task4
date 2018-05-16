import pandas as pd
import numpy as np

from sklearn.semi_supervised import label_propagation, LabelPropagation
from sklearn.semi_supervised import LabelSpreading

from sklearn.metrics import confusion_matrix, classification_report


trainlabeledfilename = '/Users/Anna/polybox/IntroductionML/Tasks/04/task4_s8n2k3nd/train_labeled.h5'
trainunlabeledfilename = '/Users/Anna/polybox/IntroductionML/Tasks/04/task4_s8n2k3nd/train_unlabeled.h5'
testfilename = '/Users/Anna/polybox/IntroductionML/Tasks/04/task4_s8n2k3nd/test.h5'

train_labeled = pd.read_hdf(trainlabeledfilename, "train")
train_unlabeled = pd.read_hdf(trainunlabeledfilename, "train")
test = pd.read_hdf(testfilename, "test")

y_train_labeled = train_labeled['y'].values
X_train_labeled = train_labeled._drop_axis(['y'], axis=1).values
X_train_unlabeled = train_unlabeled.values
X_test = test.values

n_labeled_points = 9000;
n_unlabeled_points = 21000;
n_total_samples = 30000;

indices = np.arange(n_total_samples)
unlabeled_set = indices[n_labeled_points:]

# -1 denotes unlabeled points
y_train = np.concatenate((y_train_labeled, -1*np.ones(21000)), axis=0).astype(int) #3000x1
X_train = np.append(X_train_labeled, X_train_unlabeled, axis=0) #30000x28

model = LabelPropagation()
#model = LabelSpreading(kernel='rbf')
#model = label_propagation.LabelSpreading(gamma=0.25, max_iter=30)

print("here")
model.fit(X_train, y_train)
y_pred = model.predict(X_test) #8000x128 ?
print("here1")
predicted_labels = model.transduction_[unlabeled_set]
print("here2")

d = {'Id': test.index, 'y': y_pred}
out = pd.DataFrame(d)
out.to_csv('task4_output.csv', index=False)
