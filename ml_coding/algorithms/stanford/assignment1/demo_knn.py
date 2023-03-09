'''
-- np.flatnonzero(y_train == y)

-- X_train[idx].astype('uint8')

-- mask = list(range(num_training)) ; X_train = X_train[mask]

-- X_train.shape = (5000, 32, 32, 3) ; X_train = np.reshape(X_train, (X_train.shape[0], -1))
   X_train.shape = (5000, 3072)

-- terrible idea speedwise: dists[i,j] = np.sqrt(sum(X[i,:]- self.X_train[j,:])**2)
   sum() should be replaced with np.sum()

-- X[i,:]=X[i], not the case for X[:,i] obviously.

'''

from __future__ import print_function

import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import pdb
import time

from cs231n.classifiers import KNearestNeighbor

# Load the raw CIFAR-10 data.
cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
#y_train = np.reshape(y_train, (y_train.shape[0], 1)) # get rid of (5000,) shape
#y_test = np.reshape(y_test, (y_test.shape[0], 1))

# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
C = len(classes)
n = 7

samples = np.zeros([C*n]+list(X_train.shape[1:])) # 70*32*32*3
for i, label in enumerate(classes):
    inds = np.where(y_train==i)[0]
    picks = np.random.choice(inds, n)
    samples[i*n:(i+1)*n,:,:,:] = X_train[picks]
 
num_training = 5000
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)

classifier = KNearestNeighbor()
classifier.train(X_train, y_train)

'''
Compute distance between two sets
compute_distances_two_loops uses 2 for loops for that, very inefficient
compute_distances_one_loop uses python vectorization and 1 for loop
compute_distances_no_loops uses python brodcasting and no for loop

'''
t0 = time.time()
#dists = classifier.compute_distances_two_loops(X_test)
#t1 = time.time()
#print(t1-t0)


#dists_one = classifier.compute_distances_one_loop(X_test)
#t2 = time.time()
#print(t2-t1)

dists_two = classifier.compute_distances_no_loops(X_test)
t3 = time.time()
print(t3-t0)

y_test_pred = classifier.predict_labels(dists_two, k=1)
accuracy_k1 = np.sum(y_test_pred==y_test) / num_test

y_test_pred = classifier.predict_labels(dists_two, k=5)
accuracy_k5 = np.sum(y_test_pred==y_test) / num_test



import pdb; pdb.set_trace()

'''
Cross validation:
'''
num_folds = 2#5
k_choices = [1, 3]#, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = np.array_split(X_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)

k_to_accuracies = {}
num_split = X_train.shape[0] / num_folds
acc_k = np.zeros((len(k_choices), num_folds), dtype=np.float)

for ik ,k in enumerate(k_choices):
    for i in range(num_folds):
        train_set = np.concatenate((X_train_folds[:i]+X_train_folds[i+1:]))
        label_set = np.concatenate((y_train_folds[:i]+y_train_folds[i+1:]))
        classifier.train(train_set, label_set)
        y_pred_fold = classifier.predict(X_train_folds[i], k=k, num_loops=0)
        num_correct = np.sum(y_pred_fold == y_train_folds[i])
        acc_k[ik, i] = float(num_correct) / num_split
    k_to_accuracies[k] = acc_k[ik]

# Print out the computed accuracies
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))

for k in k_choices:
    accuracies = k_to_accuracies[k]
    plt.scatter([k] * len(accuracies), accuracies)
