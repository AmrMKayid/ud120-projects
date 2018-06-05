#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

from sklearn import svm
clf = svm.SVC(kernel='rbf', C=10000.0)

# These lines effectively slice the training dataset down to 1% of its original size, tossing out 99% of the training data
features_train = features_train[:len(features_train)//100] 
labels_train = labels_train[:len(labels_train)//100] 

t0 = time()
clf.fit(features_train, labels_train)
print("training time:", round(time()-t0, 3), "s")

t1 = time()
pred = clf.predict(features_test)
print("testing time:", round(time()-t1, 3), "s")

print("Pred 10: ", pred[10])
print("Pred 26: ", pred[26])
print("Pred 50: ", pred[50])

# Count for Chris (1) class
c = 0
for i in pred:
    if i == 1:
        c += 1

print("Chris emails: ", c)

print(clf.score(features_test, labels_test))

#########################################################


