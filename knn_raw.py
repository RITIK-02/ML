from collections import Counter
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import neighbors
from mlxtend.plotting import plot_decision_regions
import warnings



def str_to_numConverter(n1):
    """Function used to convert class from string to integer"""
    n2 = []
    for i in n1:
        if i=='seniors':
            n2.append([1])
        else:
            n2.append([0])
    return np.array(n2)

def accuracy(y_pred,y_test):
    n=0
    for i in range(len(y_pred)):
        if y_test[i]==y_pred[i]:
            n+=1
    return round((n/len(y_pred)),4),n


def mean(l):
    """Used for regression datasets"""
    return sum(l)/len(l)


def mode(l):
    """Used for classification datasets"""
    return Counter(l).most_common(1)[0][0]


def euclidean_dist(p1, p2):
    """Distance function used to find shortest distance between two point"""
    sum_squared_dist = 0
    for i in range(len(p1)):
        sum_squared_dist += math.pow(p1[i] - p2[i], 2)
    return math.sqrt(sum_squared_dist)


def knn(train_data, test_data, k, choice_fn, dist_fn=euclidean_dist):
    """This is our knn model which find k number of nearest labels and then uses
        choice function to predict result"""
    dist_and_index = []
    for i,d in enumerate(train_data):
        dist = dist_fn(d[:-1], test_data)
        dist_and_index.append((dist, i))

    sorted_dist_and_index = sorted(dist_and_index)
    k_nearest_dist_and_index = sorted_dist_and_index[:k]
    k_nearest_labels = [train_data[i][-1] for d,i in k_nearest_dist_and_index]
    return k_nearest_dist_and_index, choice_fn(k_nearest_labels)


""" Now reading csv file and arranging them appropriately"""
#Extracting x values for training
cd1 = pd.read_csv("train.csv").iloc[:,:-1].values

#Extracting y values for training
cd2 = str_to_numConverter(pd.read_csv("train.csv").iloc[:,-1:].values)

#Extracting x values for testing model
cq1 = pd.read_csv("test.csv").iloc[:,:-1].values

#Extracting y values for comparing them with predicted values
cq2 = str_to_numConverter(pd.read_csv("test.csv").iloc[:,-1:].values)

#For graph as it requires proper 1D Numpy array
cq3 = pd.read_csv("test.csv").iloc[:,-1:].values
cq4 = np.array([1 if cq3[i]=="seniors" else 0 for i in range(len(cq3))])

#Concatenating them after conversion of all y values from strings into integers
clf_data = np.concatenate((cd1,cd2),axis=1)
clf_query = np.concatenate((cq1,cq2),axis=1)
k = 5
y_pred = []
for i in clf_query:
    clf_l, clf_p = knn(clf_data, i, k, choice_fn=mode)
    y_pred.append(clf_p)


acc = accuracy(y_pred,cq2)
print("STATS")
print("-"*100)
print("Value of k used :",k)
print("Number of correct predictions : ",acc[1])
print("Number of incorrect predictions : ",len(y_pred)-acc[1])
print("Total number of predictions : ",len(y_pred))
print("Accuracy :", acc[0])


#GRAPH WORK
warnings.filterwarnings('ignore')
clf = neighbors.KNeighborsClassifier(n_neighbors=k)
clf.fit(cd1, cd2)
plot_decision_regions(cq1, cq4, clf=clf, legend=2)
plt.xlabel('Shoe size')
plt.ylabel('Height')
plt.title('Knn with K='+ str(k))
plt.show()







