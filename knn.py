# -*- coding: utf-8 -*-

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


#In CSV file seniors=1, fourth=0
def str_to_numConverter(n1):
    """Function used to convert class from string to integer"""
    n2 = []
    for i in n1:
        if i=='seniors':
            n2.append(1)
        else:
            n2.append(0)
    return np.array(n2)


#Extracting data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

#Creating training values
x_train = train_data.iloc[:,[0,1]].values
y_train = str_to_numConverter(train_data.iloc[:,2].values)

#Creating values to to be tested for prediction
x_test = test_data.iloc[:,[0,1]].values
y_test = str_to_numConverter(test_data.iloc[:,2].values)

k = 5
#Creating model
model = KNeighborsClassifier(n_neighbors = k)
model.fit(x_train,y_train)

#Predicting values from that model
y_pred = model.predict(x_test)

print("Accuracy :", round(accuracy_score(y_test, y_pred),4))
print()
cm = confusion_matrix(y_test,y_pred)
print('Statistics:')
print('Value of k used :',k)
print('Total number of results predicted : ',len(y_pred))
print('Number of results correctly predicted : ',cm[0][0]+cm[1][1])
print('Number of results incorrectly predicted : ',cm[0][1]+cm[1][0])




#For plotting a graph
x_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('KNN Classification (k='+str(k)+')')
plt.xlabel('Shoe Size')
plt.ylabel('Height')
plt.legend()
plt.show()
