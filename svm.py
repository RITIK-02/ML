import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

#Function to convert Iris-type to values 0,1 and 2
def convert_iris_type(l):
    t = []
    for i in range(len(l)):
        if l[i] == "Iris-setosa":
            t.append(0)
        elif l[i] == "Iris-versicolor":
            t.append(1)
        else:
            t.append(2)
    return np.array(t)

#For plotting graph
def precision_plot():
    b1 = [1,2,3]
    b2 = [x+0.25 for x in b1]
    precision = [d["Iris-setosa"][0], d["Iris-versicolor"][0], d["Iris-virginica"][0]]
    f1 = [d["Iris-setosa"][1], d["Iris-versicolor"][1], d["Iris-virginica"][1]]
    print("[Iris-setosa, Iris-versicolor, Iris-virginica]")
    print("f1 score: ",f1)
    print("Precision score: ",precision)
    plt.bar(b1, precision, color="tab:blue",edgecolor="black", width=0.25, label="Precision")
    plt.bar(b2, f1, color="tab:orange",edgecolor="black", width=0.25, label="f1 score")
    plt.xticks([0.125+r for r in b1],["Iris-setosa","Iris-versicolor","Iris-virginica"])
    plt.ylim(0,1.2)
    plt.title("Plot of precision and f1 scores for different types of iris")
    plt.xlabel("Iris type", fontweight="bold")
    plt.ylabel("Score", fontweight="bold")
    plt.legend()
    plt.show()


#Reading csv
dataset = pd.read_csv("iris.data")
data = dataset.iloc[:,0:4]
data = np.array(data)
iris_names = dataset.iloc[:,4]
target = convert_iris_type(iris_names)

#Splitting data for training and testing model
"""If you REMOVE random_state parameter in the statement given below,
then shuffling of data happens before splitting it.
So to produce same result, random_state is set to 1."""
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size = 0.3, random_state=1)

#Model
model = svm.SVC(C=8.0, kernel = 'poly', degree=3)
model.fit(x_train, y_train)

#Predicting values
y_pred = model.predict(x_test)

#Report 
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix: \n",cm)
print('-'*50)
print("Plot of Precision and f1 score of different irises :")
clf_r = list((classification_report(y_test, y_pred)).split())
d = {"Iris-setosa":[float(clf_r[5]),float(clf_r[7])], "Iris-versicolor":[float(clf_r[10]),float(clf_r[12])], "Iris-virginica":[float(clf_r[15]),float(clf_r[17])]}
precision_plot()



