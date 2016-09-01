# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 11:31:09 2016

@author: admin
"""
'''
from sklearn import tree 
features = [[140,1],[130,1],[150,0],[170,0]]
labels = [0,0,1,1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
print(clf.predict([[150,0]]))
'''
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
print(iris.feature_names)
print(iris.target_names)
print(iris.data[0])
#print(iris.data)
for i in range(len(iris.target)):
    #print("Example %d: label %s, feature %s" % (i, iris.target[i], iris.data[i]))
    pass    
    
test_idx = [0,50,100]
#training data
train_data = np.delete(iris.data, test_idx, axis=0)
train_target = np.delete(iris.target, test_idx)

#testing data
test_data = iris.data[test_idx]
test_target = iris.target[test_idx]

#training a classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)    
print(test_target)
print(clf.predict(test_data))

#viz code
from IPython.display import Image  
from sklearn.externals.six import StringIO
import pydot
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names=iris.feature_names,
                     class_names=iris.target_names, filled=True, rounded=True,
                     impurity=False)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
#graph.write_pdf('iris.pdf')        
#print(graph)
#Image(graph.create_png())   #no attribute 'create_png'








