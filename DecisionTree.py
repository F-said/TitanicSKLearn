from sklearn import tree
from sklearn.tree import export_graphviz
import graphviz
from MLSurvival import *

# Pre-processing was handled in MLSurvival

'''Algorithm'''
clf = tree.DecisionTreeClassifier(max_depth=10, max_features="auto")
clf.fit(X_train, y_train)

'''Visualization'''
export_graphviz(clf, out_file='titanictree.dot', feature_names=list(X_train.columns.values), filled=True, rounded=True,
                special_characters=True)

