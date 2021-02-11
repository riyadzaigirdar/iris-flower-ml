# IMPORTANT: need to install brew install gprof2dot (on mac)
# https://www.geeksforgeeks.org/stringio-module-in-python/
# https://github.com/pydot/pydot
import pydot
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from io import StringIO


iris = load_iris()


# print(iris.feature_names)
# # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
# print(iris.target_names)  # ['setosa' 'versicolor' 'virginica']
# print(iris.data[0])  # [5.1 3.5 1.4 0.2]
# print(iris.target[0])  # 0 -> setosa
# print(iris.target)
# # [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# #  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
# #  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
# #  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
# #  2 2]

# for i in range(len(iris.target)):
#     print("index %d : feature %s -> label -> %s " %
#           (i, iris.data[i], iris.target[i]))
# index 0 : feature [5.1 3.5 1.4 0.2] -> label -> 0
# index 1 : feature [4.9 3.  1.4 0.2] -> label -> 0
# index 2 : feature [4.7 3.2 1.3 0.2] -> label -> 0
# index 3 : feature [4.6 3.1 1.5 0.2] -> label -> 0
# index 4 : feature [5.  3.6 1.4 0.2] -> label -> 0
# index 5 : feature [5.4 3.9 1.7 0.4] -> label -> 0
# ... ... ... .. ..  ... ... . .. ..... ... .... ..
# ... ... ... .. ..  ... ... . .. ..... ... .... ..
# ... ... ... .. ..  ... ... . .. ..... ... .... ..
# index 145 : feature [6.7 3.  5.2 2.3] -> label -> 2
# index 146 : feature [6.3 2.5 5.  1.9] -> label -> 2
# index 147 : feature [6.5 3.  5.2 2. ] -> label -> 2
# index 148 : feature [6.2 3.4 5.4 2.3] -> label -> 2
# index 149 : feature [5.9 3.  5.1 1.8] -> label -> 2

model = DecisionTreeClassifier()

test_idx = [0, 50, 100]

train_data = np.delete(iris.data, test_idx, axis=0)
train_target = np.delete(iris.target, test_idx)

test_data = iris.data[test_idx]
# [[5.1 3.5 1.4 0.2]
#  [7.  3.2 4.7 1.4]
#  [6.3 3.3 6.  2.5]]
test_target = iris.target[test_idx]
# [0 1 2]
model.fit(train_data, train_target)

print("correct data")
print(test_target)  # [0 1 2]
print("predictions")
print(model.predict(test_data))  # [0 1 2]


baby = export_graphviz(model, out_file="baby.dot",
                       feature_names=["sepal_length", "sepal_width",
                                      "petal_length", "petal_width"],
                       label='all',
                       filled=True,
                       rounded=True
                       )
# corrupted but works in tutorial 2016 year
dot_data = StringIO(baby)

graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("tree2.pdf")

# works perfect
# straight reading from dot file
graph = pydot.graph_from_dot_file("baby.dot")
graph[0].write_pdf("tree.pdf")
