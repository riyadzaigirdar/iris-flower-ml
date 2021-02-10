import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()

sepal_length = []
sepal_width = []
petal_length = []
petal_width = []

for i in iris.data:
    sepal_length.append(i[0])
    sepal_width.append(i[1])
    petal_length.append(i[2])
    petal_width.append(i[3])

dataframe = {
    "sepal_length": sepal_length,
    "sepal_width": sepal_width,
    "petal_length": petal_length,
    "petal_width": petal_width,
    "species": iris.target
}

# print(dataframe)
csv_converter = pd.DataFrame(dataframe)

csv_converter.to_csv("output.csv", index=False)
