import joblib
from sklearn.tree import export_graphviz

model = joblib.load("trained_model.joblib")

export_graphviz(model, out_file="output_tree.dot",
                feature_names=["sepal_length", "sepal_width",
                               "petal_length", "petal_width"],
                label='all',
                filled=True,
                rounded=True
                )
