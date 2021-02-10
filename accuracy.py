import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("output.csv")

feature = data.drop(labels=["species"], axis=1)
label = data["species"]

x_train, x_test, y_train, y_test = train_test_split(
    feature, label, test_size=0.2)

model = DecisionTreeClassifier()

trained_model = model.fit(x_train, y_train)

predictions = trained_model.predict(x_test)

score = accuracy_score(y_test, predictions)

print(score)
