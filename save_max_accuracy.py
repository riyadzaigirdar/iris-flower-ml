import os
import joblib
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def rewrite_enviroment_variables(accuracy):
    with open(".env", "w") as envfile:
        # save it in rom if the server crashes
        envfile.write(f"accuracy={accuracy}")
        # runtime enviroment value change
        os.environ["accuracy"] = str(accuracy)
    return True


def save_joblib(to_be_saved_model):
    joblib.dump(to_be_saved_model, "trained_model.joblib")
    return True


# load the enviroment variables
env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)

data = pd.read_csv("output.csv")

feature = data.drop(labels=["species"], axis=1)
label = data["species"]

model = DecisionTreeClassifier()

x_train, x_test, y_train, y_test = train_test_split(
    feature, label, test_size=0.2)

trained_model = model.fit(x_train, y_train)

predictions = trained_model.predict(x_test)

score = float(accuracy_score(predictions, y_test) * 100)

env_score = os.getenv("accuracy")

print("accuracy score", score)
print("env score", env_score)

if (score > float(env_score)):
    print("yes")
    save_model = save_joblib(trained_model)
    save_accuracy = rewrite_enviroment_variables(score)

    if (save_model and save_accuracy):
        print(
            f"successfulle save model and enviroment var to {score}")
else:
    print(f"No enviroment variables set , environ var {env_score}")
