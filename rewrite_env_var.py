import os
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(".") / ".env"  # read it from local dir env
load_dotenv(dotenv_path=env_path)

print(os.getenv("accuracy"))

with open(".env", "w") as envfile:
    envfile.write("accuracy=69")  # save it in rom if the server crashes
    os.environ["accuracy"] = "69"  # runtime enviroment value change

print(os.getenv("accuracy"))
