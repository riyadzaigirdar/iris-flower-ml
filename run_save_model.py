import os
from time import sleep

for i in range(100):
    cmd = "python save_max_accuracy.py"
    os.system(cmd)
    sleep(3)
    print(f"run {i+1} done")

print("done")
