import os
from natsort import natsorted

"""
categories = {"business": 0, "entertainment": 1, "politics": 2, "sport": 3, "tech": 4}
true_positives = 0
total = 0
with open("testing_dataset.txt", "r") as f:
    read_file = f.readlines()
    for file in read_file:
        file_path = file.rstrip().split(" ")[0]
        category = file.rstrip().split(" ")[1]
        with open(file_path, "r") as f2:
            read_file2 = f2.readlines()
            text = ""
            for file2 in read_file2:
                text += file2.rstrip()
            prediction = classify(text, "dataset.txt")
            if prediction == categories[category]:
                true_positives += 1
            print(f"Prediction: {prediction} whereas actual is: {categories[category]}")
        total += 1

accuracy = true_positives / total
print(accuracy)
"""

"""with open("AG_News/test.csv") as file:
    lines = file.readlines()
    name = 0

    for line in lines:
        class_index = line.split(",")[0]
        title = line.split(",")[1]
        description = line.split(",")[2]
        if class_index != "1":
            continue

        with open(f"newbus/{str(name)}.txt", "w") as f:
            f.write(description)
        name += 1"""

world_files = natsorted(os.listdir("bbc/world"))

with open("dataset.txt", "a") as file:
    for wf in world_files:
        file.write(f"bbc/world/{wf} world\n")
