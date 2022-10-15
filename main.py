"""
Work in progress.
"""
from data_proccessing import fit_and_transform_dataset, dtm2list, even_out_list
import numpy as np
from knnclassifier import KNNClassifier


features, classes = fit_and_transform_dataset("""
KL Rahul bats at his best when he has a “nothing to lose” approach and that is how former Australia all-rounder Shane Watson wants the opener to play at the T20 World Cup beginning later this month.

Rahul’s strike rate came under scanner during the Asia Cup and the ongoing home series against South Africa.

Though India had to chase down only 107 in the first T20 against the Proteas, questions were raised over Rahul’s intent as he ended with 51 not out off 56 balls
""", "dataset.txt")
last_text = features[-1]

flist = dtm2list(features)

classes_keys = {}

count = 0
for cls in classes:
    if cls not in classes_keys:
        classes_keys[cls] = count
        count += 1

num_classes = [classes_keys[cls] for cls in classes if cls in classes_keys]

invec = np.array(even_out_list(flist)[-1])

classifier = KNNClassifier(3)
classifier.fit(invec, even_out_list(flist)[:-1], num_classes)
predicted_feature_vector, predicted_label, predicted_distance = classifier.predict()

print(f"Predicted class:{predicted_label}")
