"""
Work in progress.
"""
import PyPDF2
from data_proccessing import fit_and_transform_dataset, dtm2list, even_out_list
import numpy as np
from knnclassifier import KNNClassifier

mm_pdf = open("Authorversions.pdf", "rb")
pdf_reader = PyPDF2.PdfFileReader(mm_pdf)
page_one = pdf_reader.getPage(3)
mm_text = page_one.extractText()
mm_pdf.close()

features, classes = fit_and_transform_dataset(mm_text, "dataset.txt")

flist = dtm2list(features)

classes_keys = {}

count = 0
for cls in classes:
    if cls not in classes_keys:
        classes_keys[cls] = count
        count += 1

num_classes = [classes_keys[cls] for cls in classes if cls in classes_keys]

input_vector = np.array(even_out_list(flist)[-1])
training_data = np.array(even_out_list(flist)[:-1])

classifier = KNNClassifier(3)
classifier.fit(input_vector, training_data, num_classes)
predicted_feature_vector, predicted_label, predicted_distance = classifier.predict()

print(even_out_list(flist))
print(f"Predicted class:{predicted_label}")
