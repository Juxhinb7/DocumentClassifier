"""
Work in progress.
"""
import PyPDF2
from data_proccessing import fit_and_transform_dataset, dtm2list, even_out_list
import numpy as np

mm_pdf = open("Meeting Minutes.pdf", "rb")
pdf_reader = PyPDF2.PdfFileReader(mm_pdf)
page_one = pdf_reader.getPage(0)
mm_text = page_one.extractText()
mm_pdf.close()

features, classes = fit_and_transform_dataset(mm_text, "dataset.txt")
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

distances = [np.linalg.norm(invec - np.array(even_out_list(flist)[i])) for i in range(len(even_out_list(flist)) - 1)]

print(f"Classes:{num_classes}")
print(f"Distances:{distances}")
