"""
Work in progress.
"""
import PyPDF2
from data_proccessing import fit_and_transform_dataset, dtm2list, even_out_list
import numpy as np
from knnclassifier import KNNClassifier

"""
mm_pdf = open("Authorversions.pdf", "rb")
pdf_reader = PyPDF2.PdfFileReader(mm_pdf)
page_one = pdf_reader.getPage(3)
mm_text = page_one.extractText()
mm_pdf.close()
"""


def classify(user_input: str, dataset: str):
    """
    The classify function classifies the value entered by the user.
    :param user_input: The input entered by the User. It accepts a string.
    :param dataset: The dataset path as a string containing the txt file.
    :return:
    """
    features, classes = fit_and_transform_dataset(user_input, dataset)
    flist = dtm2list(features)
    encoded_classes = {}
    count = 0
    for cls in classes:
        if cls not in encoded_classes:
            encoded_classes[cls] = count
            count += 1

    vectorised_labels = [encoded_classes[cls] for cls in classes if cls in encoded_classes]
    input_vector = np.array(even_out_list(flist)[-1])
    training_data = np.array(even_out_list(flist)[:-1])
    classifier = KNNClassifier(3)
    classifier.fit(input_vector, training_data, vectorised_labels)
    predicted_feature_vector, predicted_label, predicted_distance = classifier.predict()
    return predicted_label
