import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def fit_and_transform_dataset(x: str, dataset: str):
    with open(dataset, "r") as f:
        lines = f.readlines()
        vec = TfidfVectorizer()
        texts = []
        labels = []
        for line in lines:
            linepath = line.strip().split()[0]
            label = line.strip().split()[1]
            labels.append(label)
            with open(linepath, "r") as f2:
                lines2 = f2.readlines()
                text = ""
                for line2 in lines2:
                    text += line2.strip()
                texts.append(text)
        texts.append(x)
    return vec.fit_transform(texts), labels


features, classes = fit_and_transform_dataset("hello there", "dataset.txt")
last_doc = features[-1]


def dtm2list(fvectors):
    vectorslist = []
    for vector in fvectors:
        data_vector = vector.tolil().data
        vectoraslist = data_vector.tolist()[0]
        vectorslist.append(vectoraslist)

    return vectorslist


flist = dtm2list(features)


def even_out_list(fvectors_list: []):
    longest_list = max(len(elem) for elem in fvectors_list)
    even_lists = []

    for element in fvectors_list:
        while len(element) < longest_list:
            element.append(0)
        even_lists.append(element)

    return even_lists


classes_keys = {"business": 0, "entertainment": 1, "politics": 2, "sport": 3, "tech": 4}

num_classes = [classes_keys[cls] for cls in classes if cls in classes_keys]



