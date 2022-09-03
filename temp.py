from sklearn.feature_extraction.text import TfidfVectorizer


def read_dataset(dataset):
    with open(dataset, "r", encoding="utf-8") as f:
        lines = f.readlines()
        vec = TfidfVectorizer()
        sentences = []
        labels = []
        for line in lines:
            linepath = line.rstrip().split()[0].replace("\\", "/")
            label = line.rstrip().split()[1]
            labels.append(label)
            with open(linepath, "r") as f2:
                lines2 = f2.readlines()
                text = ""
                for line2 in lines2:
                    text += line2.strip()
                sentences.append(text)
    return vec.fit_transform(sentences), labels


features, classes = read_dataset("dataset.txt")
print(classes)
