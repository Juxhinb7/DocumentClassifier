from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from data_proccessing import fit_and_transform_dataset, dtm2list, even_out_list
import xgboost
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

features, classes = fit_and_transform_dataset("""
Antoine Griezmann says he did everything that he could to stay at Atletico Madrid.

The forward signed permanently for the club until 2026 from Barcelona.

"To stay at Atletico, enjoy being at the club, with the coach, my team-mates, the stadium, the fans, I did everything possible to stay here,” he said.

"When I saw there was the chance to stay for many years, I spoke with the club. I knew what they wanted from me, I didn't think much. I didn't care what I had to do. The one thing I wanted was to stay here.

"I will give everything for my club, the trust of the coach, the fans, Atletico. I’ll give everything for the badge. I could miss chances, or passes, but I will give everything until the last second, as much as I can. I want that to be seen and felt.
""", "dataset.txt")
flist = dtm2list(features)


classes_keys = {}
count = 0
for cls in classes:
    if cls not in classes_keys:
        classes_keys[cls] = count
        count += 1

print(classes_keys)

num_classes = [classes_keys[cls] for cls in classes if cls in classes_keys]
my_input = even_out_list(flist)[-1]
X = even_out_list(flist)[:-1]
y = num_classes
"""
tfidf = TfidfVectorizer()

train_texts = []
train_labels = []

with open("AG_News/train.csv") as file:
    readfile = file.readlines()
    count = 0
    for item in readfile[1:]:
        label = int(item.split(",")[0])
        description = item.split(",")[2]
        train_texts.append(description)
        train_labels.append(label)


test_texts = []
test_labels = []

with open("AG_News/test.csv") as test_file:
    readfile = test_file.readlines()
    for item in readfile[1:]:
        label = int(item.split(",")[0])
        description = item.split(",")[2]
        test_texts.append(description)
        test_labels.append(label)

train_dtm = tfidf.fit_transform(train_texts)
test_dtm = tfidf.fit_transform(test_texts)

train_dtm2list = list(train_dtm.tolil().data)
test_dtm2list = list(test_dtm.tolil().data)


def even_list_out(train_data, test_data):
    train_max_len = 0
    test_max_len = 0

    for item in train_data:
        if len(item) > train_max_len:
            train_max_len = len(item)

    for item in test_data:
        if len(item) > test_max_len:
            test_max_len = len(item)

    lengths = [train_max_len, test_max_len]
    max_len = max(lengths)

    for item in train_data:
        while len(item) < max_len:
            item.append(0)

    for item in test_data:
        while len(item) < max_len:
            item.append(0)


even_list_out(train_dtm2list, test_dtm2list)
clf = RandomForestClassifier(max_depth=30,
                             random_state=0,
                             n_estimators=800,
                             min_samples_leaf=1,
                             min_samples_split=2)
clf.fit(train_dtm2list, train_labels)
y_pred = clf.predict(test_dtm2list)
print(classification_report(test_labels, y_pred))
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)


clf = RandomForestClassifier(max_depth=60, n_estimators=1200, min_samples_leaf=2, min_samples_split=5)
clf.fit(X_train, y_train)

"""result = clf.predict([my_input])
print(result)"""


y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))


"""with open("AG_News/test.csv") as file:
    lines = file.readlines()
    count = 0
    name = 497

    for line in lines:
        class_index = line.split(",")[0]
        title = line.split(",")[1]
        description = line.split(",")[2]
        if class_index != "4":
            continue

        with open(f"newbus/{str(name)}.txt", "w") as f:
            f.write(description)
        name += 1"""


