from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from data_preproccessing import fit_and_transform_dataset, dtm2list, even_out_list
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from random import randint
user_input = """
Victor Wembanyama’s favorite players in the NBA right now are Kevin Durant and Giannis Antetokounmpo. That makes sense, given that they’re both taller than just about everyone else in the league and have all-world all-around games.

He might see some similarities there.

But comparisons to them, or anyone else, are not what Wembanyama is seeking. When the 7-foot-2 French teen comes to the NBA next season — by most accounts, he would be the No. 1 pick if the draft was held today — he isn’t interested in trying to become the next Durant, or Antetokounmpo, or Dirk Nowitzki.

The path he wants to take, he said in an interview with The Associated Press, will be all his own.

“I’m gonna tell you something that’s been going on in my life, like for my whole life, since I’ve been a kid, even before I played basketball,” Wembanyama said. “I’ve always tried to do (something) different. I’m not even talking about sports, whatever. Any field, I’m always trying to be original, something original, something one of one, something that’s never been done before. And this is really how it worked in my life. I don’t know where it comes from. I think I was born with it. I’ve always been trying to be original. Unique, that’s the word.”
"""
features, classes = fit_and_transform_dataset(user_input, "dataset.txt")
flist = dtm2list(features)
X = even_out_list(flist)[:-1]
my_input = even_out_list(flist)[-1]
encoded_classes = {}
count = 0
for cls in classes:
    if cls not in encoded_classes:
        encoded_classes[cls] = count
        count += 1
y = [encoded_classes[cls] for cls in classes if cls in encoded_classes]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)

"""
max_depths = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
n_estimators = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
min_samples_leaf = [1, 2, 4]
min_samples_split = [2, 5, 10]
random_max_depth_index = randint(0, len(max_depths)-1)
random_n_estimators_index = randint(0, len(n_estimators)-1)
random_min_samples_leaf_index = randint(0, len(min_samples_leaf)-1)
random_min_samples_split_index = randint(0, len(min_samples_split)-1)


clf = RandomForestClassifier(max_depth=max_depths[random_max_depth_index],
                             n_estimators=n_estimators[random_n_estimators_index],
                             min_samples_split=min_samples_split[random_min_samples_split_index],
                             min_samples_leaf=min_samples_leaf[random_min_samples_leaf_index])
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print(f"Random max depth index: {random_max_depth_index}")
print(f"Random n estimators index: {random_n_estimators_index}")
print(f"Random min samples leaf index: {random_min_samples_leaf_index}")
print(f"Random min samples split index: {random_min_samples_split_index}")
"""

clf = RandomForestClassifier(max_depth=90, n_estimators=2000, min_samples_leaf=1, min_samples_split=5)
clf.fit(X_train, y_train)
print(clf.predict([my_input]))


