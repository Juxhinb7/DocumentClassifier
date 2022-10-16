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
The first thing I notice with the Meta Quest Pro is the fit. Even after eight years, Meta’s (formerly Facebook’s) virtual reality headsets are typically bulky, front-heavy affairs. But the Quest Pro rests around my head easily, with its battery shifted to a back mount and its electronics pared down to a lighter layer over my face. Though it’s bigger than your typical pair of glasses or even your typical ski mask, it’s a major step forward for the biggest VR headset maker around.

It’s clear where that step is going, but for now, I’m less sure where it’s landed. The Quest Pro is a $1,499 variation on the $399 Meta Quest 2, improving on that headset in several ways — from better ergonomics to an upgraded processor. It adds eye tracking and a high-resolution color video feed that blurs the conventional line between virtual and augmented reality. In theory, the Quest Pro primes Meta to enter a professional-oriented VR market that has, so far, been an afterthought for the Quest.

“This is the highest-end VR device — for enthusiasts, the prosumer, the sort of people who are trying to get work done,” Meta CEO Mark Zuckerberg told The Verge and a small group of reporters during a recent demo at the company’s research division in Redmond, Washington. Meta will continue selling the Quest 2, putting the Quest Pro in a separate high-end category.

In practice, the Meta Quest Pro seems a bit like a very sophisticated development kit, more geared toward testing next-gen technology than filling specific needs. Maybe I’ll feel differently when the headset ships on October 25th. But it’s not clear how strong a case Meta will make for a $1,500 device whose pragmatic benefits for many businesses remain debatable. And there’s one major downgrade from the Quest 2: a hit to battery life that could make the Quest Pro less attractive for some of the customers it’s meant to reach.
Meta has long ceded the high end of VR to companies like HTC, Varjo, and Valve, but the Quest Pro changes that. The headset bumps the Meta Quest 2’s internal specs: there’s a Snapdragon XR2-Plus processor instead of the Quest 2’s XR2, 12GB instead of 6GB of memory, and 256GB of storage instead of 128GB and 256GB models. It weighs 722 grams to the Quest 2’s 503 grams, but it’s far better balanced. (It’s also not far from the Quest 2’s weight with an optional Elite Strap, which adds an extra 173 grams or more.) Its screens offer a respectable 1800 x 1920 pixels per eye with a maximum 90Hz refresh rate, plus new display tech that Meta says offers 75 percent more contrast than the Quest 2’s. Other headsets can beat the Quest Pro on specific features, like the wired Varjo headset’s extraordinarily high-definition screen. But the combination of better baseline specs and specialized new features pushes it out of the Quest 2’s squarely midrange comfort zone. 
"""


def classify_data(user_input: str, dataset: str):
    features, classes = fit_and_transform_dataset(user_input, dataset)
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

    clf = RandomForestClassifier(max_depth=60, n_estimators=1200, min_samples_leaf=2, min_samples_split=5)
    clf.fit(X_train, y_train)
    print(clf.predict([my_input]))


