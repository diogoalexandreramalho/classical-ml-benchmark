import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def apply_classifier(clf, X, y):
    accs = cross_val_score(clf, X, y, cv=5)
    accuracy = sum(accs) / len(accs)
    return accuracy


def apply_classifiers(X, y):
    clfs = [
        GaussianNB(),
        KNeighborsClassifier(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        GradientBoostingClassifier(),
        XGBClassifier(),
    ]
    accs = []

    for clf in clfs:
        accs += [apply_classifier(clf, X, y)]

    return accs


def nothing(X, y):
    apply_classifiers(X, y)


def standardScale(X, y):
    pass


def compareResults(data, source):
    if source == "PD":
        target = "class"
    else:
        target = "Cover_Type"

    y: np.ndarray = data.pop(target).values
    X: np.ndarray = data.values

    nothing(X, y)
    standardScale(X, y)
