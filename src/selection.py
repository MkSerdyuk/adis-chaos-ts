import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from sklearn.ensemble import AdaBoostClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .prediction import CLASSIFIER_FEATURES


def classifiers(seed=42):
    def lr(c=1.0):
        return LogisticRegression(C=c, max_iter=2000, random_state=seed)

    def svm():
        return SVC(kernel="poly", random_state=seed)

    ensemble = [(f"lr{i}", lr(c)) for i, c in enumerate([0.1, 1, 10])]
    models = {
        "lr": lr(),
        "svm": svm(),
        "dt": DecisionTreeClassifier(criterion="entropy", random_state=seed),
        "knn": KNeighborsClassifier(n_neighbors=3),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(8,), activation="logistic", max_iter=2000, random_state=seed
        ),
        "adblr": AdaBoostClassifier(estimator=lr(), n_estimators=30, random_state=seed),
        "adbsvm": AdaBoostClassifier(estimator=svm(), n_estimators=10, random_state=seed),
        "lrv": VotingClassifier(estimators=ensemble, voting="hard"),
        "lrs": StackingClassifier(estimators=ensemble, final_estimator=lr(), cv=5),
    }
    return {
        name: make_pipeline(StandardScaler(), SMOTE(random_state=seed), model)
        for name, model in models.items()
    }


def fit_selectors(features, predictions, truth, tolerance=0.05, seed=42):
    nonempty = features[..., 0] > 0
    x = features[nonempty][:, CLASSIFIER_FEATURES]
    y = (np.abs(predictions[nonempty] - truth[nonempty]) < tolerance).astype(int)
    if np.bincount(y, minlength=2).min() < 6:
        raise ValueError("Selector training needs at least six examples of each class")
    models = classifiers(seed)
    for model in models.values():
        model.fit(x, y)
    return models
