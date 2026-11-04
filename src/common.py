"""
shared helpers for the project, which include:
- a single random seed for reproducibility
- a single cross-validation object for all models, so they are compared on the same folds
- a function to load the olivetti dataset
- a function to evaluate a model or pipeline and return a dict of results

"""

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import StratifiedKFold, cross_validate

seed = 42

# one cv object for the whole project 5 folds, stratified by subject.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)


def load_data():
    x, y = fetch_olivetti_faces(return_X_y=True)
    return x, y


def evaluate(name, pipe, x, y):
    scores = cross_validate(pipe, x, y, cv=cv, scoring=['accuracy', 'f1_macro'])
    return {
        'name': name,
        'acc_mean': scores['test_accuracy'].mean(),
        'acc_std': scores['test_accuracy'].std(),
        'f1_mean': scores['test_f1_macro'].mean(),
        'f1_std': scores['test_f1_macro'].std(),
    }
