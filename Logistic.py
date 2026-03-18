import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from Partitioner import getDefaultWineSets

seed = 1234
np.random.seed(seed)  


os.chdir(os.path.dirname(os.path.abspath(__file__)))

x_tr, x_val, x_te, y_tr, y_val, y_te = getDefaultWineSets()

def remap_quality(q):
    if q <= 4:   return 0  # low
    elif q <= 6: return 1  # medium
    else:        return 2  # high

y_tr  = np.array([remap_quality(q) for q in y_tr])
y_val = np.array([remap_quality(q) for q in y_val])
y_te  = np.array([remap_quality(q) for q in y_te])

print('train class distribution:', np.unique(y_tr,  return_counts=True))
print('val   class distribution:', np.unique(y_val, return_counts=True))
print('test  class distribution:', np.unique(y_te,  return_counts=True))

scaler = StandardScaler()
x_tr  = scaler.fit_transform(x_tr)
x_val = scaler.transform(x_val)
x_te  = scaler.transform(x_te)

print('\ntraining examples:  ', x_tr.shape[0])
print('val examples:', x_val.shape[0])
print('testing examples:   ', x_te.shape[0])


def log_class(x_tr, x_val, x_te, y_tr, y_val, y_te):
       
    param_grid = [
        {   # L2 — keep all features, less aggressive
            'l1_ratio': [0],
            'C': [0.1, 1, 10, 100],        # wider C range
            'solver': ['lbfgs'],
            'max_iter': [10000],
            'class_weight': ['balanced']
        },
        {   # Mild ElasticNet — light L1 blend only
            'l1_ratio': [0.1, 0.2, 0.3],   # keep L1 ratio LOW
            'C': [1, 10, 100],             # less regularization
            'solver': ['saga'],
            'max_iter': [50000],
            'tol': [1e-3],
            'class_weight': ['balanced']
        }
    ]

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    grid = GridSearchCV(
        LogisticRegression(fit_intercept=True),
        param_grid,
        cv=cv,
        scoring='f1_macro',   # better than accuracy for imbalanced classes
        n_jobs=-1,
        verbose=1
    )

    grid.fit(x_tr, y_tr)

    print('\nBest Parameters:', grid.best_params_)
    print('Best CV F1 Macro:', format(100 * grid.best_score_, '.2f'))

    best = grid.best_estimator_

    # best = LogisticRegression(C= 10, class_weight= 'balanced', l1_ratio= 0, max_iter = 10000, solver = 'lbfgs')

    # best.fit(X_tr, y_tr)


    print('\ntrain')
    print('Accuracy:', format(100 * accuracy_score(y_tr,  best.predict(x_tr)),  '.2f'))

    print('\nval')
    print('Accuracy:', format(100 * accuracy_score(y_val, best.predict(x_val)), '.2f'))

    print('\ntest')
    print('Accuracy:', format(100 * accuracy_score(y_te,  best.predict(x_te)),  '.2f'))

    print('\nClassification Report:')
    print(classification_report(
        y_te, best.predict(x_te),
        target_names=['Low (≤4)', 'Medium (5-6)', 'High (≥7)'],
        zero_division=0
    ))

    return best


classifier = log_class(x_tr, x_val, x_te, y_tr, y_val, y_te)