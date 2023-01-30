from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

param_grids = {
    'svm': {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear'],
            'probability': [True, False]},
    'lr': {'solver': ['newton-cg', 'lbfgs', 'liblinear'], 'penalty': ['l2'], 'C': [100, 10, 1.0, 0.1, 0.01]},
    'ri': {'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]},
    'knn': {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21], 'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']},
    'rf': {'bootstrap': [True], 'max_depth': [80, 90, 100, 110], 'max_features': [2, 3],
           'min_samples_leaf': [3, 4, 5], 'min_samples_split': [8, 10, 12],
           'n_estimators': [100, 200, 300, 1000]},
    'gb': {'n_estimators': [10, 100, 1000], 'learning_rate': [0.001, 0.01, 0.1], 'subsample': [0.5, 0.7, 1.0],
           'max_depth': [3, 7, 9]}}

classifiers = {'svm': svm.SVC(),
               'rf': RandomForestClassifier(),
               'lr': LogisticRegression(),
               'ri': RidgeClassifier(),
               'knn': KNeighborsClassifier(),
               'gb': GradientBoostingClassifier()}

iterations = {'svm': 1,
              'rf': 50,
              'lr': 1,
              'ri': 1,
              'knn': 1,
              'gb': 50}
