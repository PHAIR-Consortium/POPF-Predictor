import os
import pickle
import numpy as np
import pandas as pd
import sdv
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import pearsonr
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Lasso
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from hyperparameters import classifiers, param_grids

from settings import file_extension, undersample, oversample, noise, synthetic, lasso


def create_folders():
    folders = ['results', 'models'] if "validate" not in file_extension else ['results']
    for name in folders:
        directory = file_extension + '_' + name + '/'
        if not os.path.exists(directory): os.makedirs(directory)


def read_files():
    df_radiomics = pd.read_excel('data/' + file_extension + "_radiomics.xlsx", engine='openpyxl')
    df_radiomics = df_radiomics.loc[:, ~df_radiomics.columns.duplicated()]
    start_idx = df_radiomics.columns.get_loc("original_shape_Elongation")
    df_radiomics = df_radiomics.iloc[:, start_idx:]

    df_events = pd.read_excel('data/' + "db_basis_" + file_extension + ".xlsx", engine='openpyxl')
    df_events = df_events.rename(columns={"Randomisatie nummer": "PP"})
    df_events = df_events.loc[:, ~df_events.columns.str.contains('^Unnamed')]

    df = pd.merge(df_radiomics, df_events, on='PP', how='outer').drop_duplicates()
    df = df[df['Event'].notna()]

    return df


def train_test_split(df, test_size):
    df = shuffle(df)

    df_test = df.iloc[:test_size, :]
    df_test = df_test.reset_index()
    df_test = df_test.drop(columns='index')

    df_train = df.iloc[test_size:, :]
    df_train = df_train.reset_index()
    df_train = df_train.drop(columns='index')

    if synthetic:
        model = sdv.tabular.GaussianCopula(primary_key='PP')
        model.fit(df_train)
        df_train += model.sample(200)

    return df_train, df_test


def normalize_data(df, min_max_scaler=None):
    events = df.Event.values
    vals = df.drop(columns=['Event', 'PP']).values
    cols = df.drop(columns=['Event', 'PP']).columns
    if not min_max_scaler: min_max_scaler = preprocessing.MinMaxScaler()
    vals_scaled = min_max_scaler.fit_transform(vals)
    df = pd.DataFrame(vals_scaled, columns=cols)
    df['Event'] = events
    return df, min_max_scaler


def variance_threshold_selector(data, threshold=0.5):
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]


def remove_variance(df):
    min_variance = 1e-3
    low_variance = variance_threshold_selector(df, min_variance)
    return low_variance


def pearson_correlation(df):
    df_pearson = []
    for index in range(0, len(df.columns) - 1):
        name = df.columns[index]
        df_pearson.append({'Name': name,
                           'Correlation': pearsonr(df.Event, df.iloc[:, index])[0],
                           'P-value': pearsonr(df.Event, df.iloc[:, index])[1]})
    df_pearson = pd.DataFrame(df_pearson)
    df_pearson = df_pearson[df_pearson['P-value'] < 0.05]
    features = df_pearson.Name.to_list() + ['Event']
    df_train = df.loc[:, features]
    return df_train, features


def lasso_reduction(df_train, df_test, features):
    x_train = df_train.to_numpy()[:, :-1]
    y_train = df_train.Event.to_numpy()

    pipeline = Pipeline([('scaler', StandardScaler()), ('model', Lasso())])
    search = GridSearchCV(pipeline, {'model__alpha': np.arange(0.1, 10, 0.1)}, cv=5, scoring="neg_mean_squared_error",
                          verbose=0)
    search.fit(x_train, y_train)
    coefficients = search.best_estimator_.named_steps['model'].coef_
    importance = np.abs(coefficients)
    features = np.array(features[0:len(features) - 1])
    features_lasso = list(features[importance > 0]) + ['Event']

    df_train = df_train.loc[:, features_lasso]
    df_test = df_test.loc[:, features_lasso]

    return df_train, df_test, features_lasso


def preprocess_train_data(df_train):
    df_train, min_max_scaler = normalize_data(df_train)
    df_train = remove_variance(df_train)
    if not lasso: df_train, features = pearson_correlation(df_train)
    if lasso: features = df_train.columns

    return df_train, min_max_scaler, features


def preprocess_test_data(df, min_max_scaler, features):
    df, _ = normalize_data(df, min_max_scaler=min_max_scaler)

    return df.loc[:, features]


def preprocess_data(df_train, df_test):
    df_train, min_max_scaler, features = preprocess_train_data(df_train)
    df_test = preprocess_test_data(df_test, min_max_scaler, features)
    if lasso: df_train, df_test, features = lasso_reduction(df_train, df_test, features)

    x_train, y_train = df_train.to_numpy()[:, :-1], df_train.Event.to_numpy()
    x_test, y_test = df_test.to_numpy()[:, :-1], df_test.Event.to_numpy()

    return x_train, y_train, x_test, y_test, min_max_scaler, features


def get_best_grid(x_train, y_train, model):
    if 0.9 * len(y_train) - sum(y_train) > sum(y_train):
        oversampler = RandomOverSampler(sampling_strategy=0.90)
        if oversample: x_train, y_train = oversampler.fit_resample(x_train, y_train)
    undersampler = RandomUnderSampler(sampling_strategy='majority')
    if undersample: x_train, y_train = undersampler.fit_resample(x_train, y_train)
    if noise: x_train = x_train + np.random.normal(0.0, 0.2, size=x_train.shape)

    param_grid = param_grids[model]
    grid = GridSearchCV(classifiers[model], param_grid=param_grid, scoring="roc_auc", refit=True, cv=5, n_jobs=-1,
                        verbose=0)
    grid.fit(x_train, y_train)
    return grid


def save_best_model(results, best_grid, x_test, y_test, features, scaler, model):
    if max(results['AUC']) == results['AUC'][len(results['AUC']) - 1]:
        pickle.dump(best_grid, open(file_extension + '_models/' + model + '_model.pkl', 'wb'))
        pickle.dump(x_test, open(file_extension + '_models/' + model + '_x_test.pkl', 'wb'))
        pickle.dump(y_test, open(file_extension + '_models/' + model + '_y_test.pkl', 'wb'))
        pickle.dump(features, open(file_extension + '_models/' + model + '_features.pkl', 'wb'))
        pickle.dump(scaler, open(file_extension + '_models/' + model + '_scaler.pkl', 'wb'))


def jitter(x, scale=0.1):
    return x + np.random.normal(0, scale, x.shape)


def jitter_test(classifier, x_test, y_test, scales=np.linspace(0, 0.5, 30), n=5):
    out = []
    for s in scales:
        avg = 0.0
        for r in range(n):
            avg += roc_auc_score(y_test, classifier.predict_proba(jitter(x_test, s))[:, 1])
        out.append(avg / n)
    return out, scales
