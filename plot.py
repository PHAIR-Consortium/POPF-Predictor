from matplotlib import pyplot as plt
from scipy import interpolate

from utils import *


def plot_jitter_curve(model, x_test=None, y_test=None):
    data_path = file_extension.split('_validate')[0] + '_training_models/' + model
    best_grid = pickle.load(open(data_path + '_model.pkl', 'rb'))
    if x_test is None: x_test = pickle.load(open(data_path + '_x_test.pkl', 'rb'))
    if y_test is None: y_test = pickle.load(open(data_path + '_y_test.pkl', 'rb'))
    scores, jitters = jitter_test(best_grid, x_test, y_test)

    f = interpolate.interp1d(jitters, scores, kind='linear')
    interpolated_scores = f([0, 0.5])

    plt.figure()
    lw = 2
    plt.plot(jitters, scores, color='violet', lw=lw, label='jitter curve for ' + model)
    plt.plot([0, 0.5], interpolated_scores, color='blue', lw=lw, label='jitter curve for ' + model + ' interpolated')
    plt.xlabel('Amount of Jitter')
    plt.ylabel('AUC')
    plt.title('AUC score for increasing jitter')
    plt.savefig(file_extension + '_results/' + model + '_jitter_scores.png')
    plt.close()


def plot_roc_curve(results, model):
    tprs = []
    base_fpr = np.linspace(0, 1, 101)

    plt.figure(figsize=(5, 5))
    plt.axes().set_aspect('equal', 'datalim')

    for i in range(len(results['TP'])):
        tpr = results['TP'][i]
        fpr = results['FP'][i]
        plt.plot(fpr, tpr, 'b', alpha=0.15)
        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)

    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)

    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std

    plt.plot(base_fpr, mean_tprs, 'b')
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)

    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title("Receiver operating characteristic for " + model)
    plt.savefig(file_extension + '_results/' + model + '_roc_curve.png')
    plt.close()
