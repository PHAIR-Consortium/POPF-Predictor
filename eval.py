import csv
from collections import defaultdict
from statistics import mean
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve, brier_score_loss

from utils import *


def get_metrics(real_values, pred_values, pred_prob, results):
    cm = confusion_matrix(real_values, pred_values)
    tn = cm[0][0]
    fn = cm[1][0]
    tp = cm[1][1]
    fp = cm[0][1]
    results['CM'].append(cm)

    results['f1'].append(f1_score(real_values, pred_values))

    precision = 0 if tp + fp == 0 else round(tp / (tp + fp), 4)
    results['precision'].append(precision)

    npv = 0 if tn + fn == 0 else round(tn / (tn + fn), 4)
    results['npv'].append(npv)

    sensitivity = 0 if fp + fn == 0 else round(tp / (tp + fn), 4)
    results['sensitivity'].append(sensitivity)

    specificity = 0 if tn + fp == 0 else round(tn / (tn + fp), 4)
    results['specificity'].append(specificity)

    roc_x, roc_y, _ = roc_curve(real_values.ravel(), pred_prob.ravel())
    results['AUC'].append(roc_auc_score(real_values, pred_prob))
    results['brier'].append(brier_score_loss(real_values, pred_prob))
    results['TP'].append(roc_y.tolist())
    results['FP'].append(roc_x.tolist())

    return results


def get_validation_metrics(best_grid, x_test, y_test):
    metrics = defaultdict(list)
    y_pred = best_grid.predict(x_test)
    pred_prob = best_grid.predict_proba(x_test)[:, 1]
    metrics = get_metrics(y_test, y_pred, pred_prob, metrics)

    return metrics, pred_prob


def get_results(results, metrics):
    metric_names = list(metrics.keys())
    for metric_name in metric_names:
        metric = metrics[metric_name]
        temp = [sum(x) / len(metric) for x in zip(*metric)] if metric_name in ['TP', 'FP', 'CM'] else mean(metric)
        results[metric_name].append(temp)
    return results


def save_confusion_matrix(results):
    cm = results['CM'][len(results['CM'])-1]
    tn = cm[0][0]
    fn = cm[1][0]
    tp = cm[1][1]
    fp = cm[0][1]

    # open file and overwrite existing file
    f = open(file_extension + '_results/confusion_matrix.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(['Confusion Matrix'])
    writer.writerow(['TP, FP', tp, fp])
    writer.writerow(['FN, TN', fn, tn])
    f.close()


def save_results(results, model, features, patients, true_events, pred_prob):
    metrics = list(results.keys())[1:8]
    best_results = [results[metric][len(results[metric])-1] for metric in metrics]

    f = open(file_extension + '_results/results.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(['Best results for ' + model])
    writer.writerow(metrics)
    writer.writerow(best_results)
    writer.writerow(['Model features', str(features)])
    writer.writerow(['Patients in test set', str(patients)])
    writer.writerow(['True events', str(true_events)])
    writer.writerow(['Predicted probabilities', str(pred_prob)])
    f.close()
