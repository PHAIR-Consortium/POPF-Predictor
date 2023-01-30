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
    results['sensitivty'].append(sensitivity)

    specificity = 0 if tn + fp == 0 else round(tn / (tn + fp), 4)
    results['specificity'].append(specificity)

    roc_x, roc_y, _ = roc_curve(real_values.ravel(), pred_prob.ravel())
    results['AUC'].append(roc_auc_score(real_values, pred_prob))
    results['brier'].append(brier_score_loss(real_values, pred_prob))
    results['TP'].append(roc_y.tolist())
    results['FP'].append(roc_x.tolist())

    return results


def get_validation_metrics(best_grid, x_test, y_test, model):
    metrics = defaultdict(list)
    iterations = 50 if model == 'rf' else 1
    for i in range(iterations):
        y_pred = best_grid.predict(x_test)
        pred_prob = best_grid.predict_proba(x_test)[:, 1]
        metrics = get_metrics(y_test, y_pred, pred_prob, metrics)

    return metrics


def get_results(results, metrics):
    metric_names = list(metrics.keys())
    for metric_name in metric_names:
        metric = metrics[metric_name]
        temp = [sum(x) / len(metric) for x in zip(*metric)] if metric_name in ['TP', 'FP', 'CM'] else mean(metric)
        results[metric_name].append(temp)
    return results


def save_confusion_matrix(results, model):
    tn = np.mean([results['CM'][i][0][0] for i in range(len(results['CM']))])
    fn = np.mean([results['CM'][i][1][0] for i in range(len(results['CM']))])
    tp = np.mean([results['CM'][i][1][1] for i in range(len(results['CM']))])
    fp = np.mean([results['CM'][i][0][1] for i in range(len(results['CM']))])

    f = open(file_extension + '_results/' + model + '_confusion_matrix.csv', 'a')
    writer = csv.writer(f)
    writer.writerow(['Confusion Matrix'])
    writer.writerow(['TP, FP', tp, fp])
    writer.writerow(['FN, TN', fn, tn])
    f.close()


def save_results(results, model):
    metrics = list(results.keys())[1:8]
    mean_results = [np.mean(results[metric]) for metric in metrics]
    std_results = [np.std(results[metric]) for metric in metrics]

    f = open(file_extension + '_results/results.csv', 'a')
    writer = csv.writer(f)
    writer.writerow(['score'] + metrics)
    writer.writerow(['Results for ' + model])
    writer.writerow(['mean'] + mean_results)
    writer.writerow(['std'] + std_results)
    f.close()
