from eval import *
from plot import *
from settings import num_loops


def train_model(df, models):
    test_size = int(len(df) * 0.2)
    for model in models:
        results = defaultdict(list)
        for loop in range(1, num_loops):
            print("Running ", model, " model number ", loop)

            df_train, df_test = train_test_split(df, test_size)
            x_train, y_train, x_test, y_test, scaler, features = preprocess_data(df_train, df_test)

            if len(x_train[0]) < 1 or (len(x_train[0]) < 2 and model == 'rf'): continue

            best_grid = get_best_grid(x_train, y_train, model)
            metrics = get_validation_metrics(best_grid, x_test, y_test, model)
            results = get_results(results, metrics)
            save_best_model(results, best_grid, x_test, y_test, features, scaler, model)

        if results != {}:
            save_results(results, model)
            save_confusion_matrix(results, model)
            plot_roc_curve(results, model)
            plot_jitter_curve(model)


def validate_model(df, models):
    for model in models:
        print("Running validation for model ", model)
        results = defaultdict(list)

        best_grid = pickle.load(
            open(file_extension.split('validate')[0] + 'training_models/' + model + '_model.pkl', 'rb'))
        features = pickle.load(
            open(file_extension.split('validate')[0] + 'training_models/' + model + '_features.pkl', 'rb'))
        scaler = pickle.load(
            open(file_extension.split('validate')[0] + 'training_models/' + model + '_scaler.pkl', 'rb'))

        df_validate = preprocess_test_data(df, scaler, features)
        x_val, y_val = df_validate.to_numpy()[:, :-1], df_validate.Event.to_numpy()

        metrics = get_validation_metrics(best_grid, x_val, y_val, model)
        get_results(results, metrics)
        save_results(results, model)
        save_confusion_matrix(results, model)
        plot_roc_curve(results, model)
        plot_jitter_curve(model, x_val, y_val)
