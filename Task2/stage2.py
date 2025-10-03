import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier

from sklearn.metrics import precision_score, recall_score, f1_score

import numpy as np
from scipy.stats import uniform, randint
import matplotlib.pyplot as plt


def load_training_data():
    feature_names = [f"feature_{i}" for i in range(1, 243)]  # 242 feature columns
    data = pd.read_csv('datasets/train-final.csv', header=None, names=feature_names)
    data.rename(columns={
        data.columns[-2]: 'gesture name',
        data.columns[-1]: 'gesture ID'
    }, inplace=True)
    return data


def load_test_data():
    feature_names = [f"feature_{i}" for i in range(1, 243)]  # 242 feature columns
    data = pd.read_csv('datasets/test-final.csv', header=None, names=feature_names)
    data.rename(columns={
        data.columns[-2]: 'gesture name',
        data.columns[-1]: 'gesture ID'
    }, inplace=True)
    return data


def show_results_plots():

    metrics_scores = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    # Make a plot for every metric
    fig, axs = plt.subplots(len(metrics_scores), figsize=(10, 20))

    # Iterate every metric score
    for i, metric in enumerate(metrics_scores):
        scores = [res[metric] for res in results.values()]
        names = list(results.keys())

        # best the best result
        best_index = scores.index(max(scores))
        worst_index = scores.index(min(scores))

        # Color the bars, worst red, best green color
        bar_colors = ['red' if j == worst_index else ('green' if j == best_index else 'blue') for j in
                      range(len(scores))]

        axs[i].barh(names, scores, color=bar_colors)
        axs[i].set_title(metric)
        axs[i].set_xlim(0, 1)  # score between 0 and 1
        axs[i].set_xlabel(metric)
    plt.tight_layout()
    plt.show()


def select_normalization():
    # Possible to check out multiple methods for normalization

    global normalization_methods
    normalization_methods = {
        'StandardScaler': StandardScaler(),  # best here it seems
        # 'MinMaxScaler': MinMaxScaler(),    # no
        # 'MaxAbsScaler': MaxAbsScaler(),    # no
        # 'RobustScaler': RobustScaler()     # second best
    }


def select_data_subsets():
    # Making subset of the data is not really used at this time
    setup_subsets()
    global datasets
    datasets = {
        'whole_data': X_training,
        # 'positions_data': positions_data,             # Not used
        # 'cosine_angles_data': cosine_angles_data,     # Not used
        #  'mean_positions_data': mean_positions_data,  # Not used
        #  'std_positions_data': std_positions_data     # Not used
    }


def execute_normalization():
    global dataset_name, dataset
    for norm_name, scaler in normalization_methods.items():
        for dataset_name, dataset in datasets.items():
            normalized_data_train = scaler.fit_transform(dataset)
            # Extract the same columns from X_test as in the current dataset
            dataset_test = X_test[dataset.columns]
            normalized_data_test = scaler.transform(dataset_test)
            key_name = f"{dataset_name}_{norm_name}"
            normalized_datasets_train[key_name] = normalized_data_train
            normalized_datasets_test[key_name] = normalized_data_test


def select_classifiers():
    global classifiers
    classifiers = {
        'k-NN': KNeighborsClassifier(n_neighbors=6),
        'Decision Tree': DecisionTreeClassifier(ccp_alpha=0.01),
        'MLP': MLPClassifier(hidden_layer_sizes=(200,), max_iter=1000),
        'SVM-linear': SVC(kernel='linear'),
        'SVM-poly-3': SVC(kernel='poly', degree=3),
        'SVM-poly-20': SVC(kernel='poly', degree=20),
        'SVM-sigmoid': SVC(kernel='sigmoid'),
        'SVM-rbf': SVC(kernel='rbf', gamma='scale'),
        'Random Forest': RandomForestClassifier(n_estimators=1000),
        'AdaBoost': AdaBoostClassifier(n_estimators=50, learning_rate=1),

        # bagging and boosting
        'Bagging': BaggingClassifier(estimator=SVC(kernel='linear'), n_estimators=10, random_state=0),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=0),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1,
                                                        random_state=0),
    }


def print_results_console():
    print(
        f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    results[name + "_"] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }


def cross_validation_setup():
    global rf_params, svm_params
    # RandomizedSearchCV cross validation for random forest
    rf_params = {
        'n_estimators': randint(10, 1000),
        'max_features': ['sqrt', 'log2', None] + list(np.arange(1, X_train.shape[1] + 1, dtype=int)),
        'max_depth': [None] + list(randint(1, 50).rvs(10)),
        'min_samples_split': randint(2, 11),
        'min_samples_leaf': randint(1, 5),
        'bootstrap': [True, False]
    }
    # RandomizedSearchCV cross validation for SVM , might be useless as we use different poly anyway
    svm_params = {
        'C': uniform(0.1, 5),
        'gamma': ['scale', 'auto'],
        'shrinking': [True, False],
        'degree': [2, 3, 4, 5, 6, 7]  # more seems useless
    }


def load_dataset():
    global train_data, test_data
    # Loading training dataset
    train_d = load_training_data()
    # Loading the test dataset
    test_d = load_test_data()
    # Make dataframe to work with
    train_data = pd.DataFrame(train_d)
    test_data = pd.DataFrame(test_d)


def prepare_data():
    global X_training, y, X_test, y_test
    # Remove the target labels
    X_training = train_data.drop(columns=['gesture name', 'gesture ID'])
    # Assigning the target label
    y = train_data['gesture ID']
    X_training.fillna(X_training.mean(), inplace=True)
    # drop data what's not a number
    test_data.dropna(inplace=True)
    X_test = test_data.drop(columns=['gesture name', 'gesture ID'])
    y_test = test_data['gesture ID']
    X_test.fillna(X_training.mean(), inplace=True)


def setup_subsets():
    # Subsets of the data, not really used in this stage
    positions_data = X_training.iloc[:, :60]
    cosine_angles_data = X_training.iloc[:, 60:120]
    mean_positions_data = X_training.iloc[:, 120:180]
    std_positions_data = X_training.iloc[:, 180:240]


def train_models():
    global dataset_name, dataset, X_train, results, name, accuracy, precision, recall, f1
    for dataset_name, dataset in normalized_datasets_train.items():
        # splitting dataset into training and validation set.
        X_train, _, y_train, _ = train_test_split(dataset, y,
                                                  test_size=0.2,  # Validation 20%
                                                  random_state=42)
        cross_validation_setup()

        results = {}
        for name, clf in classifiers.items():

            if name == 'Random Forest':  # Add this new condition for Random Forest
                random_search = RandomizedSearchCV(clf,
                                                   param_distributions=rf_params,
                                                   n_iter=10,
                                                   scoring='accuracy',
                                                   cv=3,
                                                   verbose=0,
                                                   n_jobs=-1,
                                                   random_state=42)
                search_results = random_search.fit(X_train, y_train)
                best_clf = search_results.best_estimator_
                y_pred = best_clf.predict(normalized_datasets_test[dataset_name])
            if "SVM" in name:  # If the classifier is an SVM
                random_search = RandomizedSearchCV(clf, param_distributions=svm_params,
                                                   n_iter=100,
                                                   scoring='accuracy',
                                                   cv=5,
                                                   verbose=0,
                                                   n_jobs=-1,
                                                   random_state=42)
                search_results = random_search.fit(X_train, y_train)
                best_clf = search_results.best_estimator_
                y_pred = best_clf.predict(normalized_datasets_test[dataset_name])

            else:
                clf.fit(X_train, y_train)
                y_pred = clf.predict(normalized_datasets_test[dataset_name])

            # Calculate different scores
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred,
                                        average='weighted',
                                        zero_division=1)  # zero need
            recall = recall_score(y_test, y_pred,
                                  average='weighted')
            f1 = f1_score(y_test, y_pred,
                          average='weighted')

            # Print out the results
            print_results_console()


if __name__ == '__main__':

    # Load dataset
    load_dataset()

    # Set labels and features
    prepare_data()

    # Select normalization methods, globally
    select_normalization()

    # Possible to select part of the dataset, only one is used atm
    select_data_subsets()

    # Placeholders for normalized data
    normalized_datasets_train = {}
    normalized_datasets_test = {}

    # Start doing the normalization
    execute_normalization()

    # Select types of classifiers
    select_classifiers()

    # Train and evaluate models
    train_models()

# Plotting
show_results_plots()
