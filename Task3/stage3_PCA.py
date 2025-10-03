import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier
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
    metrics_scores = ['Accuracy', 'Precision', 'F1-Score']
    fig, axs = plt.subplots(len(metrics_scores), figsize=(10, 20))
    for i, metric in enumerate(metrics_scores):
        scores = [res[metric] for res in results.values()]
        names = list(results.keys())
        best_index = scores.index(max(scores))
        worst_index = scores.index(min(scores))
        bar_colors = ['red' if j == worst_index else ('green' if j == best_index else 'blue') for j in
                      range(len(scores))]
        axs[i].barh(names, scores, color=bar_colors)
        axs[i].set_title(metric)
        axs[i].set_xlim(0, 1)
        axs[i].set_xlabel(metric)
    plt.tight_layout()
    plt.show()


def load_dataset():
    global train_data, test_data
    train_data = load_training_data()
    test_data = load_test_data()


def prepare_data():
    global X_training, X_validation, y_training, y_validation, X_test, y_test
    X = train_data.drop(columns=['gesture name', 'gesture ID'])
    y = train_data['gesture ID']
    X.fillna(X.mean(), inplace=True)

    # Splitting the training data
    X_training, X_validation, y_training, y_validation = train_test_split(X, y, test_size=0.2, random_state=42)

    test_data.dropna(inplace=True)
    X_test = test_data.drop(columns=['gesture name', 'gesture ID'])
    y_test = test_data['gesture ID']
    X_test.fillna(X_training.mean(), inplace=True)


def select_classifiers():
    global classifiers
    classifiers = {
        'k-NN': KNeighborsClassifier(),
        'MLP': MLPClassifier(max_iter=1000),
        'SVM-linear': SVC(kernel='linear'),
        'SVM-poly-3': SVC(kernel='poly', degree=3),
        'SVM-poly-7': SVC(kernel='poly', degree=7),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(n_estimators=1000),
        'AdaBoost': AdaBoostClassifier(),
        'Bagging': BaggingClassifier(estimator=SVC(kernel='linear')),
        'Extra Trees': ExtraTreesClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
    }


def train_models():
    global results
    results = {}
    scaler = StandardScaler()
    X_normalized_training = scaler.fit_transform(X_training)
    # Have to use transform to avoid data leakage
    X_normalized_testing = scaler.transform(X_test)

    X_normalized_training = scaler.fit_transform(X_training)
    X_normalized_validation = scaler.transform(X_validation)  # Scaling the validation data

    # Apply PCA
    pca = PCA(n_components=0.95)  # variance
    X_pca_training = pca.fit_transform(X_normalized_training)
    X_pca_testing = pca.transform(X_normalized_testing)

    for name, clf in classifiers.items():
        # Train the classifier without PCA on training data
        clf.fit(X_normalized_training, y_training)

        # Evaluate the model
        y_val_pred = clf.predict(X_normalized_validation)

        # accuracy = accuracy_score(y_validation, y_val_pred)
        # precision = precision_score(y_validation, y_val_pred, average='weighted', zero_division=1)
        # f1 = f1_score(y_validation, y_val_pred, average='weighted')

        # print(
        #    f" {name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, F1-Score: {f1:.4f}")
        # results[name] = {'Accuracy': accuracy, 'Precision': precision, 'F1-Score': f1}

        # Train and test the classifier with PCA
        clf.fit(X_pca_training, y_training)
        y_pred = clf.predict(X_pca_testing)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(
            f"PCA {name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, F1-Score: {f1:.4f}")
        results["With PCA " + name] = {'Accuracy': accuracy, 'Precision': precision, 'F1-Score': f1}


if __name__ == '__main__':
    load_dataset()

    prepare_data()

    select_classifiers()

    train_models()
    show_results_plots()
    print(results)

'''k-NN - Accuracy: 0.3889, Precision: 0.5976, F1-Score: 0.3676
 Decision Tree - Accuracy: 0.5185, Precision: 0.5809, F1-Score: 0.5052
 MLP - Accuracy: 0.7037, Precision: 0.7934, F1-Score: 0.7068
 SVM-linear - Accuracy: 0.7315, Precision: 0.8297, F1-Score: 0.7405
 SVM-poly-3 - Accuracy: 0.3241, Precision: 0.8917, F1-Score: 0.3461
 SVM-poly-7 - Accuracy: 0.1019, Precision: 0.9568, F1-Score: 0.0908
 Random Forest - Accuracy: 0.7685, Precision: 0.8726, F1-Score: 0.7599
 AdaBoost - Accuracy: 0.1574, Precision: 0.8707, F1-Score: 0.1158
 Bagging - Accuracy: 0.7037, Precision: 0.7611, F1-Score: 0.7038
 Extra Trees - Accuracy: 0.8241, Precision: 0.8807, F1-Score: 0.8198
 Gradient Boosting - Accuracy: 0.6111, Precision: 0.7006, F1-Score: 0.6122
{'k-NN': {'Accuracy': 0.3888888888888889, 'Precision': 0.597601062842291, 'F1-Score': 0.36763433158360687}, 'Decision Tree': {'Accuracy': 0.5185185185185185, 'Precision': 0.5808862433862433, 'F1-Score': 0.5051509704287482}, 'MLP': {'Accuracy': 0.7037037037037037, 'Precision': 0.7934413580246913, 'F1-Score': 0.7068107304218414}, 'SVM-linear': {'Accuracy': 0.7314814814814815, 'Precision': 0.8297178130511462, 'F1-Score': 0.7404975682753459}, 'SVM-poly-3': {'Accuracy': 0.32407407407407407, 'Precision': 0.8917113698187145, 'F1-Score': 0.34605379188712526}, 'SVM-poly-7': {'Accuracy': 0.10185185185185185, 'Precision': 0.95679012345679, 'F1-Score': 0.0907936507936508}, 'Random Forest': {'Accuracy': 0.7685185185185185, 'Precision': 0.8725749559082892, 'F1-Score': 0.7598745390412055}, 'AdaBoost': {'Accuracy': 0.1574074074074074, 'Precision': 0.8706544227377562, 'F1-Score': 0.11575862241318988}, 'Bagging': {'Accuracy': 0.7037037037037037, 'Precision': 0.7611331569664903, 'F1-Score': 0.7038486719042274}, 'Extra Trees': {'Accuracy': 0.8240740740740741, 'Precision': 0.8807319223985891, 'F1-Score': 0.8198392142836585}, 'Gradient Boosting': {'Accuracy': 0.6111111111111112, 'Precision': 0.7005511463844798, 'F1-Score': 0.6122281011169899}}
'''
