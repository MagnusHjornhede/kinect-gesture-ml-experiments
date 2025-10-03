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
        'Decision Tree': DecisionTreeClassifier(),
        'MLP': MLPClassifier(max_iter=1000),
        'SVM-linear': SVC(kernel='linear'),
        'SVM-poly-3': SVC(kernel='poly', degree=3),
        'SVM-poly-7': SVC(kernel='poly', degree=7),
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

    # Transform to avoid data leakage
    X_normalized_testing = scaler.transform(X_test)

    X_normalized_training = scaler.fit_transform(X_training)
    X_normalized_validation = scaler.transform(X_validation)  # Scaling the validation data

    # Apply PCA on the normalized data
    pca = PCA(n_components=0.95)  # variance
    X_pca_training = pca.fit_transform(X_normalized_training)
    X_pca_testing = pca.transform(X_normalized_testing)

    for name, clf in classifiers.items():
        # Train without PCA on training data
        clf.fit(X_normalized_training, y_training)

        # Evaluate
        y_val_pred = clf.predict(X_normalized_validation)

        accuracy = accuracy_score(y_validation, y_val_pred)
        precision = precision_score(y_validation, y_val_pred, average='weighted', zero_division=1)
        f1 = f1_score(y_validation, y_val_pred, average='weighted')

        print(
            f" {name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, F1-Score: {f1:.4f}")
        results[name] = {'Accuracy': accuracy, 'Precision': precision, 'F1-Score': f1}

        # Train and test with PCA
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


