from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_training_data():
    # Give the columns names like
    # column_names = ['feature1', 'feature2', 'feature3', ..., 'gesture_name', 'gesture_id', 'candidate']

    feature_names = [f"feature_{i}" for i in range(1, 243)]  # 242 feature columns
    data = pd.read_csv('datasets/train-final.csv', header=None, names=feature_names)
    data.rename(columns={
        data.columns[-2]: 'gesture name',
        data.columns[-1]: 'gesture ID'

    }, inplace=True)
    return data


def load_test_data():
    # Give the columns names
    feature_names = [f"feature_{i}" for i in range(1, 243)]  # 242 feature columns
    data = pd.read_csv('datasets/test-final.csv', header=None, names=feature_names)
    data.rename(columns={
        data.columns[-2]: 'gesture name',
        data.columns[-1]: 'gesture ID'

    }, inplace=True)
    return data


def check_missing_values(_input: pd.DataFrame):
    missing_values = _input.isnull().sum()
    columns_with_missing_values = missing_values[missing_values > 0]
    print("Missing values")
    print(columns_with_missing_values)


def print_heatmap(_input: pd.DataFrame, _name: str):
    plt.figure(figsize=(12, 5))
    sns.heatmap(_input.isnull(), cmap='viridis', cbar=True, yticklabels=False)
    plt.title(_name)
    plt.show()


def plot_boxplot(_input: pd.DataFrame, _title: str):
    _input.boxplot(figsize=(12, 8))
    plt.xticks(rotation=90)  # Rotate x labels
    plt.title(_title)
    plt.show()

# Plot the histogram
def plot_histogram(series: pd.Series, _title:str):
    plt.figure(figsize=(10, 6))
    series.value_counts().plot(kind='bar', color='blue', alpha=0.7)
    plt.xlabel('Gesture ID')
    plt.ylabel('Count')
    plt.title(_title)
    plt.show()

if __name__ == '__main__':

    ################
    # Pipeline start
    ################ 
    # Load the datasets
    train_d = load_training_data()
    test_d = load_test_data()
    # convert into pandas dataFrame
    train_data = pd.DataFrame(train_d)
    test_data = pd.DataFrame(test_d)

    # Check for missing values
    print("Checking train_data")
    check_missing_values(train_data)
    print_heatmap(train_data, "Missing data points in train-final.csv")
    print("Check test_data")
    check_missing_values(test_data)
    print_heatmap(test_data, "Missing data points in test-final.csv")

    # Assign data for training
    X_training = train_data.drop(columns=['gesture name', 'gesture ID'])
    y = train_data['gesture ID']  # gesture ID as the label
    X_training.fillna(X_training.mean(), inplace=True)
    # X.dropna(inplace=True)

    # Assign data for testing
    test_data.dropna(inplace=True)  # Just drop rows missing testdata
    X_test = test_data.drop(columns=['gesture name', 'gesture ID'])
    y_test = test_data['gesture ID']  # gesture ID as label
    X_test.fillna(X_training.mean(), inplace=True)

    # Individual datasets
    positions_data = X_training.iloc[:, :60]  # first 60 columns
    cosine_angles_data = X_training.iloc[:, 60:120]  # next 60 columns
    mean_positions_data = X_training.iloc[:, 120:180]  # next 60 columns
    std_positions_data = X_training.iloc[:, 180:240]  # next 60 columns

    # Create plotbox graphs
    plot_boxplot(X_test, "Entire test-final.csv dataset")
    plot_boxplot(X_training, "Entire train-final.csv dataset")

    # Create histograms over labels to get an idea how many samples we got
    plot_histogram(y, "Histogram over labels in train-final.csv")
    plot_histogram(y_test, "Histogram over labels in test-final.csv")

    # Normalize the dataset
    scaler = StandardScaler()
    X_normalized_training = scaler.fit_transform(X_training)
    X_normalized_testing = scaler.fit_transform(X_test)

    # Make it into a dataframe again
    X_normalized_training_df = pd.DataFrame(X_normalized_training, columns=X_training.columns)
    X_normalized_testing_df = pd.DataFrame(X_normalized_testing, columns=X_training.columns)

    # Use plotbox to visualize
    plot_boxplot(X_normalized_training_df, "Normalized training data ")
    plot_boxplot(X_normalized_testing_df, "Normalized testing data ")

    ################
    # Pipeline end
    ################ 

