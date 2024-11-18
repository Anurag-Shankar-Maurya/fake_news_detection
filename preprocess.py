import pandas as pd
from sklearn.model_selection import train_test_split


def load_data():
    """
    Load datasets from CSV files and preprocess by adding labels and combining text columns.
    """
    # fake_data = pd.read_csv('data/Fake.csv')  # Fake news dataset
    # real_data = pd.read_csv('data/True.csv')  # Real news dataset
    fake_data = pd.read_csv('data/Fake.csv', low_memory=False)  # Fake news dataset
    real_data = pd.read_csv('data/True.csv', low_memory=False)  # Real news dataset

    # Label the data: 0 for Fake, 1 for Real
    fake_data['label'] = 0
    real_data['label'] = 1

    # Combine datasets
    data = pd.concat([fake_data[['title', 'text', 'label']],
                      real_data[['title', 'text', 'label']]])
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle the data

    # Combine 'title' and 'text' into a single column for analysis
    data['combined'] = data['title'].fillna('') + " " + data['text'].fillna('')
    return data


def preprocess_data(data):
    """
    Split the dataset into training and testing sets.
    """
    X = data['combined'].values  # Features: Combined text
    y = data['label'].values  # Labels: 0 (Fake), 1 (Real)

    # Stratified split to ensure balanced classes in train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test
