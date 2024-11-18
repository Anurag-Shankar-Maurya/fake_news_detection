from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def vectorize_data(X_train, X_test):
    """
    Vectorize text data using TF-IDF to convert it into numerical format.
    """
    vectorizer = TfidfVectorizer(
        max_features=5000,  # Limit to 5000 most relevant words
        stop_words='english'  # Remove common English stop words
    )
    X_train_vec = vectorizer.fit_transform(X_train)  # Fit and transform training data
    X_test_vec = vectorizer.transform(X_test)  # Transform testing data
    return X_train_vec, X_test_vec, vectorizer


def build_and_train_model(X_train_vec, y_train):
    """
    Train a Logistic Regression model on the vectorized data.
    """
    model = LogisticRegression(max_iter=500, random_state=42)  # Initialize the model
    model.fit(X_train_vec, y_train)  # Train the model
    return model


def evaluate_model(model, X_test_vec, y_test):
    """
    Evaluate the model on test data and display performance metrics.
    """
    y_pred = model.predict(X_test_vec)  # Predict labels for the test data
    print("Model Evaluation:")
    print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
