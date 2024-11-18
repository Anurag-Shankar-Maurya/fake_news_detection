from preprocess import load_data, preprocess_data
from model import vectorize_data, build_and_train_model, evaluate_model
from predict import predict_news


def main():
    """
    Main function to execute the Fake News Detection pipeline.
    """
    # Step 1: Load and preprocess the data
    print("Loading data...")
    data = load_data()

    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(data)

    # Step 2: Vectorize the text data
    print("Vectorizing data...")
    X_train_vec, X_test_vec, vectorizer = vectorize_data(X_train, X_test)

    # Step 3: Train the model
    print("Training the model...")
    model = build_and_train_model(X_train_vec, y_train)

    # Step 4: Evaluate the model
    print("Evaluating the model...")
    evaluate_model(model, X_test_vec, y_test)

    # Step 5: Predict news articles
    print("\nFake News Detection System")
    print("Type a news article to classify or 'exit' to quit.\n")
    while True:
        user_input = input("Enter news article: ")
        if user_input.lower() == 'exit':
            print("Exiting the system.")
            break
        result = predict_news(user_input, vectorizer, model)
        print(f"The news is classified as: {result}")


if __name__ == '__main__':
    main()
