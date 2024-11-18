def predict_news(news_text, vectorizer, model):
    """
    Predict whether the input news article is Fake or Real, with confidence scores.
    """
    # Transform the user input text using the TF-IDF vectorizer
    news_vec = vectorizer.transform([news_text])

    # Predict the probabilities for Fake and Real classes
    prediction_prob = model.predict_proba(news_vec)[0]

    # Predict the class (0 for Fake, 1 for Real)
    prediction = model.predict(news_vec)[0]

    # Compute confidence percentages
    fake_percentage = prediction_prob[0] * 100
    real_percentage = prediction_prob[1] * 100

    # Return the prediction result
    if prediction == 1:
        return f"Real ({real_percentage:.2f}% confidence)"
    else:
        return f"Fake ({fake_percentage:.2f}% confidence)"
