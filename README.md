# Fake News Detection System 📊📰

## Description 🌟

This project focuses on detecting fake news articles by utilizing machine learning techniques. The system uses a dataset containing labeled news articles (fake or real) and applies text preprocessing, feature extraction, model training, and prediction to classify new articles as fake or real.

---

## Features ⚡

- **Data Preprocessing**: Cleans and prepares data for model training.
- **Text Vectorization**: Transforms text data into numerical features using TF-IDF.
- **Logistic Regression Model**: Trains a classification model to detect fake news.
- **News Prediction**: Allows users to input a news article and predict if it's fake or real.
- **Model Evaluation**: Evaluates the model's performance using various metrics like precision, recall, and F1 score.

---

## Project Structure 📁

```plaintext
fake_news_detection/
├── .venv/                 # Virtual environment
├── data/                  # Contains the datasets
│   ├── Fake.csv           # Fake news dataset
│   ├── Fake-Test.csv      # Fake news test dataset
│   ├── True.csv           # Real news dataset
│   └── True-Test.csv      # Real news test dataset
├── main.py                # Main script to run the pipeline
├── model.py               # Model building and training code
├── predict.py             # Prediction logic for new articles
├── preprocess.py          # Data preprocessing functions
├── README.md              # Project documentation
└── requirements.txt       # Dependencies for the project
```

---

## How It Works 🔧

1. **Load and Preprocess Data**: The datasets are loaded, combined, and labeled as fake or real. The text columns are then combined for analysis.
2. **Text Vectorization**: The combined text data is transformed into numerical features using the TF-IDF vectorizer.
3. **Model Training**: A logistic regression model is trained using the vectorized data.
4. **Model Evaluation**: The model's performance is evaluated using the test dataset.
5. **Prediction**: Users can input news articles, and the system predicts whether they are fake or real, with confidence scores.

---

## Installation ⚙️

### Clone the Repository 🚀

```bash
git clone https://github.com/Anurag-Shankar-Maurya/fake_news_detection
cd fake_news_detection
```

### Set Up Environment 🖥️

1. **Create a Virtual Environment**:
   ```bash
   python -m venv .venv
   ```

2. **Activate the Environment**:
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - On Mac/Linux:
     ```bash
     source .venv/bin/activate
     ```

### Install Dependencies 📦

```bash
pip install -r requirements.txt
```

### Add Dataset 📁

Place your datasets (`Fake.csv`, `True.csv`, etc.) in the `data/` directory.

---

## Usage 🚀

Run the following command to start the news prediction system:

```bash
python main.py
```

You will be prompted to enter a news article, and the system will predict whether it's fake or real. Type `exit` to quit.

---

## Each File Details 📄

- **`main.py`**: This is the entry point of the project. It loads the data, preprocesses it, trains the model, evaluates it, and allows the user to input a news article for prediction.
- **`model.py`**: Contains functions to vectorize the text data using TF-IDF and build a logistic regression model.
- **`predict.py`**: Handles the prediction logic, including calculating confidence scores for fake or real news.
- **`preprocess.py`**: Defines functions to load and preprocess data, including splitting it into training and test sets.
- **`requirements.txt`**: Lists the dependencies required to run the project.

---

## Example 📝

### Input

```
Enter news article: The moon landing was staged by NASA in a studio.
```

### Output

```
Loading data...
Preprocessing data...
Vectorizing data...
Training the model...
Evaluating the model...
Model Evaluation:
              precision    recall  f1-score   support

        Fake       0.99      0.98      0.99      4661
        Real       0.98      0.99      0.99      4243

    accuracy                           0.99      8904
   macro avg       0.99      0.99      0.99      8904
weighted avg       0.99      0.99      0.99      8904


Fake News Detection System
Type a news article to classify or 'exit' to quit.

Enter news article: The moon landing was staged by NASA in a studio.
The news is classified as: Fake (73.78% confidence)
Enter news article: 
```

---

## Dependencies 🛠️

- `pandas` 🐼
- `scikit-learn` 🔧
- `numpy` 🔢

---

## Author ✍️

**Anurag Shankar Maurya**

- GitHub: [@Anurag-Shankar-Maurya](https://github.com/Anurag-Shankar-Maurya)
- Email: anuragshankarmaurya@gmail.com

---

Thank you for reading. Suggestions are welcome via email.
