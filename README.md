# Sentiment_Analysis_with_NLP

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: MADDINENI ROHITHA

*INTERN ID*: CT06DL736

*DOMAIN*: MACHINE LEARNING

*DURATION*: 6 WEEEKS

*MENTOR*: NEELA SANTOSH

## Project Overview:
This project focuses on Sentiment Analysis using Natural Language Processing (NLP) techniques, specifically on a dataset of customer (or movie) reviews. The objective is to classify the sentiment of each review as either positive or negative based on its textual content. This is a classic binary classification problem in NLP and is widely applicable in domains such as e-commerce, social media monitoring, customer feedback systems, and more.

We perform the complete pipeline of sentiment analysis which includes:
- Text preprocessing
- Feature extraction using TF-IDF
- Model training using Logistic Regression
- Performance evaluation using classification metrics and a confusion matrix

## Dataset:
The dataset used is named Movie_reviews.csv and consists of two columns:
1. Review – Contains the text of the movie/customer review.
2. Sentiment – Indicates whether the review is positive or negative.

Before any modeling, this raw text data undergoes preprocessing to clean and prepare it for machine learning.

## Code Breakdown

### 1. Data Loading:
Using the pandas library, the CSV file is read and the columns are renamed for consistency:

```python
df = pd.read_csv('Movie_reviews.csv') 
df.columns = ['review', 'sentiment']
```

### 2. Text Preprocessing:
A custom clean_text() function is applied to each review to clean and normalize the text. This includes:
- Converting text to lowercase
- Removing HTML tags
- Removing non-alphabetic characters
- Removing extra whitespace

This step is essential to reduce noise and ensure that the model captures relevant patterns in the text.

### 3. Label Encoding:
The sentiment column is mapped to numerical labels:
- positive → 1
- negative → 0

This is required for the classification algorithm to interpret the target variable correctly.

### 4. Train-Test Split:
The dataset is split into training and testing sets using an 80-20 split. This helps in evaluating the model on unseen data and measuring its generalization ability.

### 5. TF-IDF Vectorization:
TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert the cleaned textual data into numerical vectors. The top 5000 features are extracted, and common English stopwords are removed. This technique helps quantify the importance of words in documents relative to the entire dataset.

```python
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
```

### 6. Model Training:
A LogisticRegression model from scikit-learn is trained using the TF-IDF vectors of the training data. Logistic Regression is widely used for binary classification problems and works well with high-dimensional sparse data like TF-IDF matrices.

### 7. Prediction and Evaluation:
The trained model is used to predict the sentiment labels for the test data. The performance is evaluated using:
- Accuracy Score: Overall correctness of the model.
- Classification Report: Precision, recall, and F1-score for each class.
- Confusion Matrix: A heatmap visualization of predicted vs actual labels.

## Visualization:
A seaborn heatmap of the confusion matrix is displayed to give a quick visual understanding of the model's performance in terms of true positives, false positives, true negatives, and false negatives.

## Conclusion:
This project successfully demonstrates sentiment analysis using TF-IDF vectorization and Logistic Regression. By preprocessing textual data, extracting meaningful features, and training a reliable classification model, we achieve effective sentiment prediction on customer reviews. The approach is simple, scalable, and serves as a strong foundation for more advanced NLP tasks or model enhancements in the future.
