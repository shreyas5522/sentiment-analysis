# Sentiment Analysis of Musical Instruments Reviews

This project aims to classify the sentiment of customer reviews for musical instruments using various machine learning models. The dataset used for this project is a collection of reviews from Amazon.com.

## Dataset

The dataset used for this project can be found in `instruments_reviews.csv` file. It contains the following columns:

- `reviewerID`: unique ID of the reviewer
- `asin`: unique ID of the product
- `reviewerName`: name of the reviewer
- `helpful`: number of helpful votes
- `reviewText`: text of the review
- `overall`: rating given by the reviewer (1-5)
- `summary`: brief summary of the review
- `unixReviewTime`: time of the review (unix time)
- `reviewTime`: time of the review (raw)

## Approach

The approach for this project is as follows:

1. Data cleaning and preprocessing
2. Feature extraction using CountVectorizer
3. Splitting the data into training and testing sets
4. Training various machine learning models on the training set
5. Evaluating the performance of each model on the testing set
6. Fine-tuning the best-performing model
7. Generating predictions for new reviews using the fine-tuned model

## Data Cleaning and Preprocessing

The `reviewText` column is the main feature used for sentiment analysis. Before extracting features from this column, we need to perform the following preprocessing steps:

1. Convert all text to lowercase
2. Remove punctuation and special characters
3. Remove stopwords (common words such as "the", "and", "a", etc.)
4. Stemming (converting words to their root form)

These preprocessing steps are implemented in the `preprocess_text` function in the `sentiment_analysis.py` file.

## Feature Extraction

After preprocessing the text, we need to extract features that can be used for training machine learning models. For this project, we will use the `CountVectorizer` class from the `sklearn.feature_extraction.text` module. This class converts a collection of text documents to a matrix of token counts.

The `CountVectorizer` is fit on the training data and then used to transform both the training and testing data into feature matrices. The resulting feature matrices have one row for each review and one column for each unique word in the dataset.

## Machine Learning Models

We will train and evaluate the following machine learning models:

1. Naive Bayes
2. Support Vector Machines (SVM)
3. Logistic Regression
4. Random Forest Classifier
5. Gradient Boosting Classifier

The performance of each model will be evaluated using the accuracy score and confusion matrix.

## Results

The following table shows the accuracy score and confusion matrix for each model:

| Model | Accuracy Score | Confusion Matrix |
| ----- | -------------- | ---------------- |
| Naive Bayes | 0.887 | [[12, 228], [3, 1808]] |
| SVM | 0.858 | [[84, 156], [136, 1675]] |
| Logistic Regression | 0.877 | [[63, 177], [76, 1735]] |
| Random Forest Classifier | 0.886 | [[9, 231], [3, 1808]] |
| Gradient Boosting Classifier | 0.886 | [[12, 228], [6, 1805]] |

Based on the above results, Naive Bayes has the highest accuracy score of 0.887. We will fine-tune this model to improve its performance.

## Fine-tuning the Model

We will use the `GridSearchCV` class from the `sklearn.model_selection` module to perform hyperparameter tuning for the Naive Bayes model. The hyperparameters that we will tune are:

- `alpha`: smoothing parameter for Naive Bayes

We will perform a grid search using 5-fold cross-validation and evaluate the performance of each combination of hyperparameters using the accuracy score. The best-performing model will then be used to generate predictions for new reviews.

The hyperparameter tuning is implemented in the `fine_tune_model` function in the `sentiment_analysis.py` file.

Best Hyperparameters:  {'alpha': 50}
Accuracy Score:  0.9005176924136032


### Generating Predictions

We will use the fine-tuned Naive Bayes model to generate predictions for new reviews. The `predict_sentiment` function in the `sentiment_analysis.py` file takes a list of reviews as input and returns a list of predicted sentiment labels.

### Conclusion

In this project, we performed sentiment analysis on customer reviews for musical instruments using various machine learning models. We preprocessed the text data and extracted features using the `CountVectorizer` class. We trained and evaluated several machine learning models, and found that Naive Bayes performed the best with an accuracy score of 0.887. We fine-tuned the Naive Bayes model using hyperparameter tuning, and used it to generate predictions for new reviews.

This project can be extended in several ways, such as using more sophisticated feature extraction techniques, exploring different machine learning algorithms, and analyzing the impact of different hyperparameters on model performance.

