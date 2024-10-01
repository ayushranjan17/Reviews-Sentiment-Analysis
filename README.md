# Sentiment Analysis with BiLSTM

## Project Overview

This project implements a **Sentiment Analysis** system using a **Bidirectional LSTM (BiLSTM)** model. The goal is to classify reviews into positive or negative sentiments. The model is trained on a dataset of reviews, where each review is labeled with its corresponding sentiment. 

## Dataset

Dataset contains sentences labelled with positive or negative sentiment, extracted from reviews of products, movies, and restaurants.
The dataset used for training and evaluation is stored in a file named `sentiment.txt`, where each line contains a review followed by its sentiment, separated by a tab (`\t`). The sentiments are represented as binary labels:
- `1` for positive sentiment
- `0` for negative sentiment
  
The sentences come from three different websites/fields:

*imdb.com
*amazon.com
*yelp.com
*For each website, there exist 500 positive and 500 negative sentences. Those were selected randomly for larger datasets of reviews. 
*We attempted to select sentences that have a clearly positive or negative connotaton, the goal was for no neutral sentences to be selected.

## Key Features
- **Data Preprocessing**: Reviews are tokenized, cleaned, and padded to a fixed length.
- **BiLSTM Model**: Utilizes a bidirectional LSTM for better context understanding.
- **Embedding Layer**: Maps words to dense vector representations.
- **Model Evaluation**: The model is evaluated using accuracy and loss metrics.

## Model Architecture

The model architecture consists of the following layers:
1. **Embedding Layer**: Converts input tokens into dense vector representations.
2. **Bidirectional LSTM**: Captures dependencies in both directions (forward and backward).
3. **Dropout Layer**: Reduces overfitting during training.
4. **Dense Layer**: Produces output with sigmoid activation for binary classification.

### Functions
- `load_data()`: Loads and preprocesses data from the `sentiment.txt` file.
- `split_words_reviews(data)`: Tokenizes the reviews and creates a vocabulary.
- `pad_text(tokenized_reviews, seq_length)`: Pads tokenized reviews to a fixed length.
- `preprocess_review(review)`: Prepares a review for prediction.
- `predict_review(review)`: Predicts the sentiment of a given review.
