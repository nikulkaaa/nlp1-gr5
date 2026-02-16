from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pandas as pd
import numpy as np

def clean_text(text: str) -> str:
    """
    Normalizes the input text by removing s[ecial characters, punctuation and converting to lowercase.
    
    :param text: The input text to normalize.
    :return: The normalized text.
    """
    # Convert to lowercase
    text = text.lower()

    # Remove special characters and punctuation
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

def preprocess_data(df):
    """
    Preprocesses the input DataFrame by tokenizing the 'description' column and creating a new 'tokens' column.
    
    :param df: The input DataFrame with a 'description' column.
    :return: A new DataFrame with an additional 'tokens' column.
    """
    # Clean the 'description' column
    df['description'] = df['description'].apply(clean_text)
    return df

def feature_engineering_tfidf(
    dataset: pd.DataFrame,
    column_name: str,
    max_features: int = 1000
) -> np.ndarray:
    """
    Converts text data from a dataset column into numerical representations.

    :param dataset: The input DataFrame containing the text data.
    :param column_name: The name of the column containing the text data.
    :param max_features: The maximum number of features to extract using TF-IDF.
    :return: A NumPy array containing the TF-IDF features.
    """
    text_data = dataset[column_name].apply(
        lambda x: " ".join(x) if isinstance(x, list) else x
    ).values

    # Use TF-IDF to transform the entire column
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
    X = tfidf_vectorizer.fit_transform(text_data).toarray()
    return X

