"""
Text preprocessing module for hate speech detection project.
Handles cleaning and preprocessing of tweet text data.
"""

import re
import string
import pandas as pd
import nltk
from typing import List, Union


# Download required NLTK data (run once)
def download_nltk_data():
    """Download required NLTK datasets."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")


class TextPreprocessor:
    """Class to handle text preprocessing operations."""

    def __init__(
        self,
        remove_stopwords=False,
        lowercase=True,
        remove_urls=True,
        remove_mentions=True,
        remove_hashtags=False,
        remove_numbers=False,
    ):
        """
        Initialize TextPreprocessor.

        Args:
            remove_stopwords (bool): Whether to remove stopwords
            lowercase (bool): Whether to convert text to lowercase
            remove_urls (bool): Whether to remove URLs
            remove_mentions (bool): Whether to remove @mentions
            remove_hashtags (bool): Whether to remove hashtags
            remove_numbers (bool): Whether to remove numbers
        """
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.remove_numbers = remove_numbers

        # Load stopwords if needed
        if self.remove_stopwords:
            download_nltk_data()
            from nltk.corpus import stopwords

            self.stopwords = set(stopwords.words("english"))
        else:
            self.stopwords = set()

    def clean_text(self, text: str) -> str:
        """
        Clean a single text string.

        Args:
            text (str): Input text

        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        if self.lowercase:
            text = text.lower()

        # Remove URLs
        if self.remove_urls:
            text = re.sub(r"http\S+|www.\S+", "", text)

        # Remove mentions
        if self.remove_mentions:
            text = re.sub(r"@\w+", "", text)

        # Remove hashtags (keep the word, remove #)
        if self.remove_hashtags:
            text = re.sub(r"#", "", text)

        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r"\d+", "", text)

        # Remove extra whitespace
        text = " ".join(text.split())

        # Remove stopwords
        if self.remove_stopwords:
            words = text.split()
            text = " ".join([word for word in words if word not in self.stopwords])

        return text.strip()

    def clean_texts(self, texts: Union[List[str], pd.Series]) -> List[str]:
        """
        Clean multiple texts.

        Args:
            texts: List or Series of texts

        Returns:
            List[str]: Cleaned texts
        """
        if isinstance(texts, pd.Series):
            texts = texts.tolist()

        return [self.clean_text(text) for text in texts]

    def preprocess_dataframe(
        self, df: pd.DataFrame, text_column: str, output_column: str = "cleaned_text"
    ) -> pd.DataFrame:
        """
        Preprocess text data in a DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame
            text_column (str): Name of column containing text
            output_column (str): Name for output column with cleaned text

        Returns:
            pd.DataFrame: DataFrame with cleaned text column
        """
        df = df.copy()
        df[output_column] = self.clean_texts(df[text_column])

        # Remove empty texts
        df = df[df[output_column].str.len() > 0].reset_index(drop=True)

        print(f"Preprocessed {len(df)} texts")
        print(
            f"Average text length: {df[output_column].str.len().mean():.1f} characters"
        )

        return df


def basic_clean(text: str) -> str:
    """
    Basic text cleaning function.

    Args:
        text (str): Input text

    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""

    # Remove URLs
    text = re.sub(r"http\S+|www.\S+", "", text)

    # Remove mentions and hashtags
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)

    # Remove extra whitespace
    text = " ".join(text.split())

    return text.strip()


def get_text_stats(texts: Union[List[str], pd.Series]) -> dict:
    """
    Get statistics about text data.

    Args:
        texts: List or Series of texts

    Returns:
        dict: Statistics dictionary
    """
    if isinstance(texts, pd.Series):
        texts = texts.tolist()

    lengths = [len(text) for text in texts]
    word_counts = [len(text.split()) for text in texts]

    stats = {
        "total_texts": len(texts),
        "avg_char_length": sum(lengths) / len(lengths) if lengths else 0,
        "max_char_length": max(lengths) if lengths else 0,
        "min_char_length": min(lengths) if lengths else 0,
        "avg_word_count": sum(word_counts) / len(word_counts) if word_counts else 0,
        "max_word_count": max(word_counts) if word_counts else 0,
        "min_word_count": min(word_counts) if word_counts else 0,
    }

    return stats


if __name__ == "__main__":
    # Example usage
    sample_tweets = [
        "RT @user: Check out this link http://example.com #hate #speech",
        "@someone This is an offensive tweet with numbers 123",
        "Normal tweet without any special content",
    ]

    print("=== Example Preprocessing ===")
    preprocessor = TextPreprocessor(
        remove_stopwords=False,
        lowercase=True,
        remove_urls=True,
        remove_mentions=True,
        remove_hashtags=True,
    )

    cleaned = preprocessor.clean_texts(sample_tweets)

    for original, clean in zip(sample_tweets, cleaned):
        print(f"\nOriginal: {original}")
        print(f"Cleaned:  {clean}")
