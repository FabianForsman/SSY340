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


class TextTransform:
    """
    Callable transform class for text preprocessing.
    Can be used with PyTorch Dataset's transform parameter.
    """

    def __init__(
        self,
        remove_stopwords=False,
        lowercase=True,
        remove_urls=True,
        remove_mentions=True,
        remove_hashtags=False,
        remove_numbers=False,
        remove_quotes=False,
    ):
        """
        Initialize TextTransform.

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
        self.remove_quotes = remove_quotes

        # Load stopwords if needed
        if self.remove_stopwords:
            download_nltk_data()
            from nltk.corpus import stopwords

            self.stopwords = set(stopwords.words("english"))
        else:
            self.stopwords = set()

    def __call__(self, text: str) -> str:
        """
        Transform a single text string.
        This method is called when the object is used as a function.

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
            text = re.sub(r"&#\d+;?", "", text)
            text = re.sub(r"\d+", "", text)

        # Remove quotes
        if self.remove_quotes:
            text = text.replace('"', "").replace("'", "")

        # Remove "&amp and &gt etc."
        text = re.sub(r"&\w+;?", "", text)

        # Remove "&;"
        text = text.replace("&;", "")

        # Remove "rt :"
        text = text.replace("rt :", "").replace("rt", "")

        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Remove extra whitespace
        text = " ".join(text.split())

        # Remove, if start of text, exclamation marks or colon:
        text = re.sub(r"^[!,:]+", "", text)

        # Remove stopwords
        if self.remove_stopwords:
            words = text.split()
            text = " ".join([word for word in words if word not in self.stopwords])

        return text.strip() # Final strip to remove leading/trailing spaces



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


def create_transform(
    remove_stopwords=False,
    lowercase=True,
    remove_urls=True,
    remove_mentions=True,
    remove_hashtags=True,
    remove_numbers=True,
    remove_quotes=True,
) -> TextTransform:
    """
    Factory function to create a text transform.

    Args:
        remove_stopwords (bool): Whether to remove stopwords
        lowercase (bool): Whether to convert text to lowercase
        remove_urls (bool): Whether to remove URLs
        remove_mentions (bool): Whether to remove @mentions
        remove_hashtags (bool): Whether to remove hashtags
        remove_numbers (bool): Whether to remove numbers

    Returns:
        TextTransform: Configured text transform
    """
    return TextTransform(
        remove_stopwords=remove_stopwords,
        lowercase=lowercase,
        remove_urls=remove_urls,
        remove_mentions=remove_mentions,
        remove_hashtags=remove_hashtags,
        remove_numbers=remove_numbers,
        remove_quotes=remove_quotes,
    )

def _load_config(config_path="config.yaml"):
    import yaml
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    # Example usage
    sample_tweets = [
        "RT @user: Check out this link http://example.com #hate #speech",
        "@someone This is an offensive tweet with numbers 123",
        "THIS IS NOT HATE, JUST A QUOTE: 'Be kind!' &amp; &gt;",
        "dkeal39q !!! ### $$$ bruh what the fuck???",
    ]
    config = _load_config()

    print("\n\n=== Example: Using TextTransform with Dataset ===")
    transform = create_transform(**config["preprocessing"])
    
    print(f"Transform on sample:")
    for tweet in sample_tweets:
        print(f"Original: '{tweet}'")
        print(f"Transformed: '{transform(tweet)}'")
        print("---")
    
