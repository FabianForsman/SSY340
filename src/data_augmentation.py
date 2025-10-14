from nltk.corpus import wordnet
import random
import pandas as pd

def get_synonyms(word):
    """Get synonyms for a given word using WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ').lower()
            if synonym != word.lower():
                synonyms.add(synonym)
    return list(synonyms)

def data_agumentation(text, num_aug):
    """
    Perform data augmentation on the input text using synonym replacement.

    Args:
        text (str): The original text to augment.
        num_aug (int): Number of augmented sentences to generate.

    Returns:
        List[str]: List of augmented sentences.
    """
    words = text.split()
    augmented_sentences = []

    for _ in range(num_aug):
        new_words = list(words)

        # Randomly choose a word index to replace with a synonym
        idx = random.randint(0, len(words) - 1)
        random_word = words[idx]

        # Get synonyms
        synonyms = get_synonyms(random_word)
        if synonyms:
            # Replace with a random synonym
            synonym = random.choice(synonyms)
            new_words[idx] = synonym

        augmented_sentences.append(' '.join(new_words))

    return augmented_sentences


def augment_data_to_target_count(df_class, target_count):
    """
    Augment data in the DataFrame to reach the target count for each class.

    Args:
        df_class (pd.DataFrame): DataFrame containing samples of a specific class.
        target_count (int): Desired number of samples for the class.
    Returns:
        pd.DataFrame: DataFrame with augmented samples added.
    """

    if len(df_class) >= target_count:
        return df_class
    
    current = len(df_class) 
    augmented_rows = []

    samples_to_add = target_count - current

    samples_to_augment = df_class.sample(n=samples_to_add, replace=True)

    for _, row in samples_to_augment.iterrows():
        text = row['tweet']
        augmented_text = data_agumentation(text, num_aug=1)  # Generate one augmented text per sample
        augmented_rows.append({
            'tweet': augmented_text[0], 
            'label': row['label'], 
            'label_desc': row['label_desc']
        })

    df_augmented = pd.DataFrame(augmented_rows)
    df_balanced = pd.concat([df_class, df_augmented], ignore_index=True)
    return df_balanced