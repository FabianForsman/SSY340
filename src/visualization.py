# Visualizes the class distribution in the dataset

import pandas as pd
import matplotlib.pyplot as plt
from data_loader import HateSpeechDataset

def visualize_class_dist(dataset: HateSpeechDataset):
    # Visualize class distribution
    plt.figure(figsize=(8, 6))
    dataset.df['label_desc'].value_counts().plot(kind='bar', color=['skyblue', 'salmon', 'lightgreen'])
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=0)
    HateSpeechDataset.get_dataset_info(dataset)
    plt.show()