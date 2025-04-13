import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import Counter
import random
import zipfile
import tarfile

def simple_tokenize(text):
    """
    A simple tokenizer to replace spaCy tokenizer
    """
    # Lower case the text
    text = text.lower()
    # Replace non-alphanumeric chars with space
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    # Split by whitespace and filter out empty tokens
    tokens = [token for token in text.split() if token]
    return tokens

class Vocabulary:
    def __init__(self, min_freq=5):
        self.min_freq = min_freq
        self.word2idx = {"PAD": 0, "UNK": 1}
        self.idx2word = {0: "PAD", 1: "UNK"}
        self.word_counts = Counter()
        self.n_words = 2  # Count PAD and UNK

    def add_sentence(self, sentence):
        for word in sentence:
            self.word_counts[word] += 1

    def build_vocab(self):
        # Add words that appear at least min_freq times
        for word, count in self.word_counts.items():
            if count >= self.min_freq:
                self.word2idx[word] = self.n_words
                self.idx2word[self.n_words] = word
                self.n_words += 1
        print(f"Vocabulary size: {self.n_words} (including PAD and UNK tokens)")

    def get_index(self, word):
        if word in self.word2idx:
            return self.word2idx[word]
        return self.word2idx["UNK"]

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Create token indices (instead of one-hot encoding)
        tokens = text[:self.max_length] if len(text) > self.max_length else text + ["PAD"] * (self.max_length - len(text))
        indices = torch.tensor([self.vocab.get_index(token) for token in tokens], dtype=torch.long)
        
        return indices, torch.tensor(label, dtype=torch.long)

def load_imdb_dataset(data_dir):
    """
    Load IMDB dataset
    """
    texts = []
    labels = []
    
    # Load positive reviews
    for filename in os.listdir(os.path.join(data_dir, 'pos')):
        if filename.endswith('.txt'):
            with open(os.path.join(data_dir, 'pos', filename), 'r', encoding='utf-8') as f:
                text = f.read()
                tokens = simple_tokenize(text)
                texts.append(tokens)
                labels.append(1)  # 1 for positive
    
    # Load negative reviews
    for filename in os.listdir(os.path.join(data_dir, 'neg')):
        if filename.endswith('.txt'):
            with open(os.path.join(data_dir, 'neg', filename), 'r', encoding='utf-8') as f:
                text = f.read()
                tokens = simple_tokenize(text)
                texts.append(tokens)
                labels.append(0)  # 0 for negative
    
    # Shuffle the data
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    
    return list(texts), list(labels)

def load_semeval_dataset(data_file):
    """
    Load SemEval dataset
    """
    texts = []
    labels = []
    
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    
    with open(data_file, 'r', encoding='utf-8') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                text_id, text, sentiment = parts[0], parts[2], parts[3]
                tokens = simple_tokenize(text)
                texts.append(tokens)
                labels.append(label_map.get(sentiment, 1))  # Default to neutral if sentiment is unknown
    
    return texts, labels

def calculate_average_length(texts):
    """
    Calculate average length of texts in the corpus
    """
    total_words = sum(len(text) for text in texts)
    avg_length = round(total_words / len(texts))
    return avg_length

def prepare_dataloaders(train_texts, train_labels, test_texts, test_labels, vocab, batch_size, max_length):
    """
    Prepare dataloaders for train and test sets
    """
    train_dataset = SentimentDataset(train_texts, train_labels, vocab, max_length)
    test_dataset = SentimentDataset(test_texts, test_labels, vocab, max_length)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, test_dataloader 