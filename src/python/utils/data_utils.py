"""
Data utilities for Nexora AI

This module contains functions for loading, preprocessing, and managing data
for training and inference with the Nexora model.
"""

import os
import json
import yaml
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pickle


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dict containing configuration parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_json(data: Dict, file_path: str) -> None:
    """
    Save data as JSON
    
    Args:
        data: Data to save
        file_path: Path where to save the JSON file
    """
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(file_path: str) -> Dict:
    """
    Load data from JSON
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dict containing loaded data
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def save_pickle(data: object, file_path: str) -> None:
    """
    Save data as pickle file
    
    Args:
        data: Data to save
        file_path: Path where to save the pickle file
    """
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file_path: str) -> object:
    """
    Load data from pickle file
    
    Args:
        file_path: Path to the pickle file
        
    Returns:
        Object containing loaded data
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def create_directory(directory_path: str) -> None:
    """
    Create directory if it doesn't exist
    
    Args:
        directory_path: Path to the directory
    """
    os.makedirs(directory_path, exist_ok=True)


class SimpleTokenizer:
    """
    A basic tokenizer for text processing
    """
    
    def __init__(self, vocab_size: int = 30000):
        self.vocab_size = vocab_size
        self.word_to_id = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        self.word_counts = {}
        
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from texts
        
        Args:
            texts: List of texts to build vocabulary from
        """
        # Count words
        for text in texts:
            for word in text.lower().split():
                if word not in self.word_counts:
                    self.word_counts[word] = 0
                self.word_counts[word] += 1
                
        # Sort by frequency
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Add most frequent words to vocabulary (up to vocab_size)
        for word, _ in sorted_words[:self.vocab_size - len(self.word_to_id)]:
            idx = len(self.word_to_id)
            self.word_to_id[word] = idx
            self.id_to_word[idx] = word
            
    def tokenize(self, text: str) -> List[int]:
        """
        Convert text to token IDs
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of token IDs
        """
        tokens = ["[CLS]"] + text.lower().split() + ["[SEP]"]
        token_ids = [self.word_to_id.get(token, self.word_to_id["[UNK]"]) for token in tokens]
        return token_ids
        
    def encode(self, text: str, max_length: int = None) -> Dict[str, torch.Tensor]:
        """
        Encode text as token IDs with attention mask
        
        Args:
            text: Text to encode
            max_length: Maximum sequence length (will pad or truncate)
            
        Returns:
            Dict with input_ids and attention_mask tensors
        """
        token_ids = self.tokenize(text)
        
        if max_length is not None:
            # Truncate if too long
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            
            # Pad if too short
            attention_mask = [1] * len(token_ids)
            padding_length = max_length - len(token_ids)
            
            if padding_length > 0:
                token_ids = token_ids + [self.word_to_id["[PAD]"]] * padding_length
                attention_mask = attention_mask + [0] * padding_length
        else:
            attention_mask = [1] * len(token_ids)
            
        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
        }
        
    def batch_encode(self, texts: List[str], max_length: int = None) -> Dict[str, torch.Tensor]:
        """
        Encode a batch of texts
        
        Args:
            texts: List of texts to encode
            max_length: Maximum sequence length
            
        Returns:
            Dict with batched input_ids and attention_mask tensors
        """
        # If max_length not specified, use length of longest text + 2 (for [CLS] and [SEP])
        if max_length is None:
            max_length = max(len(text.split()) for text in texts) + 2
            
        all_input_ids = []
        all_attention_masks = []
        
        for text in texts:
            encoded = self.encode(text, max_length)
            all_input_ids.append(encoded["input_ids"])
            all_attention_masks.append(encoded["attention_mask"])
            
        return {
            "input_ids": torch.stack(all_input_ids),
            "attention_mask": torch.stack(all_attention_masks)
        }
        
    def decode(self, token_ids: List[int]) -> str:
        """
        Convert token IDs back to text
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text
        """
        tokens = [self.id_to_word.get(idx, "[UNK]") for idx in token_ids]
        
        # Remove special tokens
        tokens = [token for token in tokens if token not in ["[PAD]", "[CLS]", "[SEP]"]]
        
        return " ".join(tokens)
        
    def save(self, path: str) -> None:
        """
        Save tokenizer to disk
        
        Args:
            path: Path where to save the tokenizer
        """
        create_directory(os.path.dirname(path))
        data = {
            "vocab_size": self.vocab_size,
            "word_to_id": self.word_to_id,
            "id_to_word": self.id_to_word,
            "word_counts": self.word_counts
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
            
    @classmethod
    def load(cls, path: str) -> 'SimpleTokenizer':
        """
        Load tokenizer from disk
        
        Args:
            path: Path to the saved tokenizer
            
        Returns:
            Loaded tokenizer instance
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
            
        tokenizer = cls(data["vocab_size"])
        tokenizer.word_to_id = data["word_to_id"]
        tokenizer.id_to_word = data["id_to_word"]
        tokenizer.word_counts = data["word_counts"]
        
        return tokenizer


class TextDataset(Dataset):
    """
    Dataset for text data
    """
    
    def __init__(self, texts: List[str], labels: Optional[List[int]] = None, 
                 tokenizer=None, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize text
        encoded = self.tokenizer.encode(text, self.max_length)
        
        # If we have labels, return them as well
        if self.labels is not None:
            label = self.labels[idx]
            return {**encoded, "labels": torch.tensor(label, dtype=torch.long)}
        
        return encoded


def prepare_dataloaders(config: Dict, tokenizer, texts: List[str], labels: Optional[List[int]] = None,
                       test_size: float = 0.2, valid_size: float = 0.1) -> Dict[str, DataLoader]:
    """
    Prepare DataLoaders for training, validation, and testing
    
    Args:
        config: Configuration dict
        tokenizer: Tokenizer to use for encoding texts
        texts: List of texts
        labels: Optional list of labels (for supervised learning)
        test_size: Proportion of data to use for testing
        valid_size: Proportion of training data to use for validation
        
    Returns:
        Dict containing train, validation, and test DataLoaders
    """
    # Split into train and test
    train_texts, test_texts = train_test_split(texts, test_size=test_size, random_state=42)
    
    if labels is not None:
        train_labels, test_labels = train_test_split(labels, test_size=test_size, random_state=42)
    else:
        train_labels, test_labels = None, None
    
    # Split train into train and validation
    if valid_size > 0:
        valid_size_adjusted = valid_size / (1 - test_size)  # Adjust for previous split
        train_texts, valid_texts = train_test_split(train_texts, test_size=valid_size_adjusted, random_state=42)
        
        if train_labels is not None:
            train_labels, valid_labels = train_test_split(train_labels, test_size=valid_size_adjusted, random_state=42)
        else:
            valid_labels = None
    else:
        valid_texts, valid_labels = [], []
    
    # Create datasets
    train_dataset = TextDataset(
        train_texts, train_labels, tokenizer, 
        max_length=config["model"]["max_sequence_length"]
    )
    
    valid_dataset = TextDataset(
        valid_texts, valid_labels, tokenizer, 
        max_length=config["model"]["max_sequence_length"]
    ) if valid_texts else None
    
    test_dataset = TextDataset(
        test_texts, test_labels, tokenizer, 
        max_length=config["model"]["max_sequence_length"]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=config["training"]["batch_size"],
        shuffle=False
    ) if valid_dataset else None
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config["inference"]["batch_size"],
        shuffle=False
    )
    
    return {
        "train": train_loader,
        "valid": valid_loader,
        "test": test_loader
    }


def load_csv_data(file_path: str, text_column: str, label_column: Optional[str] = None) -> Tuple[List[str], Optional[List[int]]]:
    """
    Load text and labels from CSV file
    
    Args:
        file_path: Path to the CSV file
        text_column: Name of the column containing text data
        label_column: Optional name of the column containing labels
        
    Returns:
        Tuple of (texts, labels) where labels is None if label_column is None
    """
    df = pd.read_csv(file_path)
    
    texts = df[text_column].tolist()
    
    if label_column is not None and label_column in df.columns:
        labels = df[label_column].tolist()
        return texts, labels
    
    return texts, None


def load_json_data(file_path: str, text_key: str, label_key: Optional[str] = None) -> Tuple[List[str], Optional[List[int]]]:
    """
    Load text and labels from JSON file
    
    Args:
        file_path: Path to the JSON file
        text_key: Key for text data in JSON objects
        label_key: Optional key for labels in JSON objects
        
    Returns:
        Tuple of (texts, labels) where labels is None if label_key is None
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if isinstance(data, list):
        # List of objects
        texts = [item[text_key] for item in data if text_key in item]
        
        if label_key is not None:
            labels = [item.get(label_key) for item in data if text_key in item]
            return texts, labels
            
        return texts, None
    elif isinstance(data, dict) and text_key in data:
        # Single object
        return [data[text_key]], [data.get(label_key)] if label_key else None
    else:
        raise ValueError(f"Unsupported JSON format or missing key: {text_key}")
