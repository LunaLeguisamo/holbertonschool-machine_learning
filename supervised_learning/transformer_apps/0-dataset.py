#!/usr/bin/env python3
"""
Module: 0-dataset.py
Creates and preprocesses dataset for machine translation
"""

import tensorflow_datasets as tfds
from transformers import AutoTokenizer


class Dataset:
    """Loads and preps a dataset for machine translation"""

    def __init__(self):
        """Initialize dataset and tokenizers"""

        # 1. Load the dataset
        # ted_hrlr_translate/pt_to_en is a Portuguese to English translation
        # dataset
        # as_supervised=True returns (input, label) tuples - here
        # (portuguese, english)
        dataset, info = tfds.load('ted_hrlr_translate/pt_to_en',
                                  with_info=True, as_supervised=True)

        # 2. Create instance attributes
        self.data_train = dataset['train']    # Training split
        self.data_valid = dataset['validation']  # Validation split

        # 3. Create tokenizers using the training data
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
            )

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for the dataset

        Args:
            data: tf.data.Dataset with (portuguese, english) tuples

        Returns:
            tokenizer_pt: Portuguese tokenizer
            tokenizer_en: English tokenizer
        """

        # 1. Load pre-trained tokenizers
        # neuralmind/bert-base-portuguese-cased: BERT trained on Portuguese
        # bert-base-uncased: BERT trained on English (lowercase)
        tokenizer_pt = AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased")
        tokenizer_en = AutoTokenizer.from_pretrained(
            "bert-base-uncased")

        return tokenizer_pt, tokenizer_en
