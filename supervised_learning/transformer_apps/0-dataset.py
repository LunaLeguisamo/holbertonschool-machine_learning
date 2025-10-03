#!/usr/bin/env python3
"""
Module: 0-dataset.py
Creates and preprocesses dataset for machine translation
"""

import tensorflow_datasets as tfds
import transformers


class Dataset:
    """Loads and preps a dataset for machine translation"""

    def __init__(self):
        """Initialize dataset and tokenizers"""
        # Load the dataset with both data and info
        dataset, info = tfds.load('ted_hrlr_translate/pt_to_en',
                                  with_info=True,
                                  as_supervised=True,
                                  shuffle_files=False))

        # Create instance attributes
        self.data_train = dataset['train']
        self.data_valid = dataset['validation']

        # Create tokenizers using the training data
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
        # Load pre-trained tokenizers
        # Portuguese tokenizer
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased")

        # English tokenizer
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            "bert-base-uncased")

        return tokenizer_pt, tokenizer_en
