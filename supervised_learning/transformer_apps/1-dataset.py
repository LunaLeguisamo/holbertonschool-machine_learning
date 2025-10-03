#!/usr/bin/env python3
"""
0-dataset.py
"""
import tensorflow_datasets as tfds
import transformers
import numpy as np


class Dataset:
    """Loads and preps a dataset for machine translation."""
    def __init__(self):
        dataset, _ = tfds.load('ted_hrlr_translate/pt_to_en',
                               as_supervised=True,
                               with_info=True)

        self.data_train = dataset['train']
        self.data_valid = dataset['validation']
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
            )

    def tokenize_dataset(self, data):
        """Creates sub-word tokenizers for Portuguese and English."""
        # Extraer oraciones
        pt_texts, en_texts = [], []
        for pt, en in data:
            pt_texts.append(pt.numpy().decode('utf-8'))
            en_texts.append(en.numpy().decode('utf-8'))

        # Cargar y entrenar tokenizers
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased"
            )
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            "bert-base-uncased"
            )

        # Entrenar con tamaño de vocabulario 8192
        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_texts, vocab_size=2**13
            )
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_texts, vocab_size=2**13
            )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encodes a translation into tokens with start and end tokens
        """
        # Decodificar
        pt_text = pt.numpy().decode('utf-8')
        en_text = en.numpy().decode('utf-8')

        # Tokenizar
        pt_tokens = self.tokenizer_pt.encode(pt_text)
        en_tokens = self.tokenizer_en.encode(en_text)

        # Vocab sizes
        vocab_pt = self.tokenizer_pt.vocab_size
        vocab_en = self.tokenizer_en.vocab_size

        # Agregar start/end tokens directamente con numpy
        pt_tokens = np.array([vocab_pt] + pt_tokens + [vocab_pt + 1])
        en_tokens = np.array([vocab_en] + en_tokens + [vocab_en + 1])

        return pt_tokens, en_tokens
