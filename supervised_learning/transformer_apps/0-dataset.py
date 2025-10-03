#!/usr/bin/env python3
"""
0-dataset.py
"""
import tensorflow_datasets as tfds
import transformers


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
