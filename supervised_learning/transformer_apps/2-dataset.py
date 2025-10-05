#!/usr/bin/env python3
"""
0-dataset.py
"""
import tensorflow_datasets as tfds
import tensorflow as tf
import transformers


class Dataset:
    def __init__(self):
        dataset, _ = tfds.load('ted_hrlr_translate/pt_to_en',
                               as_supervised=True,
                               with_info=True)

        # Guardar los datasets originales
        self.raw_data_train = dataset['train']
        self.raw_data_valid = dataset['validation']

        # Crear tokenizers
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.raw_data_train
            )

        # Tokenizar todo el dataset
        self.data_train = self.raw_data_train.map(self.tf_encode)
        self.data_valid = self.raw_data_valid.map(self.tf_encode)

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
            pt_texts,
            vocab_size=2**13
            )
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_texts,
            vocab_size=2**13
            )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encodes a translation into tokens with start and end tokens
        """
        # Decodificar
        pt_text = pt.numpy().decode('utf-8')
        en_text = en.numpy().decode('utf-8')

        # Tokenizar - esto retorna listas de Python
        pt_tokens = self.tokenizer_pt.encode(pt_text)
        en_tokens = self.tokenizer_en.encode(en_text)

        # Vocab sizes
        vocab_pt = self.tokenizer_pt.vocab_size
        vocab_en = self.tokenizer_en.vocab_size

        # Agregar start/end tokens - crear listas de Python
        pt_tokens = [vocab_pt] + pt_tokens + [vocab_pt + 1]
        en_tokens = [vocab_en] + en_tokens + [vocab_en + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        TensorFlow wrapper for the encode method
        """
        # Usar tf.py_function para convertir Python → TensorFlow
        pt_encoded, en_encoded = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )

        # Definir shapes (longitudes variables)
        pt_encoded.set_shape([None])
        en_encoded.set_shape([None])

        return pt_encoded, en_encoded
