#!/usr/bin/env python3
"""
0-dataset.py
"""
import tensorflow_datasets as tfds
import tensorflow as tf
import transformers


class Dataset:
    def __init__(self, batch_size, max_len):
        """
        Initialize dataset with data pipeline optimizations
        """
        self.batch_size = batch_size
        self.max_len = max_len
        
        # Load dataset
        dataset, _ = tfds.load('ted_hrlr_translate/pt_to_en',
                             as_supervised=True,
                             with_info=True)
        
        # Store raw datasets
        self.raw_data_train = dataset['train']
        self.raw_data_valid = dataset['validation']
        
        # Create tokenizers
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.raw_data_train)
        
        # Build optimized data pipelines
        self.data_train = self._build_train_pipeline()
        self.data_valid = self._build_validation_pipeline()

    def tokenize_dataset(self, data):
        """Create sub-word tokenizers for the dataset"""
        pt_texts, en_texts = [], []
        for pt, en in data:
            pt_texts.append(pt.numpy().decode('utf-8'))
            en_texts.append(en.numpy().decode('utf-8'))
        
        # Load and train tokenizers
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
        tokenizer_en = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
        
        tokenizer_pt = tokenizer_pt.train_new_from_iterator(pt_texts, vocab_size=2**13)
        tokenizer_en = tokenizer_en.train_new_from_iterator(en_texts, vocab_size=2**13)
        
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """Encode translation into tokens with start and end tokens"""
        pt_text = pt.numpy().decode('utf-8')
        en_text = en.numpy().decode('utf-8')
        
        pt_tokens = self.tokenizer_pt.encode(pt_text)
        en_tokens = self.tokenizer_en.encode(en_text)
        
        vocab_pt = self.tokenizer_pt.vocab_size
        vocab_en = self.tokenizer_en.vocab_size
        
        pt_tokens = [vocab_pt] + pt_tokens + [vocab_pt + 1]
        en_tokens = [vocab_en] + en_tokens + [vocab_en + 1]
        
        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """TensorFlow wrapper for encode method"""
        pt_encoded, en_encoded = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]  # ← Esto produce tensores int64
        )
        
        pt_encoded.set_shape([None])
        en_encoded.set_shape([None])
        
        return pt_encoded, en_encoded

    def _filter_max_length(self, pt, en):
        """Filter examples longer than max_len"""
        return tf.logical_and(
            tf.size(pt) <= self.max_len,
            tf.size(en) <= self.max_len
        )

    def _build_train_pipeline(self):
        """Build optimized training pipeline"""
        # 🔥 CORRECCIÓN: Usar tf.cast para convertir a int64
        padding_value = tf.cast(0, tf.int64)
        
        pipeline = (self.raw_data_train
                   .map(self.tf_encode, num_parallel_calls=tf.data.AUTOTUNE)
                   .filter(self._filter_max_length)
                   .cache()
                   .shuffle(buffer_size=20000)
                   .padded_batch(
                       batch_size=self.batch_size,
                       padded_shapes=([None], [None]),
                       padding_values=(padding_value, padding_value)  # ← Ahora son int64
                   )
                   .prefetch(tf.data.AUTOTUNE)
        )
        return pipeline

    def _build_validation_pipeline(self):
        """Build validation pipeline"""
        # 🔥 CORRECCIÓN: Usar tf.cast para convertir a int64
        padding_value = tf.cast(0, tf.int64)
        
        pipeline = (self.raw_data_valid
                   .map(self.tf_encode, num_parallel_calls=tf.data.AUTOTUNE)
                   .filter(self._filter_max_length)
                   .padded_batch(
                       batch_size=self.batch_size,
                       padded_shapes=([None], [None]),
                       padding_values=(padding_value, padding_value)  # ← Ahora son int64
                   )
        )
        return pipeline