#!/usr/bin/env python3
"""
1-wgan_clip.py
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_clip(keras.Model):
    """
    Implementation of a Wasserstein GAN with weight clipping.

    This class manages the generator and discriminator (critic),
    defines their custom loss functions according to the WGAN
    formulation, and performs the training step with weight clipping.
    """

    def __init__(
        self, generator, discriminator,
        latent_generator, real_examples,
        batch_size=200, disc_iter=2, learning_rate=0.005
    ):
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = 0.5
        self.beta_2 = 0.9

        # --- Generator loss & optimizer ---
        # Como el discr. devuelve tanh, invertimos con atanh.
        self.generator.loss = (
            lambda fake_pred: -tf.reduce_mean(
                tf.atanh(tf.clip_by_value(fake_pred, -0.999999, 0.999999))
            )
        )
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
        )
        self.generator.compile(
            optimizer=self.generator.optimizer, loss=self.generator.loss
        )

        # --- Discriminator loss & optimizer ---
        self.discriminator.loss = (
            lambda real_pred, fake_pred:
                tf.reduce_mean(tf.atanh(tf.clip_by_value(real_pred, -0.999999, 0.999999)))
                - tf.reduce_mean(tf.atanh(tf.clip_by_value(fake_pred, -0.999999, 0.999999)))
        )
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
        )
        self.discriminator.compile(
            optimizer=self.discriminator.optimizer, loss=self.discriminator.loss
        )

    def get_fake_sample(self, size=None, training=False):
        if not size:
            size = self.batch_size
        return self.generator(
            self.latent_generator(size), training=training
        )

    def get_real_sample(self, size=None):
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def train_step(self, useless_argument):
        # --- Train Discriminator ---
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                real_sample = self.get_real_sample()
                fake_sample = self.get_fake_sample(training=True)

                real_pred = self.discriminator(real_sample, training=True)
                fake_pred = self.discriminator(fake_sample, training=True)

                discr_loss = self.discriminator.loss(real_pred, fake_pred)

            grads = tape.gradient(
                discr_loss, self.discriminator.trainable_variables
            )
            self.discriminator.optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_variables)
            )

            # clip de pesos del discriminador en [-1, 1]
            for var in self.discriminator.trainable_variables:
                var.assign(tf.clip_by_value(var, -1.0, 1.0))

        # --- Train Generator ---
        with tf.GradientTape() as tape:
            fake_sample = self.get_fake_sample(training=True)
            fake_pred = self.discriminator(fake_sample, training=True)
            gen_loss = self.generator.loss(fake_pred)

        grads = tape.gradient(
            gen_loss, self.generator.trainable_variables
        )
        self.generator.optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables)
        )

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
