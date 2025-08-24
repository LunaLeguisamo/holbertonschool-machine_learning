#!/usr/bin/env python3
"""
Function that creates a variational autoencoder
"""
import tensorflow.keras as keras
import tensorflow as tf


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    input_dims: integer, dimensions of the model input
    hidden_layers: list, number of nodes for each hidden layer in the encoder
    latent_dims: integer, dimensions of the latent space representation
    Returns: encoder, decoder, auto
    """
    # -------- ENCODER --------
    encoder_input = keras.layers.Input(shape=(input_dims,))
    x = encoder_input

    # Hidden layers
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation="relu")(x)

    # Latent variables: mean and log variance
    mu = keras.layers.Dense(latent_dims, activation=None)(x)
    log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    # Reparameterization trick
    def sampling(args):
        mu, log_var = args
        epsilon = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(log_var / 2) * epsilon

    z = keras.layers.Lambda(sampling)([mu, log_var])

    encoder = keras.Model(encoder_input, [z, mu, log_var], name="encoder")

    # -------- DECODER --------
    decoder_input = keras.layers.Input(shape=(latent_dims,))
    y = decoder_input

    for units in reversed(hidden_layers):
        y = keras.layers.Dense(units, activation="relu")(y)

    decoder_output = keras.layers.Dense(input_dims, activation="sigmoid")(y)
    decoder = keras.Model(decoder_input, decoder_output, name="decoder")

    # -------- VAE AUTOENCODER --------
    auto_output = decoder(encoder(encoder_input)[0])
    auto = keras.Model(encoder_input, auto_output, name="autoencoder")

    # VAE loss = reconstruction + KL divergence
    reconstruction_loss = keras.losses.binary_crossentropy(encoder_input, auto_output)
    reconstruction_loss *= input_dims
    kl_loss = -0.5 * keras.backend.sum(1 + log_var - keras.backend.square(mu) - keras.backend.exp(log_var), axis=1)
    vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)

    auto.add_loss(vae_loss)
    auto.compile(optimizer="adam")

    return encoder, decoder, auto
