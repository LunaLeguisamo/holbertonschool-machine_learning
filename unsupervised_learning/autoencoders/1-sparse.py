#!/usr/bin/env python3
"""
Function that creates a sparse autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    input_dims is an integer containing the dimensions of the model input
    hidden_layers is a list containing the number of nodes for each hidden
    layer in the encoder, respectively
    the hidden layers should be reversed for the decoder
    latent_dims is an integer containing the dimensions of the latent space
    representation
    lambtha is the regularization parameter used for L1 regularization on the
    encoded output
    Returns: encoder, decoder, auto
    encoder is the encoder model
    decoder is the decoder model
    auto is the sparse autoencoder model
    The sparse autoencoder model should be compiled using adam optimization and
    binary cross-entropy loss
    All layers should use a relu activation except for the last layer in the
    decoder, which should use sigmoid
    """
    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)
    latent = keras.layers.Dense(
        latent_dims,
        activation="relu",
        activity_regularizer=keras.regularizers.l1(lambtha)
        )(x)
    encoder = keras.Model(inputs=encoder_input, outputs=latent)

    decoder_input = keras.Input(shape=(latent_dims,))
    y = decoder_input
    for units in reversed(hidden_layers):
        y = keras.layers.Dense(units, activation="relu")(y)
    decoder_output = keras.layers.Dense(input_dims, activation="sigmoid")(y)
    decoder = keras.Model(inputs=decoder_input, outputs=decoder_output)

    # Autoencoder
    auto_output = decoder(encoder(encoder_input))
    auto = keras.Model(inputs=encoder_input, outputs=auto_output)

    # Compile with adam optimizer + binary crossentropy loss
    auto.compile(optimizer="adam", loss="binary_crossentropy")

    return encoder, decoder, auto
