#!/usr/bin/env python3
"""
Function that creates a convolutional autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    input_dims: tuple -> dimensions of the input
    filters: list -> number of filters for each Conv2D in encoder
    latent_dims: tuple -> dimensions of the latent space representation
    Returns: encoder, decoder, auto
    """
    # ----- ENCODER -----
    encoder_input = keras.Input(shape=input_dims)
    z = encoder_input

    for f in filters:
        z = keras.layers.Conv2D(f, (3, 3), activation="relu", padding="same")(z)
        z = keras.layers.MaxPooling2D((2, 2), padding="same")(z)

    encoder_output = z
    encoder = keras.Model(encoder_input, encoder_output, name="encoder")

    # ----- DECODER -----
    decoder_input = keras.Input(shape=latent_dims)
    z = decoder_input

    # Apply reversed filters except the first one
    for f in filters[::-1][1:]:
        z = keras.layers.Conv2D(f, (3, 3), activation="relu", padding="same")(z)
        z = keras.layers.UpSampling2D((2, 2))(z)

    # Penultimate convolution with valid padding
    z = keras.layers.Conv2D(filters[0], (3, 3), activation="relu", padding="same")(z)

    # Final reconstruction layer
    decoder_output = keras.layers.Conv2D(
        input_dims[-1], (3, 3), activation="sigmoid", padding="same"
    )(z)

    decoder = keras.Model(decoder_input, decoder_output, name="decoder")

    # ----- AUTOENCODER -----
    auto_input = keras.Input(shape=input_dims)
    encoded = encoder(auto_input)
    decoded = decoder(encoded)
    auto = keras.Model(auto_input, decoded, name="autoencoder")

    auto.compile(optimizer="adam", loss="binary_crossentropy")

    return encoder, decoder, auto
