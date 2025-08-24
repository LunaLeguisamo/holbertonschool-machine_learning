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
    # -------- ENCODER --------
    encoder_input = keras.layers.Input(shape=input_dims)
    x = encoder_input

    for f in filters:
        x = keras.layers.Conv2D(f, (3, 3), activation="relu", padding="same")(x)
        x = keras.layers.MaxPooling2D((2, 2), padding="same")(x)

    latent = x
    encoder = keras.Model(encoder_input, latent, name="encoder")

    # -------- DECODER --------
    decoder_input = keras.layers.Input(shape=latent_dims)
    y = decoder_input

    # Invertimos los filtros menos el último
    for f in reversed(filters[:-1]):
        y = keras.layers.Conv2D(f, (3, 3), activation="relu", padding="same")(y)
        y = keras.layers.UpSampling2D((2, 2))(y)

    # Penúltima convolución con padding="valid"
    y = keras.layers.Conv2D(filters[0], (3, 3), activation="relu", padding="valid")(y)
    y = keras.layers.UpSampling2D((2, 2))(y)

    # Última capa: misma cantidad de canales que input
    decoder_output = keras.layers.Conv2D(
        input_dims[-1], (3, 3), activation="sigmoid", padding="same"
    )(y)

    decoder = keras.Model(decoder_input, decoder_output, name="decoder")

    # -------- AUTOENCODER --------
    auto_output = decoder(encoder(encoder_input))
    auto = keras.Model(encoder_input, auto_output, name="autoencoder")

    # Compilar
    auto.compile(optimizer="adam", loss="binary_crossentropy")

    return encoder, decoder, auto
