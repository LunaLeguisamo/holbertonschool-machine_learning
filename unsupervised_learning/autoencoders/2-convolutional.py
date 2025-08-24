#!/usr/bin/env python3

"""
Function that creates a convolutional autoencoder:
"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    input_dims is a tuple of integers containing the dimensions
    of the model input
    filters is a list containing the number of filters for each
    convolutional layer in the encoder, respectively
    the filters should be reversed for the decoder
    latent_dims is a tuple of integers containing the dimensions
    of the latent space representation
    Each convolution in the encoder should use a kernel size of
    (3, 3) with same padding and relu activation, followed by max
    pooling of size (2, 2)
    Each convolution in the decoder, except for the last two, should
    use a filter size of (3, 3) with same padding and relu activation,
    followed by upsampling of size (2, 2)
    The second to last convolution should instead use valid padding
    The last convolution should have the same number of filters as the
    number of channels in input_dims with sigmoid activation and no upsampling
    Returns: encoder, decoder, auto
    encoder is the encoder model
    decoder is the decoder model
    auto is the full autoencoder model
    The autoencoder model should be compiled using adam optimization and
    binary cross-entropy loss
    """
    encoder_input = keras.Input(shape=input_dims)
    x = encoder_input

    for f in filters:
        x = keras.layers.Conv2D(f, (3, 3), activation="relu", padding="same")(x)
        x = keras.layers.MaxPooling2D((2, 2), padding="same")(x)

    # Flatten + Dense to latent space
    x = keras.layers.Flatten()(x)
    latent = keras.layers.Dense(
        units=int(keras.backend.prod(latent_dims)), activation="relu"
        )(x)
    latent = keras.layers.Reshape(latent_dims)(latent)

    encoder = keras.Model(encoder_input, latent, name="encoder")

    # -------- DECODER --------
    decoder_input = keras.Input(shape=latent_dims)
    y = decoder_input

    # Upsample con filtros invertidos menos el último
    for f in reversed(filters[:-1]):
        y = keras.layers.Conv2D(f, (3, 3), activation="relu", padding="same")(y)
        y = keras.layers.UpSampling2D((2, 2))(y)

    # Penúltima convolución con valid padding
    y = keras.layers.Conv2D(filters[0], (3, 3), activation="relu", padding="valid")(y)
    y = keras.layers.UpSampling2D((2, 2))(y)

    # Última capa → misma cantidad de canales que input
    decoder_output = keras.layers.Conv2D(
        input_dims[-1], (3, 3), activation="sigmoid", padding="same")(y)

    decoder = keras.Model(decoder_input, decoder_output, name="decoder")

    # -------- AUTOENCODER --------
    auto_output = decoder(encoder(encoder_input))
    auto = keras.Model(encoder_input, auto_output, name="autoencoder")

    # Compilar modelo
    auto.compile(optimizer="adam", loss="binary_crossentropy")

    return encoder, decoder, auto
