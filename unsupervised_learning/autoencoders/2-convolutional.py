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
    
    # Build encoder
    encoder_inputs = keras.layers.Input(shape=input_dims)
    x = encoder_inputs
    
    # Track dimensions through the encoder
    current_height, current_width = input_dims[0], input_dims[1]
    
    # Add convolutional layers to encoder
    for f in filters:
        x = keras.layers.Conv2D(f, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        # Update dimensions after pooling (for same padding, dimensions are halved, rounded up)
        current_height = (current_height + 1) // 2
        current_width = (current_width + 1) // 2
    
    # Final convolution for latent space
    latent = keras.layers.Conv2D(latent_dims[2], (3, 3), activation='relu', padding='same')(x)
    
    # Create encoder model
    encoder = keras.models.Model(encoder_inputs, latent, name='encoder')
    
    # Build decoder
    decoder_inputs = keras.layers.Input(shape=latent_dims)
    x = decoder_inputs
    
    # Calculate how many upsampling steps we need
    # We need to get from latent_dims back to input_dims
    target_height, target_width = input_dims[0], input_dims[1]
    
    # Add convolutional layers with upsampling
    reversed_filters = list(reversed(filters))
    
    for i, f in enumerate(reversed_filters):
        x = keras.layers.Conv2D(f, (3, 3), activation='relu', padding='same')(x)
        
        # Only upsample if we're not at the target dimensions yet
        current_decoder_height = x.shape[1] if x.shape[1] is not None else latent_dims[0]
        current_decoder_width = x.shape[2] if x.shape[2] is not None else latent_dims[1]
        
        if current_decoder_height * 2 <= target_height and current_decoder_width * 2 <= target_width:
            x = keras.layers.UpSampling2D((2, 2))(x)
    
    # If we're still not at the target dimensions, add more upsampling
    while x.shape[1] is not None and x.shape[1] < target_height and x.shape[2] is not None and x.shape[2] < target_width:
        x = keras.layers.UpSampling2D((2, 2))(x)
    
    # Final convolution to reconstruct original image
    decoder_outputs = keras.layers.Conv2D(input_dims[-1], (3, 3), 
                                         activation='sigmoid', padding='same')(x)
    
    # Create decoder model
    decoder = keras.models.Model(decoder_inputs, decoder_outputs, name='decoder')
    
    # Build autoencoder
    autoencoder_outputs = decoder(encoder(encoder_inputs))
    auto = keras.models.Model(encoder_inputs, autoencoder_outputs, name='autoencoder')
    
    # Compile autoencoder
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    
    return encoder, decoder, auto
