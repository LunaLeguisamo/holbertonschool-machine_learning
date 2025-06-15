#!/usr/bin/env python3
"""Transfer learning with ResNet50 for CIFAR-10"""
from tensorflow import keras as K
import tensorflow as tf


def preprocess_data(X, Y):
    """Preprocess CIFAR-10 data for ResNet50"""
    X_p = K.applications.resnet50.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


def build_model():
    """Builds and returns the ResNet50-based model"""
    # Input layer for 32x32 CIFAR-10 images
    inputs = K.Input(shape=(32, 32, 3))

    # Resize to 224x224 for ResNet50
    resized = K.layers.Lambda(lambda x: tf.image.resize(x, (224, 224)))(inputs)

    # Load pre-trained ResNet50 without top
    base_model = K.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_tensor=resized,
        pooling='avg'
    )
    base_model.trainable = False  # Freeze initially

    # Add custom layers
    x = base_model.output
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Dense(256, activation='relu')(x)
    x = K.layers.Dropout(0.3)(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)

    return K.Model(inputs=inputs, outputs=outputs), base_model


def train_model():
    """Trains the model and saves it as cifar10.h5"""
    # Load and preprocess data
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    val_split = int(0.15 * len(X_train))
    X_val, Y_val = X_train[:val_split], Y_train[:val_split]
    X_train, Y_train = X_train[val_split:], Y_train[val_split:]

    X_train_p, Y_train_p = preprocess_data(X_train, Y_train)
    X_val_p, Y_val_p = preprocess_data(X_val, Y_val)
    X_test_p, Y_test_p = preprocess_data(X_test, Y_test)

    # Data augmentation
    datagen = K.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(X_train_p)

    # Build model
    model, base_model = build_model()

    # Compile
    model.compile(optimizer=K.optimizers.Adam(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Callbacks
    callbacks = [
        K.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        K.callbacks.ModelCheckpoint('cifar10.h5', save_best_only=True, monitor='val_accuracy', mode='max')
    ]

    # Train with frozen base
    model.fit(datagen.flow(X_train_p, Y_train_p, batch_size=28),
              validation_data=(X_val_p, Y_val_p),
              epochs=20,
              callbacks=callbacks,
              verbose=1)

    # Fine-tune last 30 layers of ResNet50
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # Re-compile with lower LR
    model.compile(optimizer=K.optimizers.Adam(1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(datagen.flow(X_train_p, Y_train_p, batch_size=64),
              validation_data=(X_val_p, Y_val_p),
              epochs=10,
              callbacks=callbacks,
              verbose=1)

    # Save final model
    model.save('cifar10.h5', save_format='h5')

    # Final evaluation
    loss, acc = model.evaluate(X_test_p, Y_test_p, verbose=1)
    print(f"Test Accuracy: {acc*100:.2f}%")


if __name__ == '__main__':
    train_model()
