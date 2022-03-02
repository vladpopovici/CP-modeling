#
# TRAINERS: various training functions for deep models
#
import tensorflow as tf
import numpy as np
from cpm.metrics import mean_iou


def train_basic(model: tf.keras.Model,
                X_train: np.array, Y_train: np.array,
                checkpoints_path: str="model_checkpoints.h5",
                batch_size: int=128, epochs: int=100,
                validation_split: float=0.1) -> tuple[tf.keras.callbacks.History, tf.keras.Model]:
    """
    A simple training procedure with early stop and checkpoints.

    Args:
        model (keras.Model)
        X_train, Y_train (numpy.array)
        batch_size (int)
        epochs (int)
        validation_split (float)

    Returns:
        a trained model

    """
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
    model.summary()

    # Fit model
    es  = tf.keras.callbacks.EarlyStopping(patience=5, verbose=1)
    ckp = tf.keras.callbacks.ModelCheckpoint(
        checkpoints_path, save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
      save_best_only=True
    )
    pb  = tf.keras.callbacks.ProgbarLogger(count_mode='steps')

    trace = model.fit(X_train,
                      Y_train,
                      validation_split=validation_split,
                      batch_size=batch_size,
                      epochs=epochs,
                      callbacks=[es, ckp, pb])

    return trace, model


def train_basic_generator(model: tf.keras.Model,
                train_data_generator: tf.keras.preprocessing.image.ImageDataGenerator,
                valid_data_generator: tf.keras.preprocessing.image.ImageDataGenerator,
                checkpoints_path: str="model_checkpoints.h5",
                batch_size: int=128, epochs: int=100) -> tuple[tf.keras.callbacks.History, tf.keras.Model]:
    """
    A simple training procedure with early stop and checkpoints.

    Args:
        model (keras.Model)
        train_data_generator, valid_data_generator
        epochs (int)
        validation_split (float)

    Returns:
        a trained model

    """
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
    model.summary()

    # Fit model
    es  = tf.keras.callbacks.EarlyStopping(patience=5, verbose=1)
    ckp = tf.keras.callbacks.ModelCheckpoint(
        checkpoints_path, save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
      save_best_only=True
    )
    pb  = tf.keras.callbacks.ProgbarLogger(count_mode='steps')

    trace = model.fit_generator(
        train_data_generator,
        validation_data=valid_data_generator,
        steps_per_epoch=train_data_generator.sample_size // batch_size,
        validation_steps=valid_data_generator.sample_size // batch_size,
        max_queue_size=batch_size,
        epochs=epochs,
        callbacks=[es, ckp, pb]
    )

    return trace, model
