import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, initializers
from tensorflow.keras.regularizers import L2
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.layers import Dropout, BatchNormalization, Dense, LeakyReLU, Input


def get_stem(x, filter_number, seed_value, regularization_factor, batch_norm, dropout_rate):
    x = Conv1D(filters=filter_number,
               kernel_size=(3),
               strides=2,
               kernel_regularizer=L2(regularization_factor),
               kernel_initializer=initializers.HeNormal(seed_value),
               activation=LeakyReLU())(x)
    # x = Dropout(dropout_rate)(x)
    if batch_norm:
        x = BatchNormalization()(x)

    x = Conv1D(filters=filter_number,
               kernel_size=(3),
               strides=1,
               kernel_regularizer=L2(regularization_factor),
               kernel_initializer=initializers.HeNormal(seed_value),
               activation=LeakyReLU())(x)
    # x = Dropout(dropout_rate)(x)
    if batch_norm:
        x = BatchNormalization()(x)

    x = Conv1D(filters=filter_number * 2,
               kernel_size=(3),
               strides=1,
               kernel_regularizer=L2(regularization_factor),
               kernel_initializer=initializers.HeNormal(seed_value),
               activation=LeakyReLU())(x)
    # x = Dropout(dropout_rate)(x)
    if batch_norm:
        x = BatchNormalization()(x)

    return x


def get_35x35_module(x, filter_number, seed_value, regularization_factor, batch_norm, dropout_rate):
    # First line
    l1_c1 = MaxPooling1D()(x)
    l1_c2 = Conv1D(filters=filter_number * 1,
                   kernel_size=(1),
                   strides=2,
                   kernel_regularizer=L2(regularization_factor),
                   kernel_initializer=initializers.HeNormal(seed_value),
                   activation=LeakyReLU())(x)
    # l1_c2 = Dropout(dropout_rate)(l1_c2)
    if batch_norm:
        l1_c1 = BatchNormalization()(l1_c1)
    l1_c3 = Conv1D(filters=filter_number * 1,
                   kernel_size=(1),
                   strides=2,
                   kernel_regularizer=L2(regularization_factor),
                   kernel_initializer=initializers.HeNormal(seed_value),
                   activation=LeakyReLU())(x)
    # l1_c3 = Dropout(dropout_rate)(l1_c3)
    if batch_norm:
        l1_c2 = BatchNormalization()(l1_c2)

    # Second line
    l2_c1 = Conv1D(filters=filter_number * 2,
                   kernel_size=(1),
                   strides=2,
                   kernel_regularizer=L2(regularization_factor),
                   kernel_initializer=initializers.HeNormal(seed_value),
                   activation=LeakyReLU())(l1_c1)
    # l2_c1 = Dropout(dropout_rate)(l2_c1)
    if batch_norm:
        l2_c1 = BatchNormalization()(l2_c1)
    l2_c2 = Conv1D(filters=filter_number * 2,
                   kernel_size=(3),
                   strides=1,
                   kernel_regularizer=L2(regularization_factor),
                   kernel_initializer=initializers.HeNormal(seed_value),
                   activation=LeakyReLU())(l1_c2)
    # l2_c2 = Dropout(dropout_rate)(l2_c2)
    if batch_norm:
        l2_c2 = BatchNormalization()(l2_c2)
    l2_c3 = Conv1D(filters=filter_number * 2,
                   kernel_size=(3),
                   strides=2,
                   kernel_regularizer=L2(regularization_factor),
                   kernel_initializer=initializers.HeNormal(seed_value),
                   activation=LeakyReLU())(l1_c3)
    # l2_c3 = Dropout(dropout_rate)(l2_c3)
    if batch_norm:
        l2_c3 = BatchNormalization()(l2_c3)
    l2_c4 = Conv1D(filters=filter_number * 2,
                   kernel_size=(1),
                   strides=2,
                   kernel_regularizer=L2(regularization_factor),
                   kernel_initializer=initializers.HeNormal(seed_value),
                   activation=LeakyReLU())(x)
    # l2_c4 = Dropout(dropout_rate)(l2_c4)
    if batch_norm:
        l2_c4 = BatchNormalization()(l2_c4)

    # Third line
    l3_c3 = Conv1D(filters=filter_number * 2,
                   kernel_size=(3),
                   strides=2,
                   kernel_regularizer=L2(regularization_factor),
                   kernel_initializer=initializers.HeNormal(seed_value),
                   activation=LeakyReLU())(l2_c3)
    l3_c3 = Dropout(dropout_rate)(l3_c3)

    return tf.keras.layers.concatenate([l2_c1, l2_c2, l3_c3, l2_c4], axis=1)


def get_ipa_model(input_spectra: np.array,
                  seed_value: int,
                  regularization_factor: float,
                  dropout_rate: float,
                  model_name: str,
                  filter_number: int = 16,
                  batch_norm: bool = False):
    x = get_stem(input_spectra, filter_number, seed_value, regularization_factor, batch_norm, dropout_rate=.0)
    x = get_35x35_module(x, filter_number * 2, seed_value, regularization_factor, batch_norm, dropout_rate=.0)

    x = Flatten()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(1, kernel_initializer=initializers.HeNormal(seed_value), activation=LeakyReLU())(x)

    return Model(inputs=[input_spectra], outputs=[x], name=model_name)
