import tensorflow as tf
from typing import List, Tuple

def MhAttnRnn(
    input_shape,
    num_classes: int,
    proj_dim: int,
    num_heads: int = 8,
    cnn_channels: List[int] = [16, 16],
    cnn_kernel_size: Tuple[int, int] = (3,3),
    rnn_units: int = 64,
    rnn_layers: int = 2,
    rnn_type: str = "lstm",
    dropout: float = 0.0,
    kernel_regularizer=None,
):
    input_audio = tf.keras.layers.Input(
    shape=input_shape,
    )
    net = input_audio

    """
    The input shape will be [batch, time, frequencies] to make it compatible with the CNN layer,
    we should expand the dimensions to [batch, time, frequencies, 1]
    """
    net = tf.expand_dims(net, axis=-1)

    # Pass it through the cnn layers
    for i, filters in enumerate(cnn_channels):
        net = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=cnn_kernel_size,
            strides=(1, 1),
            padding="same",
            activation="relu",
            kernel_regularizer=kernel_regularizer,
        )(net)
        net = tf.keras.layers.BatchNormalization()(net)

    shape = net.shape
    net = tf.keras.layers.Reshape((-1, shape[2] * shape[3]))(net)

    # dims [batch, time, features]
    if rnn_type == "lstm":
        rnn = tf.keras.layers.LSTM
    elif rnn_type == "gru":
        rnn = tf.keras.layers.GRU
    for _ in range(rnn_layers):
        # Build Bi-directional connections
        net = tf.keras.layers.Bidirectional(
            rnn(
                units=rnn_units,
                return_sequences=True,
                unroll=True,
                kernel_regularizer=kernel_regularizer,
                dropout=dropout,
            )
        )(net)
