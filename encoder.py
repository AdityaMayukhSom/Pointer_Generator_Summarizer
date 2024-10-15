import keras
import tensorflow as tf


class Encoder(keras.layers.Layer):
    # TODO: Positional Encoding Helps Recurrent Neural Networks Handle a Large Vocabulary
    # https://arxiv.org/pdf/2402.00236v1
    def __init__(self, enc_units: int, batch_sz: int, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.batch_sz = batch_sz
        self.enc_units = enc_units

        self.forward_layer = keras.layers.LSTM(
            enc_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
            activation="sigmoid",
            go_backwards=False,
        )

        backward_layer = keras.layers.LSTM(
            enc_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
            activation="sigmoid",
            go_backwards=True,
        )

        self.bidirectional = keras.layers.Bidirectional(
            self.forward_layer, backward_layer=backward_layer, merge_mode="ave"
        )

        # self.gru = keras.layers.GRU(
        #     self.enc_units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform"
        # )

    def compute_mask(self, inputs, mask=None):
        # Just pass the received mask from previous layer, to the next layer or
        # manipulate it if this layer changes the shape of the input
        return mask

    def call(self, x, hidden):
        # https://github.com/keras-team/keras/issues/19754
        # https://github.com/keras-team/keras/pull/19789

        output, forward_h, forward_c, backward_h, backward_c = self.bidirectional(x)
        return output, forward_h, forward_c, backward_h, backward_c
        # output, forward_h = self.gru(x, initial_state=hidden)
        # return output, forward_h

    def initialize_hidden_state(self):
        initializer = keras.initializers.GlorotUniform()

        # multiply with 2 in case of bidirectional RNN as it will be split into
        # two, half to forward layer, half to backward layer
        return initializer(shape=(2 * self.batch_sz, self.enc_units))

        # return initializer(shape=(self.batch_sz, self.enc_units))


class EncoderReducer(keras.layers.Layer):
    """
    Reduces the Bi-LSTM outputs to a unidirectional LSTM output space.
    """

    def __init__(self, units: int, **kwargs):
        super(EncoderReducer, self).__init__(**kwargs)
        self.c_reducer = keras.layers.Dense(units, activation="relu")
        self.h_recuder = keras.layers.Dense(units, activation="relu")

    def call(
        self,
        forward_c: tf.Tensor,
        forward_h: tf.Tensor,
        backward_c: tf.Tensor,
        backward_h: tf.Tensor,
    ):
        old_c = tf.concat([forward_c, backward_c], axis=1)
        old_h = tf.concat([forward_h, backward_h], axis=1)

        new_c = self.c_reducer(old_c)
        new_h = self.h_recuder(old_h)

        return new_c, new_h
