import keras
import tensorflow as tf


class FeedForward(keras.layers.Layer):
    """
    A class that implements the feed-forward layer.

    Methods:
        call: Performs the forward pass of the layer.

    Attributes:
        seq (tf.keras.Sequential): The sequential layer that contains the feed-forward layers. It applies the two feed-forward layers and the dropout layer.
        add (tf.keras.layers.Add): The Add layer.
        layer_norm (tf.keras.layers.LayerNormalization): The LayerNormalization layer.
    """

    def __init__(self, d_model: int, dff: int, dropout_rate: float = 0.1):
        """
        Constructor of the FeedForward layer.

        Args:
            d_model (int): The dimensionality of the model.
            dff (int): The dimensionality of the feed-forward layer.
            dropout_rate (float): The dropout rate.
        """
        super().__init__()
        self.seq = keras.Sequential(
            [
                keras.layers.Dense(dff, activation="relu"),
                keras.layers.Dense(d_model),
                keras.layers.Dropout(dropout_rate),
            ]
        )
        self.add = keras.layers.Add()
        self.layer_norm = keras.layers.LayerNormalization()

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        The call function that performs the feed-forward operation.

        Args:
            x (tf.Tensor): The input sequence of shape (batch_size, seq_length, d_model).

        Returns:
            tf.Tensor: The output sequence of shape (batch_size, seq_length, d_model).
        """
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x


class DecoderLayer(keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = keras.layers.MultiHeadAttention(num_heads, d_model, d_model, dropout=rate)
        self.mha2 = keras.layers.MultiHeadAttention(num_heads, d_model, d_model, dropout=rate)

        self.ffn = FeedForward(d_model, dff)

        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout = keras.layers.Dropout(rate)

    def call(self, x, enc_output, training: bool, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        # (batch_size, target_seq_len, d_model)
        attn1, attn_weights_block1 = self.mha1(
            query=x,
            key=x,
            value=x,
            attention_mask=look_ahead_mask,
            training=training,
            return_attention_scores=True,
        )
        out1 = self.layernorm1(attn1 + x)

        # (batch_size, target_seq_len, d_model)
        attn2, attn_weights_block2 = self.mha2(
            query=out1,
            key=enc_output,
            value=enc_output,
            attention_mask=padding_mask,
            training=training,
            return_attention_scores=True,
        )
        # (batch_size, target_seq_len, d_model)
        out2 = self.layernorm2(attn2 + out1)

        # (batch_size, target_seq_len, d_model)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout(ffn_output, training=training)
        # (batch_size, target_seq_len, d_model)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2


class Decoder(keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.depth = d_model // self.num_heads
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = keras.layers.Dropout(rate)
        self.Wh = keras.layers.Dense(1)
        self.Ws = keras.layers.Dense(1)
        self.Wx = keras.layers.Dense(1)
        self.V = keras.layers.Dense(1)

    def call(self, embed_x, enc_output, training, look_ahead_mask, padding_mask):

        attention_weights = {}
        out = self.dropout(embed_x, training=training)

        for i in range(self.num_layers):
            out, block1, block2 = self.dec_layers[i](
                out,
                enc_output,
                training=training,
                look_ahead_mask=look_ahead_mask,
                padding_mask=padding_mask,
            )

            attention_weights["decoder_layer{}_block1".format(i + 1)] = block1
            attention_weights["decoder_layer{}_block2".format(i + 1)] = block2

        # out.shape == (batch_size, target_seq_len, d_model)

        # context vectors
        enc_out_shape = tf.shape(enc_output)
        # shape : (batch_size, input_seq_len, num_heads, depth)
        context = tf.reshape(
            enc_output,
            (enc_out_shape[0], enc_out_shape[1], self.num_heads, self.depth),  # type: ignore
        )
        # (batch_size, num_heads, input_seq_len, depth)
        context = tf.transpose(context, [0, 2, 1, 3])
        # (batch_size, num_heads, 1, input_seq_len, depth)
        context = tf.expand_dims(context, axis=2)

        # (batch_size, num_heads, target_seq_len, input_seq_len, 1)
        attn = tf.expand_dims(block2, axis=-1)

        # (batch_size, num_heads, target_seq_len, input_seq_len, depth)
        context = context * attn
        # (batch_size, num_heads, target_seq_len, depth)
        context = tf.reduce_sum(context, axis=3)
        # (batch_size, target_seq_len, num_heads, depth)
        context = tf.transpose(context, [0, 2, 1, 3])
        # (batch_size, target_seq_len, d_model)
        context = tf.reshape(context, (tf.shape(context)[0], tf.shape(context)[1], self.d_model))  # type: ignore

        # P_gens computing
        a = self.Wx(embed_x)
        b = self.Ws(out)
        c = self.Wh(context)
        p_gens = tf.sigmoid(self.V(a + b + c))

        return out, attention_weights, p_gens
