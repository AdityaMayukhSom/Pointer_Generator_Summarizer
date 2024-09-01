import numpy as np
import keras
import tensorflow as tf


class Encoder(keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, enc_units: int, batch_sz: int):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = keras.layers.GRU(
            self.enc_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )

        forward_layer = keras.layers.LSTM(10, return_sequences=True, recurrent_initializer="glorot_uniform")
        backward_layer = keras.layers.LSTM(10, activation="relu", return_sequences=True, go_backwards=True, recurrent_initializer="glorot_uniform")
        self.bidirectional = keras.layers.Bidirectional(forward_layer, backward_layer=backward_layer)

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        initializer = keras.initializers.GlorotUniform()
        return initializer(shape=(self.batch_sz, self.enc_units))


class BahdanauAttention(keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = keras.layers.Dense(units)
        self.W2 = keras.layers.Dense(units)
        self.V = keras.layers.Dense(1)

    def call(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, tf.squeeze(attention_weights, -1)


class PositionalEmbedding(keras.layers.Layer):
    """
    A positional embedding layer combines the input embedding with a positional encoding that helps the Transformer
    to understand the relative position of the input tokens. This layer takes the input of tokens and converts them
    into sequence of embeddings vector. Then, it adds the positional encoding to the embeddings.

    Methods:
        compute_mask: Computes the mask to be applied to the embeddings.
        call: Performs the forward pass of the layer.
    """

    def positional_encoding(self, length: int, depth: int):
        """
        Generates a positional encoding for a given length and depth.

        Args:
            length (int): The length of the input sequence.
            depth (int): The depth that represents the dimensionality of the encoding.

        Returns:
            tf.Tensor: The positional encoding of shape (length, depth).
        """
        depth = depth // 2

        positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
        depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

        angle_rates = 1 / (10000**depths)  # (1, depth)
        angle_rads = positions * angle_rates  # (pos, depth)

        pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)

        return tf.cast(pos_encoding, dtype=tf.float32)

    def __init__(self, vocab_size: int, d_model: int, embedding: keras.layers.Embedding | None = None):
        """Constructor of the PositionalEmbedding layer.

        Args:
            vocab_size (int): The size of the vocabulary. I. e. the number of unique tokens in the input sequence.
            d_model (int): The dimensionality of the embedding vector.
            embedding (tf.keras.layers.Embedding): The custom embedding layer. If None, a default embedding layer will be created.
        """
        super().__init__()
        self.d_model = d_model
        self.embedding = keras.layers.Embedding(vocab_size, d_model, mask_zero=True) if embedding is None else embedding
        self.pos_encoding = self.positional_encoding(length=2048, depth=d_model)

    def compute_mask(self, *args, **kwargs):
        """Computes the mask to be applied to the embeddings."""
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Performs the forward pass of the layer.

        Args:
            x (tf.Tensor): The input tensor of shape (batch_size, seq_length).

        Returns:
            tf.Tensor: The output sequence of embedding vectors with added positional information. The shape is
                (batch_size, seq_length, d_model).
        """
        x = self.embedding(x)
        length = tf.shape(x)[1]  # type: ignore
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]  # type: ignore
        return x


class MultiHeadAttentionBlock(keras.layers.Layer):
    @staticmethod
    def __attention(
        query: tf.Tensor, key: tf.Tensor, value: tf.Tensor, mask: tf.Tensor | None = None, dropout: keras.layers.Dropout | None = None
    ) -> tuple[tf.Tensor, tf.Tensor]:
        # returns the computed attention for further processing along with attention
        # scores for visualization of the impact of words in a sentence

        d_k = query.shape[-1]  # query is a tensor with shape (batch, h, seq_len, d_k)

        # (batch, h, seq_len, d_k) @ (batch, h, d_k, seq_len) [produced via transposing last 2 dimensions]
        # multiplication produces a tensor of dimensions (batch, h, seq_len, seq_len)
        attention_scores = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(tf.cast(d_k, tf.float32))

        if mask is not None:
            attention_scores += tf.multiply(mask, -1e9)

        attention_scores = tf.nn.softmax(attention_scores, axis=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        if not isinstance(attention_scores, tf.Tensor):
            raise ValueError("attention scores are not an instance of tf.Tensor")

        # (batch, h, seq_len, seq_len) @ (batch, h, seq_len, d_k) -> (batch, h, seq_len, d_k)
        H = tf.matmul(attention_scores, value)

        return H, attention_scores

    def __init__(self, d_model: int, h: int, dropout: float):
        super(MultiHeadAttentionBlock, self).__init__()

        self.d_model = d_model
        self.h = h

        # divide the embedding dimension into h equal dimensional heads
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h

        # all of the below layers do x*W^t + b, we can disable the bias via passing use_bias as False
        self.W_q = keras.layers.Dense(d_model, use_bias=False)
        self.W_k = keras.layers.Dense(d_model, use_bias=False)
        self.W_v = keras.layers.Dense(d_model, use_bias=False)
        self.W_o = keras.layers.Dense(d_model, use_bias=False)
        self.dropout = keras.layers.Dropout(dropout)

    def call(self, Q: tf.Tensor, K: tf.Tensor, V: tf.Tensor, mask: tf.Tensor | None = None):

        batch_size = tf.shape(query)[0]  # type: ignore
        seq_len = tf.shape(query)[1]  # type: ignore

        # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        query = self.W_q(Q)

        # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        key = self.W_k(K)

        # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        value = self.W_v(V)

        # remember that, d_model = d_k * h

        # (batch, seq_len, d_model) -> (batch, seq_len, h, d_k)
        query = tf.reshape(query, (batch_size, seq_len, self.h, self.d_k))
        # (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
        query = tf.transpose(query, perm=[0, 2, 1, 3])

        # (batch, seq_len, d_model) -> (batch, seq_len, h, d_k)
        key = tf.reshape(key, (batch_size, seq_len, self.h, self.d_k))
        # (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
        key = tf.transpose(key, perm=[0, 2, 1, 3])

        # (batch, seq_len, d_model) -> (batch, seq_len, h, d_k)
        value = tf.reshape(value, (batch_size, seq_len, self.h, self.d_k))
        # (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
        value = tf.transpose(value, perm=[0, 2, 1, 3])

        # dim(H) = (batch, h, seq_len, d_k), dim(attention_scores) = (batch, h, seq_len, seq_len)
        H, self_attention_scores = MultiHeadAttentionBlock.__attention(query, key, value, mask, self.dropout)

        # (batch, h, seq_len, d_k) -> (batch, seq_len, h, d_k) -> (batch, seq_len, d_model)
        H = tf.transpose(H, perm=[0, 2, 1, 3])
        H = tf.reshape(H, (batch_size, seq_len, self.d_model))

        # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        MH_A = self.W_o(H)

        return MH_A


class BaseAttention(keras.layers.Layer):
    """
    Base class for all attention layers. It contains the common functionality of all attention layers.
    This layer contains a MultiHeadAttention layer, a LayerNormalization layer and an Add layer.
    It is used as a base class for the GlobalSelfAttention, CausalSelfAttention and CrossAttention layers.
    And it is not intended to be used directly.

    Methods:
        call: Performs the forward pass of the layer.

    Attributes:
        mha (tf.keras.layers.MultiHeadAttention): The MultiHeadAttention layer.
        layernorm (tf.keras.layers.LayerNormalization): The LayerNormalization layer.
        add (tf.keras.layers.Add): The Add layer.
    """

    def __init__(self, **kwargs):
        """Constructor of the BaseAttention layer.

        Args:
            **kwargs: Additional keyword arguments that are passed to the MultiHeadAttention layer, e. g.
                        num_heads (number of heads), key_dim (dimensionality of the key space), etc.
        """
        super().__init__()
        self.mha = keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = keras.layers.LayerNormalization()
        self.add = keras.layers.Add()


class CrossAttention(BaseAttention):
    """
    A class that implements the cross-attention layer by inheriting from the BaseAttention class.
    This layer is used to process two different sequences and attends to the context sequence while processing the query sequence.

    Methods:
        call: Performs the forward pass of the layer.

    Attributes:
        mha (tf.keras.layers.MultiHeadAttention): The MultiHeadAttention layer.
        layernorm (tf.keras.layers.LayerNormalization): The LayerNormalization layer.
        add (tf.keras.layers.Add): The Add layer.
    """

    def call(self, x: tf.Tensor, context: tf.Tensor) -> tf.Tensor:
        """
        The call function that performs the cross-attention operation.

        Args:
            x (tf.Tensor): The query (expected Transformer results) sequence of shape (batch_size, seq_length, d_model).
            context (tf.Tensor): The context (inputs to the Encoder layer) sequence of shape (batch_size, seq_length, d_model).

        Returns:
            tf.Tensor: The output sequence of shape (batch_size, seq_length, d_model).
        """
        attn_output, attn_scores = self.mha(query=x, key=context, value=context, return_attention_scores=True)

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x


class CausalSelfAttention(BaseAttention):
    """
    Call self attention on the input sequence, ensuring that each position in the
    output depends only on previous positions (i.e. a causal model).

    Methods:
        call: Performs the forward pass of the layer.

    Attributes:
        mha (tf.keras.layers.MultiHeadAttention): The MultiHeadAttention layer.
        layernorm (tf.keras.layers.LayerNormalization): The LayerNormalization layer.
        add (tf.keras.layers.Add): The Add layer.
    """

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        The call function that performs the causal self-attention operation.

        Args:
            x (tf.Tensor): The input sequence of shape (batch_size, seq_length, d_model).

        Returns:
            tf.Tensor: The output sequence of shape (batch_size, seq_length, d_model).
        """
        attn_output = self.mha(query=x, value=x, key=x, use_causal_mask=True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class DecoderLayer(keras.layers.Layer):
    """
    A single layer of the Decoder. Usually there are multiple layers stacked on top of each other.

    Methods:
        call: Performs the forward pass of the layer.

    Attributes:
        causal_self_attention (CausalSelfAttention): The causal self-attention layer.
        cross_attention (CrossAttention): The cross-attention layer.
        ffn (FeedForward): The feed-forward layer.
    """

    def __init__(self, d_model: int, num_heads: int, dff: int, dropout_rate: float = 0.1):
        """
        Constructor of the DecoderLayer.

        Args:
            d_model (int): The dimensionality of the model.
            num_heads (int): The number of heads in the multi-head attention layer.
            dff (int): The dimensionality of the feed-forward layer.
            dropout_rate (float): The dropout rate.
        """
        super(DecoderLayer, self).__init__()

        self.causal_self_attention = CausalSelfAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout_rate)

        self.cross_attention = CrossAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def call(self, x: tf.Tensor, context: tf.Tensor) -> tf.Tensor:
        """
        The call function that performs the forward pass of the layer.

        Args:
            x (tf.Tensor): The input sequence of shape (batch_size, seq_length, d_model). x is usually the output of the previous decoder layer.
            context (tf.Tensor): The context sequence of shape (batch_size, seq_length, d_model). Context is usually the output of the encoder.
        """
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
        return x


class Decoder(keras.layers.Layer):
    """
    A custom TensorFlow layer that implements the Decoder. This layer is mostly used in the Transformer models
    for natural language processing tasks, such as machine translation, text summarization or text classification.

    Methods:
        call: Performs the forward pass of the layer.

    Attributes:
        d_model (int): The dimensionality of the model.
        num_layers (int): The number of layers in the decoder.
        pos_embedding (PositionalEmbedding): The positional embedding layer.
        dec_layers (list): The list of decoder layers.
        dropout (tf.keras.layers.Dropout): The dropout layer.
    """

    def __init__(self, num_layers: int, d_model: int, num_heads: int, dff: int, vocab_size: int, dropout_rate: float = 0.1):
        """
        Constructor of the Decoder.

        Args:
            num_layers (int): The number of layers in the decoder.
            d_model (int): The dimensionality of the model.
            num_heads (int): The number of heads in the multi-head attention layer.
            dff (int): The dimensionality of the feed-forward layer.
            vocab_size (int): The size of the vocabulary.
            dropout_rate (float): The dropout rate.
        """
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model)
        self.dropout = keras.layers.Dropout(dropout_rate)
        self.dec_layers = [DecoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate) for _ in range(num_layers)]

        self.last_attn_scores = None

    def call(self, x: tf.Tensor, context: tf.Tensor) -> tf.Tensor:
        """
        The call function that performs the forward pass of the layer.

        Args:
            x (tf.Tensor): The input sequence of shape (batch_size, target_seq_len).
            context (tf.Tensor): The context sequence of shape (batch_size, input_seq_len, d_model).
        """
        # `x` is token-IDs shape (batch, target_seq_len)
        x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context)

        self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        # The shape of x is (batch_size, target_seq_len, d_model).
        return x


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


class DecoderGRU(keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(DecoderGRU, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = keras.layers.GRU(
            self.dec_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )
        self.fc = keras.layers.Dense(vocab_size, activation=keras.activations.softmax)

    def call(self, x, hidden, enc_output, context_vector):
        # enc_output shape == (batch_size, max_length, hidden_size)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        out = self.fc(output)

        return x, out, state


class Pointer(keras.layers.Layer):
    def __init__(self):
        super(Pointer, self).__init__()
        self.w_s_reduce = keras.layers.Dense(1)
        self.w_i_reduce = keras.layers.Dense(1)
        self.w_c_reduce = keras.layers.Dense(1)

    def call(self, context_vector, state, dec_inp):
        return tf.nn.sigmoid(self.w_s_reduce(state) + self.w_c_reduce(context_vector) + self.w_i_reduce(dec_inp))
