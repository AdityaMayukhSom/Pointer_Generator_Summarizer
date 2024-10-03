import keras
import numpy as np
import tensorflow as tf


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

    def __init__(
        self,
        max_sentence_length: int,
        vocab_size: int,
        d_model: int,
        embedding: keras.layers.Embedding | None = None,
    ):
        """Constructor of the PositionalEmbedding layer.

        Args:
            vocab_size (int): The size of the vocabulary. I. e. the number of unique tokens in the input sequence.
            d_model (int): The dimensionality of the embedding vector.
            embedding (tf.keras.layers.Embedding): The custom embedding layer. If None, a default embedding layer will be created.
        """
        super().__init__()
        self.d_model = d_model
        self.embedding = (
            keras.layers.Embedding(vocab_size, d_model, mask_zero=False) if embedding is None else embedding
        )
        self.pos_encoding = self.positional_encoding(length=max_sentence_length, depth=d_model)

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
